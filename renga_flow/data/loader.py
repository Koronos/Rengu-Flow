"""PipelineDataLoader: wraps dataset, calls model.prepare_inputs, splits into micro-batches."""

import torch

try:
    from deepspeed import comm as dist
except Exception:
    dist = None


def split_batch(batch, pieces: int) -> list:
    """Split (features, label) into micro-batches for gradient accumulation."""
    features, label = batch
    split_size = features[0].size(0) // pieces
    split_features = zip(
        *(
            torch.split(tensor, split_size) if tensor is not None else [torch.tensor([])] * pieces
            for tensor in features
        )
    )
    split_label = zip(
        *(
            torch.split(tensor, split_size) if tensor is not None else [torch.tensor([])] * pieces
            for tensor in label
        )
    )
    return list(zip(split_features, split_label))


class PipelineDataLoader:
    """Iterates over dataset, prepares inputs via model.prepare_inputs, yields micro-batches. Syncs epoch."""

    def __init__(self, dataset, model_engine, gradient_accumulation_steps, model, num_dataloader_workers=0):
        if len(dataset) == 0:
            msg = "Dataset is empty."
            if hasattr(dataset, "dataset_config"):
                msg += f" Config: {dataset.dataset_config}"
            raise RuntimeError(msg)
        self.model = model
        self.dataset = dataset
        self.model_engine = model_engine
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_dataloader_workers = num_dataloader_workers
        self.iter_called = False
        self.eval_quantile = None
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.recreate_dataloader = False
        self._create_dataloader()
        self.data = self._pull_batches_from_dataloader()

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def reset(self):
        """Reset loader for reuse (e.g. between eval quantiles)."""
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.data = self._pull_batches_from_dataloader()

    def __iter__(self):
        self.iter_called = True
        return self

    def __len__(self):
        return len(self.dataset) * self.gradient_accumulation_steps

    def __next__(self):
        if self.next_micro_batch is None:
            self.next_micro_batch = next(self.data)
        ret = self.next_micro_batch
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            if self.recreate_dataloader:
                self._create_dataloader()
                self.recreate_dataloader = False
            self.data = self._pull_batches_from_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = None
            self.epoch += 1
        return ret

    def _create_dataloader(self, skip_first_n_batches=None):
        if skip_first_n_batches is not None and skip_first_n_batches > 0:

            class SkipFirstN(torch.utils.data.Sampler):
                def __init__(self, n, length):
                    self.n = n
                    self.length = length

                def __len__(self):
                    return self.length

                def __iter__(self):
                    return iter(range(self.n, self.length))

            sampler = SkipFirstN(skip_first_n_batches, len(self.dataset))
        else:
            sampler = None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=self.num_dataloader_workers,
            persistent_workers=(self.num_dataloader_workers > 0),
        )

    def _pull_batches_from_dataloader(self):
        for batch in self.dataloader:
            features, label = self.model.prepare_inputs(batch, timestep_quantile=self.eval_quantile)
            target, mask = label
            target = self._broadcast_target(target)
            label = (target, mask)
            self.num_batches_pulled += 1
            for micro_batch in split_batch((features, label), self.gradient_accumulation_steps):
                yield micro_batch

    def _broadcast_target(self, target):
        if dist is None or not self.model_engine.is_pipe_parallel:
            return target
        if not (self.model_engine.is_first_stage() or self.model_engine.is_last_stage()):
            return target
        grid = self.model_engine.grid
        src_rank = grid.stage_to_global(0)
        dest_rank = grid.stage_to_global(self.model_engine.num_stages - 1)
        target = target.to("cuda")
        if self.model_engine.is_first_stage():
            dist.send(target, dest_rank)
        else:
            dist.recv(target, src_rank)
        return target

    def sync_epoch(self):
        if dist is None:
            return
        try:
            world_size = dist.get_world_size()
        except Exception:
            return
        result = [None] * world_size
        torch.distributed.all_gather_object(result, self.epoch)
        self.epoch = max(x for x in result if x is not None)

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "num_batches_pulled": self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
        assert not self.iter_called
        self.epoch = state_dict["epoch"]
        self.num_batches_pulled = state_dict["num_batches_pulled"]
        self._create_dataloader(skip_first_n_batches=self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()
        self.recreate_dataloader = True
