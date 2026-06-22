"""PipelineDataLoader: wraps dataset, calls model.prepare_inputs, splits into micro-batches."""

from __future__ import annotations

import queue
import threading

import torch

from rengu_flow import distributed as dist

_SENTINEL = object()


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

    def __init__(
        self,
        dataset,
        model_engine,
        gradient_accumulation_steps,
        model,
        num_dataloader_workers: int = 0,
        dataloader_prefetch: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ):
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
        self.dataloader_prefetch = dataloader_prefetch
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.iter_called = False
        self.eval_quantile = None
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.recreate_dataloader = False
        # Opt-in (main.py, STATIC compile runs): announce the first batch of each
        # new latent shape so the per-shape compile stall is explained in the log.
        # (Under compile_dynamic shapes share one graph per micro-batch signature,
        # so per-shape announces would be noise — main leaves this off there.)
        self.announce_new_shapes = False
        self._seen_latent_shapes: set = set()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_error: list[BaseException] = []
        self._create_dataloader()
        self.data = self._pull_batches_from_dataloader()

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def reset(self):
        """Reset loader for reuse (e.g. between eval quantiles)."""
        self._stop_prefetch_thread()
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self._create_dataloader()
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
        self._maybe_announce_shape(ret)
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            self._stop_prefetch_thread()
            self.epoch += 1
            # Advance the dataset's epoch before re-creating/re-iterating so a non-static
            # max_images rotates its per-epoch window for the new epoch.
            self._refresh_dataset_epoch()
            if self.recreate_dataloader or self._rotation_needs_worker_refresh():
                self._create_dataloader()
                self.recreate_dataloader = False
            self.data = self._pull_batches_from_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = None
        return ret

    def _maybe_announce_shape(self, micro_batch) -> None:
        """Print a one-time heads-up when a not-yet-seen latent shape is about to be fed.

        With compile on, the first step on each shape compiles its kernels — a
        one-time stall that can look like a hang. Announced here (data side)
        because the loader sees the shape *before* the stall starts.
        """
        if not self.announce_new_shapes:
            return
        try:
            features = micro_batch[0]
            latents = features[0] if isinstance(features, (tuple, list)) else features
            shape = tuple(latents.shape)
        except Exception:
            return
        key = shape[1:]  # ignore micro-batch dim; (C, T, H, W) identifies the graph
        if key in self._seen_latent_shapes:
            return
        self._seen_latent_shapes.add(key)
        print(
            f"[compile] new latent shape {'x'.join(str(d) for d in shape)} "
            f"({len(self._seen_latent_shapes)} seen) — first step on this shape "
            "compiles its kernels: expect a one-time slow step (disk-cached for "
            "later runs), not a hang.",
            flush=True,
        )

    def _refresh_dataset_epoch(self) -> None:
        """Tell the dataset which epoch we are on (no-op for datasets without set_epoch)."""
        set_epoch = getattr(self.dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.epoch)

    def _rotation_needs_worker_refresh(self) -> bool:
        """Persistent/forked workers hold a stale dataset copy; re-fork them when rotating."""
        return self.num_dataloader_workers > 0 and getattr(
            self.dataset, "rotation_active", False
        )

    def refresh_for_step(self, step: int) -> None:
        """Step-accurate resolution-schedule hook. Call once per optimizer step,
        before pulling that step's micro-batches. If the step crossed a stage
        boundary, the dataset rebuilds its iteration order and we restart iteration
        (mid-epoch) so the new resolution(s) take effect immediately — even when a
        single-resolution epoch spans more steps than a stage."""
        update = getattr(self.dataset, "update_active_stage", None)
        if callable(update) and update(step):
            self._restart_iteration()

    def _restart_iteration(self) -> None:
        """Discard the current (old-stage) iterator and start fresh on the rebuilt
        dataset without advancing the epoch counter. Re-creating the DataLoader also
        re-forks any workers so they pick up the new iteration order."""
        self._stop_prefetch_thread()
        self._create_dataloader()
        self.data = self._pull_batches_from_dataloader()
        self.num_batches_pulled = 0
        self.next_micro_batch = None

    def _use_thread_prefetch(self) -> bool:
        return self.dataloader_prefetch and self.num_dataloader_workers == 0

    def _stop_prefetch_thread(self) -> None:
        if self._prefetch_queue is not None:
            try:
                self._prefetch_queue.put(_SENTINEL, block=False)
            except queue.Full:
                pass
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=30)
        self._prefetch_thread = None
        self._prefetch_queue = None

    def _create_dataloader(self, skip_first_n_batches=None):
        self._stop_prefetch_thread()
        # Set before constructing the DataLoader so forked workers inherit the right epoch.
        self._refresh_dataset_epoch()
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

        loader_kwargs: dict = {
            "batch_size": None,
            "sampler": sampler,
            "num_workers": self.num_dataloader_workers,
        }
        if self.num_dataloader_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        if self.pin_memory:
            loader_kwargs["pin_memory"] = True

        self.dataloader = torch.utils.data.DataLoader(self.dataset, **loader_kwargs)

    def _prepare_batch(self, batch):
        features, label = self.model.prepare_inputs(batch, timestep_quantile=self.eval_quantile)
        target, mask = label
        target = self._broadcast_target(target)
        label = (target, mask)
        self.num_batches_pulled += 1
        return split_batch((features, label), self.gradient_accumulation_steps)

    def _iter_raw_batches(self):
        if not self._use_thread_prefetch():
            yield from self.dataloader
            return

        self._prefetch_error = []
        self._prefetch_queue = queue.Queue(maxsize=2)

        def producer():
            try:
                for batch in self.dataloader:
                    self._prefetch_queue.put(batch)
            except BaseException as exc:
                self._prefetch_error.append(exc)
            finally:
                self._prefetch_queue.put(_SENTINEL)

        self._prefetch_thread = threading.Thread(target=producer, daemon=True)
        self._prefetch_thread.start()

        while True:
            item = self._prefetch_queue.get()
            if item is _SENTINEL:
                if self._prefetch_error:
                    raise self._prefetch_error[0]
                break
            yield item

    def _pull_batches_from_dataloader(self):
        for batch in self._iter_raw_batches():
            for micro_batch in self._prepare_batch(batch):
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
        if not dist.is_initialized():
            return  # single-process (engine='accelerate' / no backend): nothing to gather
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
