"""DatasetManager and _cache_fn: orchestrate latent + text embedding cache (from diffusion-pipe)."""

from __future__ import annotations

from collections import defaultdict
from inspect import signature

import torch
from torch import nn

try:
    import multiprocess as mp
except ImportError:
    import multiprocessing as mp  # type: ignore[no-redef]

from renga_flow.utils.common import is_main_process

try:
    from deepspeed import comm as dist
except Exception:
    dist = None


def _cache_fn(
    datasets_list,
    queue,
    preprocess_media_file_fn,
    num_text_encoders: int,
    regenerate_cache: bool,
    trust_cache: bool,
    caching_batch_size: int,
    cache_num_proc: int | None,
    cache_keep_in_memory: bool,
    cache_format: str,
) -> None:
    """Worker process: run cache_metadata, cache_latents, cache_text_embeddings; send GPU work via queue."""
    torch.set_num_threads(1)

    for ds in datasets_list:
        ds.cache_metadata(
            regenerate_cache=regenerate_cache,
            trust_cache=trust_cache,
            cache_num_proc=cache_num_proc,
        )

    pipes = {}

    def latents_map_fn(example, rank):
        is_edit = "control_file" in example
        first_size_bucket = example["size_bucket"][0]
        tensors_and_masks = []
        image_specs = []
        captions = []
        control_tensors_and_masks = []
        for i, (image_spec, mask_path, size_bucket, caption) in enumerate(
            zip(
                example["image_spec"],
                example["mask_file"],
                example["size_bucket"],
                example["caption"],
            )
        ):
            assert size_bucket == first_size_bucket
            items = preprocess_media_file_fn(
                image_spec, mask_path, size_bucket
            )
            tensors_and_masks.extend(items)
            image_specs.extend([image_spec] * len(items))
            captions.extend([caption] * len(items))
            if is_edit:
                control_file = example["control_file"][i]
                control_items = preprocess_media_file_fn(
                    (None, control_file), None, size_bucket
                )
                assert len(control_items) == 1 and len(items) == 1
                control_tensors_and_masks.append(control_items[0])
            else:
                control_tensors_and_masks.append(None)

        if len(tensors_and_masks) == 0:
            assert not is_edit
            return {
                "latents": [],
                "mask": [],
                "image_spec": [],
                "caption": [],
            }

        batch_size = len(example["image_spec"])
        results = defaultdict(list)
        for i in range(0, len(tensors_and_masks), batch_size):
            tensor = torch.stack(
                [t[0] for t in tensors_and_masks[i : i + batch_size]]
            )
            c_tensor = None
            if is_edit:
                c_tensor = torch.stack(
                    [
                        t[0]
                        for t in control_tensors_and_masks[i : i + batch_size]
                    ]
                )
            if rank not in pipes:
                pipes[rank] = mp.Pipe(duplex=False)
            parent_conn, child_conn = pipes[rank]
            queue.put((0, tensor, c_tensor, child_conn))
            result = parent_conn.recv()
            for k, v in result.items():
                results[k].append(v)
        for k in results:
            results[k] = torch.cat(results[k])
        results["image_spec"] = image_specs
        results["mask"] = [t[1] for t in tensors_and_masks]
        results["caption"] = captions
        return results

    for ds in datasets_list:
        ds.cache_latents(
            latents_map_fn,
            regenerate_cache=regenerate_cache,
            trust_cache=trust_cache,
            caching_batch_size=caching_batch_size,
            cache_num_proc=cache_num_proc,
            cache_keep_in_memory=cache_keep_in_memory,
            cache_format=cache_format,
        )

    for text_encoder_idx in range(num_text_encoders):
        def text_embedding_map_fn(example, rank):
            if rank not in pipes:
                pipes[rank] = mp.Pipe(duplex=False)
            parent_conn, child_conn = pipes[rank]
            control_file = example.get("control_file")
            queue.put(
                (
                    text_encoder_idx + 1,
                    example["caption"],
                    example["is_video"],
                    control_file,
                    child_conn,
                )
            )
            result = parent_conn.recv()
            result["image_spec"] = example["image_spec"]
            return result

        for ds in datasets_list:
            ds.cache_text_embeddings(
                text_embedding_map_fn,
                text_encoder_idx + 1,
                regenerate_cache=regenerate_cache,
                caching_batch_size=caching_batch_size,
                cache_num_proc=cache_num_proc,
                cache_keep_in_memory=cache_keep_in_memory,
                cache_format=cache_format,
            )

    queue.put(None)


class DatasetManager:
    """Registers train/eval datasets and runs latent + text embedding cache (VAE + TE on GPU)."""

    def __init__(
        self,
        model,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        self.model = model
        self.vae = model.get_vae()
        self.text_encoders = model.get_text_encoders()
        self.submodels = [self.vae] + list(self.text_encoders)
        self.call_vae_fn = model.get_call_vae_fn(self.vae)
        self.call_text_encoder_fns = [
            model.get_call_text_encoder_fn(te) for te in self.text_encoders
        ]
        self.te_fn_requires_control_file = [
            len(signature(fn).parameters) == 3
            for fn in self.call_text_encoder_fns
        ]
        self.regenerate_cache = regenerate_cache
        self.trust_cache = trust_cache
        self.caching_batch_size = caching_batch_size
        self.cache_format = cache_format
        self.cache_num_proc = cache_num_proc
        self.cache_keep_in_memory = cache_keep_in_memory
        self.datasets = []

    def register(self, dataset) -> None:
        self.datasets.append(dataset)

    def cache(self, unload_models: bool = True) -> None:
        if dist is None:
            raise RuntimeError(
                "DatasetManager.cache() requires distributed (e.g. deepspeed)."
            )
        if is_main_process():
            manager = mp.Manager()
            queue = [manager.Queue()]
        else:
            queue = [None]
        torch.distributed.broadcast_object_list(
            queue, src=0, group=dist.get_world_group()
        )
        queue = queue[0]

        if is_main_process():
            proc = mp.Process(
                target=_cache_fn,
                args=(
                    self.datasets,
                    queue,
                    self.model.get_preprocess_media_file_fn(),
                    len(self.text_encoders),
                    self.regenerate_cache,
                    self.trust_cache,
                    self.caching_batch_size,
                    self.cache_num_proc,
                    self.cache_keep_in_memory,
                    self.cache_format,
                ),
            )
            proc.start()

        while True:
            task = queue.get()
            if task is None:
                queue.put(None)
                break
            self._handle_task(task)

        if unload_models:
            model_name = getattr(self.model, "name", None)
            for i, submodel in enumerate(self.submodels):
                if not isinstance(submodel, nn.Module):
                    continue
                if model_name == "sdxl" and submodel is self.vae:
                    submodel.to("cpu")
                else:
                    submodel.to("meta")

        dist.barrier()
        if is_main_process():
            proc.join()

        for ds in self.datasets:
            ds.cache_metadata(trust_cache=True)
            ds.cache_latents(None, trust_cache=True)
            for i in range(1, len(self.text_encoders) + 1):
                ds.cache_text_embeddings(None, i)

    @torch.no_grad()
    def _handle_task(self, task) -> None:
        task_id = task[0]
        submodel = self.submodels[task_id]
        if isinstance(submodel, nn.Module):
            if next(submodel.parameters()).device.type != "cuda":
                for i, sm in enumerate(self.submodels):
                    if i != task_id and isinstance(sm, nn.Module):
                        sm.to("cpu")
                submodel.to("cuda")
        else:
            if hasattr(submodel, "load_model_if_needed"):
                submodel.load_model_if_needed()

        if task_id == 0:
            tensor, control_tensor, pipe = task[1:]
            if control_tensor is not None:
                results = self.call_vae_fn(tensor, control_tensor)
            else:
                results = self.call_vae_fn(tensor)
        elif task_id > 0:
            caption, is_video, control_file, pipe = task[1:]
            args = [caption, is_video]
            idx = task_id - 1
            if self.te_fn_requires_control_file[idx]:
                args.append(control_file)
            results = self.call_text_encoder_fns[idx](*args)
        else:
            raise RuntimeError("Invalid task id")

        cpu_results = {}
        for k, v in results.items():
            if isinstance(v, (list, tuple)):
                cpu_results[k] = [x.to("cpu") for x in v]
            else:
                cpu_results[k] = v.to("cpu")
        pipe.send(cpu_results)
