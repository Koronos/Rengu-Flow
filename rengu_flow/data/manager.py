"""DatasetManager and _cache_fn: orchestrate latent + text embedding cache (from diffusion-pipe)."""

from __future__ import annotations

import shutil
import sys
import threading
from collections import defaultdict
from inspect import signature
from pathlib import Path

import datasets as datasets_mod
import torch
from torch import nn

try:
    import multiprocess as mp
except ImportError:
    import multiprocessing as mp  # type: ignore[no-redef]

from rengu_flow.control.progress_stream import ProgressEmitter
from rengu_flow.data import caching_progress
from rengu_flow.data.cache_paths import caption_cache_key
from rengu_flow.distributed import is_main_process
from rengu_flow import distributed as dist



def _to_pipe(obj):
    """torch -> numpy for pipe transport.

    Unpickling a torch tensor LEAKS its full storage on torch 2.12 (pickle.loads
    retains ~tensor-size RSS per call; measured 200 loads -> +5.4 GB, never freed).
    Every cached text embedding crossing the GPU<->worker pipes leaked ~20 MB, eating
    ~60 GB of RAM over one caching pass. numpy arrays round-trip clean, so tensors
    travel as numpy (bf16 as a tagged uint16 view) and rebuild on the other side.
    """
    if torch.is_tensor(obj):
        t = obj.detach().to("cpu").contiguous()
        if t.dtype == torch.bfloat16:
            return ("__pipe_bf16__", t.view(torch.uint16).numpy())
        return t.numpy()
    if isinstance(obj, dict):
        return {k: _to_pipe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_pipe(v) for v in obj]
    return obj


def _from_pipe(obj):
    import numpy as np

    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__pipe_bf16__":
        return torch.from_numpy(obj[1]).view(torch.bfloat16)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: _from_pipe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_pipe(v) for v in obj]
    return obj


class _TextEmbeddingDedup:
    """Per-sample disk-backed memo for text-embedding caching.

    Embeddings spill to a temporary mmap Cache; RAM holds only
    ``(text_encoder_idx, caption_hash) -> row index``. The TE map runs on *batches*
    whose tensors are padded to the batch max, so rows are stored and looked up per
    sample — a full-batch hit returns per-sample lists (``unbatch_iter`` indexes lists
    and tensors alike, and the bucket cache pads per item). Thread-safe (the TE map
    runs on a thread pool)."""

    def __init__(self, spill_dir: Path) -> None:
        from rengu_flow.utils.cache import Cache

        self.dir = Path(spill_dir)
        shutil.rmtree(self.dir, ignore_errors=True)
        self.cache = Cache(self.dir, "te-dedup-spill")
        self.index: dict[tuple[int, str], int] = {}
        self.lock = threading.Lock()
        self.dirty = False

    def lookup(self, keys: list[tuple[int, str]]) -> dict | None:
        """Return the batch as per-sample lists when EVERY key is cached, else None."""
        with self.lock:
            idxs = [self.index.get(k) for k in keys]
            if any(i is None for i in idxs):
                return None
            if self.dirty:
                self.cache.refresh_reads()
                self.dirty = False
            rows = [self.cache[i] for i in idxs]
        return {
            key: [r[key] for r in rows] for key in rows[0] if rows[0][key] is not None
        }

    def store(self, keys: list[tuple[int, str]], result: dict) -> None:
        """Split a batched TE result into per-sample rows and cache the new ones."""
        with self.lock:
            for i, key in enumerate(keys):
                if key not in self.index:
                    self.cache.add(
                        {k: v[i] for k, v in result.items() if k != "image_spec"}
                    )
                    self.index[key] = self.cache.count - 1
                    self.dirty = True

    def close(self) -> None:
        self.cache.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def _count_latent_units(datasets_list) -> int:
    """Latent-encode buckets across all datasets: one unit per (size|ar-resolution) bucket."""
    total = 0
    for ds in datasets_list:
        for dd in getattr(ds, "directory_datasets", []):
            if dd.use_size_buckets:
                total += len(dd.size_bucket_datasets)
            else:
                total += sum(len(ar.resolutions) for ar in dd.ar_bucket_datasets)
    return total


def _count_te_units(datasets_list) -> int:
    """Text-embedding caches across all datasets: one unit per (size|ar) bucket."""
    total = 0
    for ds in datasets_list:
        for dd in getattr(ds, "directory_datasets", []):
            total += len(
                dd.size_bucket_datasets if dd.use_size_buckets else dd.ar_bucket_datasets
            )
    return total


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
    cache_dedup_text_embeddings: bool,
) -> None:
    """Worker process: run cache_metadata, cache_latents, cache_text_embeddings; send GPU work via queue."""
    torch.set_num_threads(1)
    # HF datasets renders its own tqdm bars ("Saving the dataset (x/y shards)", map descs).
    # In a captured log (web UI) they are pure noise between our phase lines; keep them on a
    # real terminal. The information itself stays: each step logs a [cache] line.
    if not sys.stderr.isatty():
        try:
            datasets_mod.disable_progress_bars()
        except Exception:
            pass
    from rengu_flow.utils.logging import tag_third_party_console_logs

    tag_third_party_console_logs()

    # One coordinator for every phase: unified "[cache] ..." log lines plus a single
    # monotonic progress bar (stage index + intra-stage fraction) instead of per-bucket
    # counters that made the UI bar bounce. Worker process is single-purpose: install
    # for its whole lifetime.
    progress = caching_progress.CachingProgress(
        emitter=ProgressEmitter() if is_main_process() else None
    )
    stage_names = ["metadata", "latents"] + [
        f"text embeddings {i + 1}" for i in range(num_text_encoders)
    ]
    progress.plan(stage_names)
    caching_progress.set_active(progress)

    dedup = None
    if cache_dedup_text_embeddings:
        dedup = _TextEmbeddingDedup(
            Path(datasets_list[0].directory_datasets[0].cache_dir) / "te_dedup_spill"
        )

    with progress.stage(
        "metadata", units=sum(len(ds.directory_datasets) for ds in datasets_list)
    ):
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
        control_tensors_and_masks = []
        # Captions are intentionally not read or stored here: a latent is shared across an
        # image's N captions, and the caption that reaches the model is resolved per
        # (image, caption_number) at sample time (see SizeBucketDataset._sample_from_entry).
        for i, (image_spec, mask_path, size_bucket) in enumerate(
            zip(
                example["image_spec"],
                example["mask_file"],
                example["size_bucket"],
            )
        ):
            assert size_bucket == first_size_bucket
            items = preprocess_media_file_fn(
                image_spec, mask_path, size_bucket
            )
            tensors_and_masks.extend(items)
            image_specs.extend([image_spec] * len(items))
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
                "valid": [],
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
            queue.put((0, _to_pipe(tensor), _to_pipe(c_tensor), child_conn))
            result = _from_pipe(parent_conn.recv())
            for k, v in result.items():
                results[k].append(v)
        for k in results:
            results[k] = torch.cat(results[k])
        results["image_spec"] = image_specs
        results["mask"] = [t[1] for t in tensors_and_masks]
        # Tombstone flag: a corrupt/truncated image yields a zero-placeholder latent
        # marked invalid here; it's filtered out when the iteration order is built, so
        # it's never sampled at train time — while the cache stays strictly 1:1.
        if is_edit:
            results["valid"] = [
                bool(t[2]) and bool(c[2])
                for t, c in zip(tensors_and_masks, control_tensors_and_masks)
            ]
        else:
            results["valid"] = [bool(t[2]) for t in tensors_and_masks]
        return results

    with progress.stage("latents", units=_count_latent_units(datasets_list)):
        for ds in datasets_list:
            ds.cache_latents(
                latents_map_fn,
                regenerate_cache=regenerate_cache,
                trust_cache=trust_cache,
                caching_batch_size=caching_batch_size,
                cache_num_proc=cache_num_proc,
                cache_keep_in_memory=cache_keep_in_memory,
            )

    for text_encoder_idx in range(num_text_encoders):
        def text_embedding_map_fn(example, rank):
            captions = example["caption"]
            # Key includes the encoder index: multi-encoder models (SDXL) produce a
            # different embedding per encoder for the same caption.
            keys = (
                [(text_encoder_idx, caption_cache_key(c)) for c in captions]
                if dedup is not None
                else None
            )
            if keys is not None:
                cached = dedup.lookup(keys)
                if cached is not None:
                    return {**cached, "image_spec": example["image_spec"]}
            if rank not in pipes:
                pipes[rank] = mp.Pipe(duplex=False)
            parent_conn, child_conn = pipes[rank]
            control_file = example.get("control_file")
            queue.put(
                (
                    text_encoder_idx + 1,
                    captions,
                    example["is_video"],
                    control_file,
                    child_conn,
                )
            )
            result = _from_pipe(parent_conn.recv())
            result["image_spec"] = example["image_spec"]
            if keys is not None:
                dedup.store(keys, result)
            return result

        with progress.stage(
            f"text embeddings {text_encoder_idx + 1}",
            units=_count_te_units(datasets_list),
        ):
            for ds in datasets_list:
                ds.cache_text_embeddings(
                    text_embedding_map_fn,
                    text_encoder_idx + 1,
                    regenerate_cache=regenerate_cache,
                    caching_batch_size=caching_batch_size,
                    cache_num_proc=cache_num_proc,
                    cache_keep_in_memory=cache_keep_in_memory,
                )

    if dedup is not None:
        # The spill only serves this run's caching: every embedding now lives in the
        # per-bucket caches, so drop the duplicate bytes.
        dedup.close()

    queue.put(None)


def _run_cache_worker(args, queue) -> None:
    """Run ``_cache_fn`` and, on any failure, signal the consumer instead of hanging it.

    ``_cache_fn`` enqueues ``None`` on success to end the consumer's drain loop. If it raises
    before that (e.g. an OSError while writing the Arrow cache on Windows), the consumer's
    blocking ``queue.get()`` would wait forever — a failure that looks identical to a hang.
    Surface it as an error sentinel so the main process raises with the real traceback.
    """
    try:
        _cache_fn(*args)
    except BaseException as exc:  # noqa: BLE001 - cross-thread/process error hand-off
        import traceback as _tb

        queue.put(("__cache_worker_error__", repr(exc), _tb.format_exc()))


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
        cache_dedup_text_embeddings: bool = False,
        backend=None,
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
        self.cache_num_proc = cache_num_proc
        self.cache_keep_in_memory = cache_keep_in_memory
        self.cache_dedup_text_embeddings = cache_dedup_text_embeddings
        self.backend = backend
        self.datasets = []

    def register(self, dataset) -> None:
        self.datasets.append(dataset)

    def cache(self, unload_models: bool = True) -> None:
        if dist is None:
            raise RuntimeError(
                "DatasetManager.cache() requires distributed (e.g. deepspeed)."
            )

        resolvers = [
            ds.get_augmentation_resolver()
            for ds in self.datasets
            if hasattr(ds, "get_augmentation_resolver")
        ]
        resolvers = [r for r in resolvers if r is not None]
        augmentation_resolver = resolvers[0] if resolvers else None

        worker = None
        queue = None
        if is_main_process():
            # Build cache_args with a queue placeholder at index 1 (the slot _cache_fn expects).
            # make_cache_worker creates the real queue; we patch it in before starting the worker.
            cache_args = [
                self.datasets,
                None,  # replaced with the real queue below
                self.model.get_preprocess_media_file_fn(
                    augmentation_resolver=augmentation_resolver
                ),
                len(self.text_encoders),
                self.regenerate_cache,
                self.trust_cache,
                self.caching_batch_size,
                self.cache_num_proc,
                self.cache_keep_in_memory,
                self.cache_dedup_text_embeddings,
            ]
            worker, queue = self.backend.make_cache_worker(_run_cache_worker, cache_args)
            cache_args[1] = queue  # inject the real queue so _cache_fn can enqueue GPU tasks
        if self.backend.is_distributed:
            qbox = [queue if is_main_process() else None]
            torch.distributed.broadcast_object_list(qbox, src=0, group=dist.get_world_group())
            queue = qbox[0]
        if worker is not None:
            worker.start()

        while True:
            task = queue.get()
            if task is None:
                queue.put(None)
                break
            if isinstance(task, tuple) and task and task[0] == "__cache_worker_error__":
                raise RuntimeError(
                    f"Dataset caching worker failed: {task[1]}\n{task[2]}"
                )
            self._handle_task(task)

        if unload_models:
            # The model decides which submodels must keep their weights on CPU (because save_model
            # reads them) vs. can go to meta to free RAM. Default: none. SDXL keeps the VAE always,
            # and all submodels for a full-model checkpoint. Keeps model-specific save semantics out
            # of the data layer.
            for submodel in self.submodels:
                if not isinstance(submodel, nn.Module):
                    continue
                submodel.to("cpu" if self.model.keep_submodel_on_cpu_after_cache(submodel) else "meta")

        dist.barrier()
        if worker is not None:
            worker.join()

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
            tensor = _from_pipe(tensor)
            control_tensor = _from_pipe(control_tensor)
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
        pipe.send(_to_pipe(cpu_results))  # numpy transport: torch unpickling leaks (see _to_pipe)
