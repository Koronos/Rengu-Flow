"""DatasetManager and _cache_fn: orchestrate latent + text embedding cache (from diffusion-pipe)."""

from __future__ import annotations

import sys
from collections import defaultdict
from inspect import signature

import datasets as datasets_mod
import torch
from torch import nn

try:
    import multiprocess as mp
except ImportError:
    import multiprocessing as mp  # type: ignore[no-redef]

from rengu_flow.control.progress_stream import ProgressEmitter
from rengu_flow.data import caching_progress
from rengu_flow.data.control import ControlRow, control_signature, control_size, load_control_image
from rengu_flow.data.dataset import CONTROL_IDENTITY_COLUMNS as _CONTROL_IDENTITY_COLUMNS
from rengu_flow.data.dataset import control_round_to_multiple
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


def _control_pil_to_tensor(img) -> torch.Tensor:
    """PIL control image -> (C, 1, H, W) float in [-1, 1] (C = 3 for RGB, 4 for RGBA)."""
    import numpy as np

    arr = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())
    if arr.ndim == 2:
        arr = arr.unsqueeze(-1)
    t = arr.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return t.unsqueeze(1)


def _load_control_tensors(files, dims, resolution: int, multiple: int):
    """``([tensor (C, 1, H_i, W_i), ...], valid)`` for one edit row's control images.

    A control whose pixels fail to decode (truncated file) tombstones the row like a corrupt
    target: a zero placeholder at the size its header promised, marked invalid (never sampled).
    """
    tensors = []
    valid = True
    for path, (w, h) in zip(files, dims):
        try:
            tensors.append(_control_pil_to_tensor(load_control_image(path, resolution, multiple)))
        except (OSError, SyntaxError) as e:  # UnidentifiedImageError is an OSError subclass
            print(f"[cache] corrupt control image tombstoned: {path} ({e})", flush=True)
            cw, ch = control_size(w, h, resolution, multiple)
            tensors.append(torch.zeros((3, 1, ch, cw)))
            valid = False
    return tensors, valid


def _load_control_images(files, dims, resolution: int, multiple: int) -> list:
    """PIL control images of one edit row for the text encoder. A corrupt one (already tombstoned
    by the latent pass, so its row is never sampled) becomes a gray placeholder at the size its
    header promised instead of aborting the caching run."""
    from PIL import Image

    images = []
    for path, (w, h) in zip(files, dims):
        try:
            images.append(load_control_image(path, resolution, multiple))
        except (OSError, SyntaxError) as e:  # UnidentifiedImageError is an OSError subclass
            print(f"[cache] corrupt control image, placeholder for its text embedding: {path} ({e})", flush=True)
            images.append(Image.new("RGB", control_size(w, h, resolution, multiple), (127, 127, 127)))
    return images


def _stack_control_slot(tensors: list) -> torch.Tensor:
    """Stack one control slot across a batch; RGB rows get an opaque alpha if any row is RGBA."""
    channels = max(t.shape[0] for t in tensors)
    if channels == 4:
        tensors = [
            t if t.shape[0] == 4 else torch.cat([t, torch.ones_like(t[:1])], dim=0)
            for t in tensors
        ]
    return torch.stack(tensors)


def _control_rows(datasets_list) -> list[ControlRow]:
    """Every edit row with the final sizes of its control images, from the metadata alone.

    Size-bucket mode: each bucket already carries its ``control_signature``. AR-bucket mode splits
    by signature only in ``cache_latents``, so the sizes are computed here per resolution, with the
    same ``control_signature`` / ``control_resolution_for`` the split uses. A target trained at
    several resolutions appears once per distinct set of sizes; text-to-image rows never appear.
    """
    rows: dict[ControlRow, None] = {}
    for ds in datasets_list:
        for dd in getattr(ds, "directory_datasets", []):
            if getattr(dd, "control_path", None) is None:
                continue
            if dd.use_size_buckets:
                for sb in dd.size_bucket_datasets:
                    if not sb.control_signature or len(sb.metadata_dataset) == 0:
                        continue
                    for spec in sb.metadata_dataset["image_spec"]:
                        rows[ControlRow(str(spec[1]), sb.control_signature)] = None
                continue
            multiple = dd.control_round_to_multiple
            for ar in dd.ar_bucket_datasets:
                meta = ar.metadata_dataset
                if len(meta) == 0 or "control_dims" not in meta.column_names:
                    continue
                specs, all_dims = meta["image_spec"], meta["control_dims"]
                for res in ar.resolutions:
                    control_res = dd.control_resolution_for(int(res))
                    for spec, dims in zip(specs, all_dims):
                        sizes = control_signature(dims, control_res, multiple)
                        rows[ControlRow(str(spec[1]), sizes)] = None
    return list(rows)


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


class _QueueChannel:
    """Same-process result channel: a plain in-memory handoff with no pickling and no torch
    shared-memory (memfd) reduction — ~36x cheaper per call than mp.Pipe for the CPU latent/
    embedding dict. Used only when the cache worker is a thread (single device). The multi-GPU
    worker is a real process, so it keeps mp.Pipe (genuine cross-process IPC)."""

    __slots__ = ("_q",)

    def __init__(self) -> None:
        import queue as _queue

        self._q = _queue.Queue(maxsize=1)

    def send(self, obj) -> None:
        self._q.put(obj)

    def recv(self):
        return self._q.get()


def _make_channel(single_process: bool):
    """(reader, writer) both exposing .recv()/.send(). Single-GPU thread -> one shared in-memory
    Queue channel (same object at both ends); multi-GPU process worker -> a real mp.Pipe."""
    if single_process:
        ch = _QueueChannel()
        return ch, ch
    return mp.Pipe(duplex=False)


# Queue task id: the worker asks the consumer to run model.validate_control_rows(rows).
_VALIDATE_CONTROL_ROWS = "__validate_control_rows__"


def _cache_fn(
    datasets_list,
    queue,
    preprocess_media_file_fn,
    num_text_encoders: int,
    regenerate_cache: bool,
    regenerate_text_cache: bool,
    trust_cache: bool,
    caching_batch_size: int,
    cache_num_proc: int | None,
    cache_keep_in_memory: bool,
    single_process_channel: bool = False,
    control_round_to_multiple: int = 32,
    validate_control_rows: bool = False,
) -> None:
    """Worker process: run cache_metadata, cache_latents, cache_text_embeddings; send GPU work via queue.

    ``control_round_to_multiple`` is the pixel multiple control images of edit datasets are
    floored to (see rengu_flow/data/control.py). ``validate_control_rows``: the model defines the
    hook of that name; the edit rows are sent to it (over the queue — the model lives in the
    consumer) after the metadata stage, before any latent or text embedding is encoded.
    """
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

    with progress.stage(
        "metadata", units=sum(len(ds.directory_datasets) for ds in datasets_list)
    ):
        for ds in datasets_list:
            ds.cache_metadata(
                # Metadata is cheap and feeds the text pass, so --regenerate_text_cache rebuilds
                # it too; the expensive VAE latents below stay keyed only on --regenerate_cache.
                regenerate_cache=regenerate_cache or regenerate_text_cache,
                trust_cache=trust_cache,
                cache_num_proc=cache_num_proc,
            )

    if validate_control_rows:
        rows = _control_rows(datasets_list)
        if rows:
            reader, writer = _make_channel(single_process_channel)
            queue.put((_VALIDATE_CONTROL_ROWS, rows, writer))
            error = reader.recv()
            if error is not None:
                # The consumer raises the model's own error; this only ends the worker.
                raise RuntimeError(f"control images rejected by the model: {error}")

    pipes = {}

    def latents_map_fn(example, rank):
        is_edit = "control_file" in example
        first_size_bucket = example["size_bucket"][0]
        tensors_and_masks = []
        image_specs = []
        # Edit rows: per row, (list of N control tensors (C, 1, H_i, W_i), valid). Each control
        # keeps its own aspect ratio (NOT the target's bucket): control.load_control_image is the
        # single sizing rule, shared with the text-encoder pass below.
        control_rows = []
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
                assert len(items) == 1, "edit datasets are images only"
                control_rows.append(
                    _load_control_tensors(
                        example["control_file"][i],
                        example["control_dims"][i],
                        int(example["control_resolution"][i]),
                        control_round_to_multiple,
                    )
                )

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
            c_tensors = None
            if is_edit:
                # One (B, C, 1, H_j, W_j) tensor per control slot j. Rows of one map batch share a
                # size bucket AND a control signature (Dataset groups buckets by it), so every
                # slot stacks.
                chunk = [r[0] for r in control_rows[i : i + batch_size]]
                c_tensors = [
                    _stack_control_slot([row[j] for row in chunk])
                    for j in range(len(chunk[0]))
                ]
            if rank not in pipes:
                pipes[rank] = _make_channel(single_process_channel)
            parent_conn, child_conn = pipes[rank]
            queue.put((0, _to_pipe(tensor), _to_pipe(c_tensors), child_conn))
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
                bool(t[2]) and bool(c[1])
                for t, c in zip(tensors_and_masks, control_rows)
            ]
            # Stored per row: the salvage identity of an edit latent (see SizeBucketDataset
            # cache_latents), so a replaced control is re-encoded instead of copied.
            for col in _CONTROL_IDENTITY_COLUMNS:
                results[col] = list(example[col])
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
            if rank not in pipes:
                pipes[rank] = _make_channel(single_process_channel)
            parent_conn, child_conn = pipes[rank]
            # Contract point 6: one entry per caption — the row's control images as PIL, sized by
            # the same helper as the latent pass — or None for a text-to-image row.
            if "control_file" in example:
                control_images = [
                    _load_control_images(files, dims, int(res), control_round_to_multiple)
                    for files, dims, res in zip(
                        example["control_file"], example["control_dims"], example["control_resolution"]
                    )
                ]
            else:
                control_images = [None] * len(captions)
            queue.put(
                (
                    text_encoder_idx + 1,
                    captions,
                    example["is_video"],
                    control_images,
                    child_conn,
                )
            )
            result = _from_pipe(parent_conn.recv())
            result["image_spec"] = example["image_spec"]
            # The caption is the identity of a text embedding: stored per row so another
            # bucket's cache can donate this row instead of re-encoding an identical caption
            # (see _map_and_cache's salvage path). Negligible next to the embedding itself.
            result["caption"] = captions
            # Edit rows: the control images are part of that identity too.
            for col in _CONTROL_IDENTITY_COLUMNS:
                if col in example:
                    result[col] = list(example[col])
            return result

        with progress.stage(
            f"text embeddings {text_encoder_idx + 1}",
            units=_count_te_units(datasets_list),
        ):
            for ds in datasets_list:
                ds.cache_text_embeddings(
                    text_embedding_map_fn,
                    text_encoder_idx + 1,
                    regenerate_cache=regenerate_cache or regenerate_text_cache,
                    caching_batch_size=caching_batch_size,
                    cache_num_proc=cache_num_proc,
                    cache_keep_in_memory=cache_keep_in_memory,
                )

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
        regenerate_text_cache: bool = False,
        trust_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
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
        # A 3-parameter text-encoder fn is edit-aware: it receives the rows' control images
        # (contract point 6). 2-parameter fns keep the text-only call.
        self.te_fn_accepts_control_images = [
            len(signature(fn).parameters) == 3
            for fn in self.call_text_encoder_fns
        ]
        self.regenerate_cache = regenerate_cache
        self.regenerate_text_cache = regenerate_text_cache
        self.trust_cache = trust_cache
        self.caching_batch_size = caching_batch_size
        self.cache_num_proc = cache_num_proc
        self.cache_keep_in_memory = cache_keep_in_memory
        self.backend = backend
        # Optional model hook: validate_control_rows(rows: list[ControlRow]) raises when the
        # model cannot take some edit rows' control images. Called once, after the metadata stage
        # and before any encode, so a bad dataset fails in seconds instead of mid-caching.
        hook = getattr(model, "validate_control_rows", None)
        self._validate_control_rows = hook if callable(hook) else None
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
                self.regenerate_text_cache,
                self.trust_cache,
                self.caching_batch_size,
                self.cache_num_proc,
                self.cache_keep_in_memory,
                # Single-device worker is a thread: use the in-memory Queue channel (no pickling).
                # Multi-GPU worker is a real process: keep mp.Pipe (cross-process IPC).
                not self.backend.is_distributed,
                control_round_to_multiple(self.model),
                self._validate_control_rows is not None,
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
        if task_id == _VALIDATE_CONTROL_ROWS:
            rows, pipe = task[1:]
            try:
                self._validate_control_rows(rows)
            except Exception as exc:
                pipe.send(str(exc))  # unblock the worker so it stops, then fail with the real error
                raise
            pipe.send(None)
            return
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
            tensor, control_tensors, pipe = task[1:]
            tensor = _from_pipe(tensor)
            control_tensors = _from_pipe(control_tensors)
            if control_tensors is not None:
                # Edit rows: list of N (B, C, 1, H_i, W_i) tensors; the fn adds
                # control_latents_0 .. control_latents_{N-1} to its result.
                results = self.call_vae_fn(tensor, control_tensors)
            else:
                results = self.call_vae_fn(tensor)
        elif task_id > 0:
            caption, is_video, control_images, pipe = task[1:]
            args = [caption, is_video]
            idx = task_id - 1
            if self.te_fn_accepts_control_images[idx]:
                args.append(control_images)
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
