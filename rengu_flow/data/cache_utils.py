"""Cache helpers: _map_and_cache, bucket_suffix, dedup_and_sort (from diffusion-pipe)."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets.fingerprint import Hasher
from tqdm import tqdm

try:
    import multiprocess as mp
except ImportError:
    import multiprocessing as mp  # type: ignore[no-redef]

from rengu_flow.utils.cache_factory import CACHE_FORMAT_V2, open_disk_cache

NUM_PROC = min(8, os.cpu_count() or 1)
ROUND_DECIMAL_DIGITS = 3


def resolve_cache_num_proc(value: int | None) -> int:
    """Return a positive worker count for cache map/pool (default capped at 8)."""
    if value is None:
        return NUM_PROC
    return max(1, int(value))


def content_fingerprint(dataset, columns: list[str]) -> str:
    """Stable hash of specific column *contents*, independent of HuggingFace's chained
    fingerprint.

    ``dataset._fingerprint`` is derived from the dataset's whole transform history, so a
    cache keyed on it is invalidated by any unrelated upstream change (e.g. a reshuffled
    caption column invalidating the latent cache). Hashing only the columns that actually
    determine a cache's contents lets each cache invalidate solely when its own inputs
    change. Order-dependent on purpose: rows are stored positionally, so a reordering must
    rebuild the cache anyway.
    """
    hasher = Hasher()
    for col in columns:
        hasher.update(col)
        # dataset[col] is a lazy Column whose hash carries the source dataset's identity;
        # materialize to plain Python so the hash depends only on the values.
        hasher.update(list(dataset[col]))
    return hasher.hexdigest()


def bucket_suffix(key: tuple) -> str:
    """Format a bucket key as a path-safe suffix."""
    if len(key) == 2:
        return f"{key[0]:.{ROUND_DECIMAL_DIGITS}f}_{key[1]}"
    if len(key) == 3:
        return f"{key[0]}x{key[1]}x{key[2]}"
    if len(key) == 4:
        return f"{key[0]:.{ROUND_DECIMAL_DIGITS}f}x{key[1]}x{key[2]}x{key[3]}"
    raise RuntimeError(f"Unexpected bucket key: {key}")


def dedup_and_sort(values: list[float]) -> np.ndarray:
    """Deduplicate and sort values; round to ROUND_DECIMAL_DIGITS."""
    values = set(round(x, ROUND_DECIMAL_DIGITS) for x in values)
    values = list(values)
    values.sort()
    return np.array(values)


def seed_from_hash(item) -> int:
    """Deterministic seed from a path or bucket key (stable across processes)."""
    return int(hashlib.md5(str(item).encode()).hexdigest(), 16) % int(1e9)


def _map_and_cache(
    dataset,
    map_fn,
    cache_dir: str | Path,
    cache_file_prefix: str = "",
    new_fingerprint_args: list | None = None,
    fingerprint_override: str | None = None,
    regenerate_cache: bool = False,
    caching_batch_size: int = 1,
    num_proc: int | None = None,
    keep_in_memory: bool = False,
    cache_format: str = CACHE_FORMAT_V2,
):
    """Map over dataset with map_fn(example, rank), persist results in Cache.

    Cache key = new_fingerprint_args + (fingerprint_override or dataset._fingerprint) +
    cache_format. Pass ``fingerprint_override`` (e.g. a ``content_fingerprint`` over the
    columns that actually determine this cache) to decouple it from the dataset's chained
    HuggingFace fingerprint. If map_fn is None, loads existing cache only (trust_cache path).
    """
    new_fingerprint_args = list(new_fingerprint_args or [])
    new_fingerprint_args.append(
        fingerprint_override if fingerprint_override is not None else dataset._fingerprint
    )
    new_fingerprint_args.append(f"cache_format={cache_format}")
    new_fingerprint = Hasher.hash(new_fingerprint_args)
    cache_dir = Path(cache_dir)
    if cache_file_prefix:
        cache_dir = cache_dir / cache_file_prefix.strip("_")

    cache = open_disk_cache(cache_dir, new_fingerprint, cache_format=cache_format)

    if map_fn is None:
        assert new_fingerprint == cache.fingerprint
        return cache

    if regenerate_cache:
        cache.clear()

    cache_size = len(cache)
    dataset_size = len(dataset)
    assert cache_size <= dataset_size
    if cache_size == dataset_size:
        return cache
    dataset = dataset.select(
        range(cache_size, dataset_size), keep_in_memory=keep_in_memory
    )

    pool_workers = resolve_cache_num_proc(num_proc)
    manager = mp.Manager()
    id_queue = manager.Queue()

    def init(queue):
        global rank
        rank = queue.get()

    for i in range(pool_workers):
        id_queue.put(i)

    pool = mp.Pool(pool_workers, init, (id_queue,))

    def wrapper(example):
        global rank
        return map_fn(example, rank)

    def recursive_clone_tensors(obj):
        if torch.is_tensor(obj):
            return obj.clone()
        if isinstance(obj, dict):
            return {k: recursive_clone_tensors(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [recursive_clone_tensors(x) for x in obj]
        return obj

    def unbatch_iter(batch):
        length = len(next(iter(batch.values())))
        for i in range(length):
            result = {key: batch[key][i] for key in batch}
            yield recursive_clone_tensors(result)

    completed_batches = cache_size // caching_batch_size
    total_batches = dataset_size // caching_batch_size

    # Throttled "caching" progress marker for the web UI (rank 0 only). The per-update
    # tqdm bar is disabled when stdout is not a TTY so the UI-captured log isn't spammed.
    from rengu_flow.control.progress_stream import ProgressEmitter
    from rengu_flow.utils import is_main_process

    cache_emitter = ProgressEmitter() if is_main_process() else None

    map_iter = pool.imap(wrapper, dataset.iter(batch_size=caching_batch_size))
    pbar = tqdm(
        map_iter,
        initial=completed_batches,
        total=total_batches,
        disable=not sys.stderr.isatty(),
    )
    done = completed_batches
    for batch in pbar:
        for example in unbatch_iter(batch):
            cache.add(example)
        done += 1
        if cache_emitter is not None:
            is_last = total_batches and done >= total_batches
            percent = (
                round(min(100.0, 100.0 * done / total_batches), 1)
                if total_batches
                else None
            )
            cache_emitter.emit(
                {
                    "phase": "caching",
                    "current": done,
                    "total": total_batches,
                    "percent": percent,
                },
                force=bool(is_last),
            )

    pool.close()
    cache.finalize_current_shard()
    return cache
