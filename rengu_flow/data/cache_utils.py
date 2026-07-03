"""Cache helpers: _map_and_cache, bucket_suffix, dedup_and_sort (from diffusion-pipe)."""

from __future__ import annotations

import hashlib
import os
import queue as _queue_mod
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from datasets.fingerprint import Hasher
from tqdm import tqdm

from rengu_flow.utils.cache import open_disk_cache

NUM_PROC = min(8, os.cpu_count() or 1)
ROUND_DECIMAL_DIGITS = 3


def resolve_cache_num_proc(value: int | None) -> int:
    """Return a positive worker count for the cache map/pool (default capped at 8).

    Delegated to the platform strategy: Windows forces in-process (1) because a spawned worker
    pool there cannot share the in-process queue/pipe the GPU-encode handoff relies on (deadlock)
    and re-imports torch/CUDA per worker. Elsewhere: ``NUM_PROC`` when unset, else the request.
    """
    from rengu_flow.platform_compat import PLATFORM

    return PLATFORM.cache_worker_count(value, default=NUM_PROC)


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
    return np.unique(np.round(values, ROUND_DECIMAL_DIGITS))


def seed_from_hash(item) -> int:
    """Deterministic seed from a path or bucket key (stable across processes)."""
    return int(hashlib.md5(str(item).encode()).hexdigest(), 16) % int(1e9)


# Per-worker rank lives in thread-local storage: each ThreadPoolExecutor thread pulls a distinct
# rank in its initializer so its GPU-handoff Pipe (pipes[rank] in the map_fn) never collides with
# another thread's. The in-thread (num_proc<=1) path sets rank 0 on the calling thread.
_thread_ctx = threading.local()


def _ordered_parallel_map(executor, fn, iterable, *, max_inflight):
    """Apply ``fn`` over ``iterable`` on ``executor`` threads, yielding results in input order
    (like ``pool.imap``) while keeping at most ``max_inflight`` tasks in flight. Bounded so workers
    can't run far ahead of the in-order GPU consumer and pile up results in RAM."""
    from collections import deque

    it = iter(iterable)
    pending: deque = deque()
    for _ in range(max_inflight):
        try:
            pending.append(executor.submit(fn, next(it)))
        except StopIteration:
            break
    while pending:
        result = pending.popleft().result()
        try:
            pending.append(executor.submit(fn, next(it)))
        except StopIteration:
            pass
        yield result


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
):
    """Map over dataset with map_fn(example, rank), persist results in Cache.

    Cache key = new_fingerprint_args + (fingerprint_override or dataset._fingerprint).
    Pass ``fingerprint_override`` (e.g. a ``content_fingerprint`` over the columns that
    actually determine this cache) to decouple it from the dataset's chained HuggingFace
    fingerprint. If map_fn is None, loads existing cache only (trust_cache path).
    """
    new_fingerprint_args = list(new_fingerprint_args or [])
    new_fingerprint_args.append(
        fingerprint_override if fingerprint_override is not None else dataset._fingerprint
    )
    # Literal (not a param): keeps existing cache fingerprints stable now that the
    # disk cache has a single format.
    new_fingerprint_args.append("cache_format=v2")
    new_fingerprint = Hasher.hash(new_fingerprint_args)
    cache_dir = Path(cache_dir)
    if cache_file_prefix:
        cache_dir = cache_dir / cache_file_prefix.strip("_")

    cache = open_disk_cache(cache_dir, new_fingerprint)

    from rengu_flow.data import caching_progress

    progress = caching_progress.get_active()
    label = cache_file_prefix.strip("_") or "items"

    if map_fn is None:
        assert new_fingerprint == cache.fingerprint
        if progress is not None:
            progress.add_reused(len(cache))
            progress.note(f"{label}: {len(cache)} loaded from cache (trusted)")
        return cache

    if regenerate_cache:
        cache.clear()

    cache_size = len(cache)
    dataset_size = len(dataset)
    assert cache_size <= dataset_size
    if progress is not None:
        # The audit line: how much of this cache is actually being reused vs re-encoded.
        progress.add_reused(cache_size)
        progress.add_encoded(dataset_size - cache_size)
        progress.note(f"{label}: {dataset_size - cache_size} to encode, {cache_size} cached")
    if cache_size == dataset_size:
        return cache
    dataset = dataset.select(
        range(cache_size, dataset_size), keep_in_memory=keep_in_memory
    )

    pool_workers = resolve_cache_num_proc(num_proc)
    # Parallelize the CPU-side preprocessing (image load/decode/resize/augment) on a
    # ThreadPoolExecutor while the GPU encode stays serialized in the main process (each worker
    # marshals its tensor over the queue+Pipe handoff). Threads, not an mp.Pool: the GPU encode is
    # the bottleneck (the pool sat mostly idle waiting on it), and threads have no fork-after-CUDA
    # hazard and no cross-process queue to deadlock on — the pattern kohya/OneTrainer use.
    # num_proc<=1 (Windows, or explicit) runs the map on the calling thread.
    use_threads = pool_workers > 1
    executor = None
    if use_threads:
        rank_queue = _queue_mod.Queue()
        for i in range(pool_workers):
            rank_queue.put(i)

        def _assign_rank():
            _thread_ctx.rank = rank_queue.get()

        executor = ThreadPoolExecutor(max_workers=pool_workers, initializer=_assign_rank)
    else:
        _thread_ctx.rank = 0

    def wrapper(example):
        return map_fn(example, _thread_ctx.rank)

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

    batch_iter = dataset.iter(batch_size=caching_batch_size)
    map_iter = (
        _ordered_parallel_map(executor, wrapper, batch_iter, max_inflight=pool_workers * 2)
        if use_threads
        else map(wrapper, batch_iter)
    )
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
        if progress is not None:
            progress.unit_progress(done, total_batches)

    if executor is not None:
        executor.shutdown(wait=True)
    cache.finalize_current_shard()
    return cache
