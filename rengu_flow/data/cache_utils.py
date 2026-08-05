"""Cache helpers: _map_and_cache, bucket_suffix, dedup_and_sort (from diffusion-pipe)."""

from __future__ import annotations

import hashlib
import os
import queue as _queue_mod
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from datasets.fingerprint import Hasher
from tqdm import tqdm

from rengu_flow.utils.cache import (
    FORMAT_VERSION as CACHE_FORMAT_VERSION,
    freeze_identity,
    open_disk_cache,
    peek_manifest,
)

NUM_PROC = min(8, os.cpu_count() or 1)
ROUND_DECIMAL_DIGITS = 3


def _make_malloc_trim():
    """Best-effort glibc malloc_trim(0): the multi-threaded CPU preprocessing churns
    differently-sized PIL/tensor buffers, fragmenting per-thread arenas that glibc
    never returns to the OS — so RSS creeps up the whole run. Trimming periodically
    hands the freed top-of-heap back. No-op off glibc/Linux."""
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=False)
        libc.malloc_trim(0)

        def trim() -> None:
            try:
                libc.malloc_trim(0)
            except Exception:  # noqa: BLE001
                pass

        return trim
    except Exception:  # noqa: BLE001 — non-glibc platform
        return lambda: None


_malloc_trim = _make_malloc_trim()


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


def _stage_salvage(cache_dir: Path, salvage_dir: Path, new_fingerprint: str, reuse_key: str) -> None:
    """Move a still-usable-but-rekeyed cache aside so its rows can be salvaged by identity.

    Opening a cache whose fingerprint moved on clears it, so the old rows have to be preserved
    *before* that: this renames the directory to ``<name>.salvage``, leaving the real path empty
    for a fresh build that copies what it can from the donor. A leftover ``.salvage`` from an
    interrupted merge is kept (the merge resumes and finishes consuming it); a stale one whose
    reuse key no longer matches is discarded.
    """
    manifest = peek_manifest(cache_dir)
    if manifest is None:
        return
    if manifest.get("fingerprint") == new_fingerprint:
        return  # cache is already keyed right; an existing .salvage belongs to a resumed merge
    salvageable = (
        manifest.get("format_version") == CACHE_FORMAT_VERSION
        # A cache from before reuse keys existed gets one stamped the first time it is opened
        # while still valid (see Cache.init). If it is missing here, this cache was rekeyed
        # before that ever happened, so there is no proof of which augmentation baked its rows —
        # rebuild rather than risk reusing stale ones.
        and manifest.get("reuse_key") == reuse_key
        and int(manifest.get("count") or 0) > 0
    )
    if not salvageable:
        return
    if salvage_dir.exists():
        # A previous salvage that was never consumed; the current donor is the fresher one.
        shutil.rmtree(salvage_dir, ignore_errors=True)
    try:
        os.replace(cache_dir, salvage_dir)
    except OSError:
        # Cannot stage (e.g. handles still open on Windows): fall back to today's behavior of
        # rebuilding from scratch rather than failing the run.
        return


def _open_donors(salvage_dir: Path, donor_dirs, reuse_key: str) -> list:
    """Open the caches that may donate rows: the staged previous version, then any siblings.

    Each donor is opened with the fingerprint recorded in its own manifest, so opening never
    clears it, and only donors whose reuse key matches are considered.
    """
    donors = []
    for path in [salvage_dir, *(donor_dirs or [])]:
        path = Path(path)
        manifest = peek_manifest(path)
        if manifest is None or manifest.get("reuse_key") != reuse_key:
            continue
        if manifest.get("format_version") != CACHE_FORMAT_VERSION:
            continue
        if int(manifest.get("count") or 0) <= 0:
            continue
        try:
            donors.append(open_disk_cache(path, manifest["fingerprint"], reuse_key=reuse_key))
        except (ValueError, OSError):
            continue
    return donors


def _match_donor_rows(dataset, pending: list[int], identity_columns, donors) -> dict[int, tuple]:
    """Match pending rows to donor rows by identity → ``{row: (donor, donor_idx)}``.

    Identity comes from the dataset columns that determine the cached value (the image and its
    augmentation variant for latents; the caption for text embeddings), never from row position.
    Earlier donors win, and each donor row is handed out at most once so duplicate identities map
    to distinct rows.
    """
    columns = [c for c in identity_columns if c in dataset.column_names]
    if not columns:
        return {}
    subset = dataset.select(pending)
    identities = {
        row: freeze_identity(values)
        for row, values in zip(pending, zip(*(subset[c] for c in columns)))
    }
    matched: dict[int, tuple] = {}
    for donor in donors:
        available = donor.identity_index(tuple(columns))
        if not available:
            continue
        for row in pending:
            if row in matched:
                continue
            slot = available.get(identities[row])
            if slot:
                matched[row] = (donor, slot.pop(0))
    return matched


def _discard_salvage(salvage_dir: Path) -> None:
    """Drop a consumed donor directory (best effort — a leftover is re-staged next run)."""
    if salvage_dir.exists():
        shutil.rmtree(salvage_dir, ignore_errors=True)


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
    identity_columns: tuple[str, ...] = (),
    donor_dirs: list | None = None,
):
    """Map over dataset with map_fn(example, rank), persist results in Cache.

    Cache key = new_fingerprint_args + (fingerprint_override or dataset._fingerprint).
    Pass ``fingerprint_override`` (e.g. a ``content_fingerprint`` over the columns that
    actually determine this cache) to decouple it from the dataset's chained HuggingFace
    fingerprint. If map_fn is None, loads existing cache only (trust_cache path).

    ``identity_columns`` enables **row salvage**: the cache key covers the whole row set in
    order, so adding/removing/reordering rows (a new resolution, excluded images) changes it and
    would otherwise throw away every already-computed row. When set, rows are matched by identity
    (these columns) against donor caches and copied from disk instead of re-encoded on the GPU.
    Donors are this cache's own previous version plus any ``donor_dirs`` (e.g. sibling buckets for
    text embeddings, which do not depend on the size bucket). ``new_fingerprint_args`` doubles as
    the *reuse key*: donors must agree on it (augmentation config, encoder index), so a genuine
    recompute — e.g. edited augmentation — still re-encodes rather than reusing stale rows.
    """
    new_fingerprint_args = list(new_fingerprint_args or [])
    # The row-set-independent part of the key: what the rows were computed WITH, not WHICH rows.
    reuse_key = Hasher.hash([*new_fingerprint_args, "cache_format=v2"])
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

    salvage_dir = cache_dir.with_name(cache_dir.name + ".salvage")
    if map_fn is not None and identity_columns and not regenerate_cache:
        _stage_salvage(cache_dir, salvage_dir, new_fingerprint, reuse_key)
    elif regenerate_cache and salvage_dir.exists():
        shutil.rmtree(salvage_dir, ignore_errors=True)

    cache = open_disk_cache(cache_dir, new_fingerprint, reuse_key=reuse_key)

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
    if cache_size == dataset_size:
        if progress is not None:
            progress.add_reused(cache_size)
            progress.note(f"{label}: 0 to encode, {cache_size} cached")
        _discard_salvage(salvage_dir)
        return cache

    pending = list(range(cache_size, dataset_size))
    # Rows already computed elsewhere (this cache's pre-rekey version, or a sibling bucket) are
    # copied from disk; only what is genuinely missing reaches the GPU.
    salvaged: dict[int, tuple] = {}
    donors: list = []
    if identity_columns:
        donors = _open_donors(salvage_dir, donor_dirs, reuse_key)
        if donors:
            salvaged = _match_donor_rows(dataset, pending, identity_columns, donors)

    to_encode = [row for row in pending if row not in salvaged]
    if progress is not None:
        # The audit line: how much of this cache is actually being reused vs re-encoded.
        progress.add_reused(cache_size + len(salvaged))
        progress.add_encoded(len(to_encode))
        note = f"{label}: {len(to_encode)} to encode, {cache_size} cached"
        if salvaged:
            note += f", {len(salvaged)} salvaged (reused without re-encoding)"
        progress.note(note)
    dataset_all = dataset
    dataset = dataset.select(to_encode, keep_in_memory=keep_in_memory)

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

    batch_iter = dataset.iter(batch_size=caching_batch_size)
    map_iter = (
        _ordered_parallel_map(executor, wrapper, batch_iter, max_inflight=pool_workers * 2)
        if use_threads
        else map(wrapper, batch_iter)
    )

    if salvaged:
        # Write rows in dataset order regardless of where each one came from, so the cache stays
        # positionally 1:1 with the metadata (the iteration order indexes it by position).
        def encoded_rows():
            for batch in map_iter:
                yield from unbatch_iter(batch)

        encoded = encoded_rows()
        pbar = tqdm(total=len(pending), disable=not sys.stderr.isatty())
        for done, row in enumerate(pending, start=1):
            hit = salvaged.get(row)
            if hit is not None:
                donor, donor_idx = hit
                cache.add(donor[donor_idx])
            else:
                try:
                    cache.add(next(encoded))
                except StopIteration:
                    raise RuntimeError(
                        f"{label}: the encode produced fewer rows than the {len(to_encode)} "
                        "it was asked for, so salvaged rows cannot be placed in order"
                    ) from None
            pbar.update(1)
            if done % 64 == 0:
                _malloc_trim()  # return fragmented arena memory to the OS during the run
            if progress is not None:
                progress.unit_progress(done, len(pending))
        pbar.close()
    else:
        # No salvage: drain the map as before. A map may emit a different number of rows than it
        # was handed (a video split into several clips), which this path tolerates.
        completed_batches = cache_size // caching_batch_size
        total_batches = max(1, dataset_size // caching_batch_size)
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
            if done % 64 == 0:
                _malloc_trim()  # return fragmented arena memory to the OS during the run
            if progress is not None:
                progress.unit_progress(done, total_batches)

    if executor is not None:
        executor.shutdown(wait=True)
    cache.finalize_current_shard()
    for donor in donors:
        donor.close()
    _discard_salvage(salvage_dir)
    del dataset_all
    return cache
