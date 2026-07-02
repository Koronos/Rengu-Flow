"""Disk cache: stacked bf16/fp tensors (mmap) + SQLite for per-item metadata.

``FORMAT_VERSION`` is stamped into the manifest and checked on load; a mismatch
clears the cache (forward-compat guard). A legacy ``metadata.db`` cache is rejected.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
META_DB_NAME = "meta.db"
TENSORS_DIR = "tensors"
CHECKPOINT_EVERY = 128  # items between resume checkpoints (flush + commit + manifest)

# Keys whose dim 0 is a token-sequence length and may grow/pad per item at any rank
# (see _align_tensor_to_spec). 2-D tensors are always treated as sequences.
_SEQUENCE_TENSOR_KEYS = frozenset({"prompt_embeds", "text_mask"})


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_str(name: str) -> torch.dtype:
    return getattr(torch, name)


def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * torch.empty((), dtype=dtype).element_size()


def _storage_dtype(dtype: torch.dtype) -> torch.dtype:
    """On-disk dtype for tensor payload (bf16 for float types to save space)."""
    if dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        return torch.bfloat16
    return dtype


class Cache:
    """Disk cache with mmap tensor stacks and JSON metadata in SQLite.

    Same public surface as ``Cache`` (legacy v1): ``__len__``, ``__getitem__``,
    ``get_many``, ``add``, ``clear``, ``finalize_current_shard``.
    """

    def __init__(self, path: str | Path, fingerprint: str) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.count = 0
        self.tensor_specs: dict[str, dict] = {}
        self._meta_con: sqlite3.Connection | None = None
        self._meta_read_only: bool = False
        self._tensor_files: dict[str, object] = {}
        self._mmaps: dict[str, np.memmap] = {}
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path / TENSORS_DIR, exist_ok=True)
        self.init()

    def init(self) -> None:
        manifest_path = self.path / MANIFEST_NAME
        if not manifest_path.is_file():
            self._write_manifest()
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            format_version = manifest["format_version"]
            count = int(manifest["count"])
            tensor_specs = manifest["tensors"]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Manifest truncated/corrupt (e.g. crash mid-write — it is not written
            # atomically). Regenerate rather than crash on resume.
            print("[CACHE] Manifest unreadable, clearing")
            self.clear()
            return
        if format_version != FORMAT_VERSION:
            print("[CACHE] Format version mismatch, clearing")
            self.clear()
            return

        existing_fp = manifest.get("fingerprint")
        if existing_fp != self.fingerprint:
            print(f"[CACHE] Fingerprint changed ({existing_fp} -> {self.fingerprint}), clearing")
            self.clear()
            return

        self.count = count
        self.tensor_specs = tensor_specs
        # SQLite + mmaps open lazily on first use — an opened-but-unread cache holds no
        # file handles, so many bucket caches can coexist without exhausting the fd limit.

    def _write_manifest(self) -> None:
        payload = {
            "format_version": FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "count": self.count,
            "tensors": self.tensor_specs,
        }
        # Atomic: a checkpoint manifest must never be left half-written (init() would
        # then clear the whole cache). Write to a temp file and rename into place.
        final = self.path / MANIFEST_NAME
        tmp = final.with_suffix(final.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, final)

    def _row_size(self, key: str) -> int:
        spec = self.tensor_specs[key]
        storage_dtype = _dtype_from_str(spec["storage_dtype"])
        return len(self._row_bytes(torch.zeros(spec["shape"], dtype=storage_dtype), storage_dtype))

    def _checkpoint(self) -> None:
        """Persist a consistent resume point: flush tensor bytes, commit the meta,
        then record the count in the manifest (in that order, so the manifest count
        never exceeds what's durably on disk). A crash after this resumes from here."""
        for f in self._tensor_files.values():
            f.flush()
        if self._meta_con is not None:
            self._meta_con.commit()
        self._write_manifest()

    def _open_meta(self, read_only: bool = False) -> None:
        if self._meta_con is not None:
            if self._meta_read_only == read_only:
                return
            if not self._meta_read_only:
                self._meta_con.commit()
            self._meta_con.close()
            self._meta_con = None
        db_path = self.path / META_DB_NAME
        # check_same_thread=False: the connection is opened once but read from DataLoader
        # prefetch/worker threads during training. SQLite's default serialized threading mode
        # keeps a single connection safe across threads; reads here are read-only, writes only
        # happen single-threaded while building the cache.
        if read_only:
            self._meta_con = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
            )
        else:
            self._meta_con = sqlite3.connect(db_path, check_same_thread=False)
            # WAL + NORMAL: cheap durable commits so add() can checkpoint mid-bucket
            # (bounds the open transaction; an OOM/crash no longer loses the whole bucket).
            self._meta_con.execute("PRAGMA journal_mode=WAL")
            self._meta_con.execute("PRAGMA synchronous=NORMAL")
            self._meta_con.execute(
                "CREATE TABLE IF NOT EXISTS item_meta(idx INTEGER PRIMARY KEY, payload TEXT)"
            )
            self._meta_con.commit()
        self._meta_read_only = read_only

    def _open_mmap(self, key: str) -> None:
        """Memory-map a single tensor stack on demand (one fd), if not already open."""
        if key in self._mmaps:
            return
        spec = self.tensor_specs.get(key)
        if spec is None:
            return
        path = self.path / TENSORS_DIR / f"{key}.bin"
        if not path.is_file():
            return
        dtype = _dtype_from_str(spec["storage_dtype"])
        shape = (self.count,) + tuple(spec["shape"])
        if dtype == torch.bfloat16:
            self._mmaps[key] = np.memmap(path, dtype=np.uint16, mode="r", shape=shape)
        else:
            np_dtype = np.dtype(str(dtype).removeprefix("torch."))
            self._mmaps[key] = np.memmap(path, dtype=np_dtype, mode="r", shape=shape)

    def _close_mmaps(self) -> None:
        # np.memmap holds the file open until its underlying mmap is closed; dropping the dict
        # reference alone leaves the handle until GC. Windows cannot unlink an open mmap'd file.
        for m in self._mmaps.values():
            mm = getattr(m, "_mmap", None)
            if mm is not None:
                mm.close()
        self._mmaps.clear()

    def _close_meta(self) -> None:
        """Commit and close the SQLite connection; reads re-open it read-only on demand."""
        if self._meta_con is not None:
            if not self._meta_read_only:
                self._meta_con.commit()
            self._meta_con.close()
            self._meta_con = None
            self._meta_read_only = False

    def _ensure_meta_for_read(self) -> None:
        """Open the meta DB read-only if it is closed; leave a write connection as-is."""
        if self._meta_con is None:
            self._open_meta(read_only=True)

    def close(self) -> None:
        """Release all file handles (mmaps + SQLite + tensor files) without deleting the cache.

        Call this when done with a cache before another run may clear it: on Windows the clear
        ``unlink`` fails (WinError 32) while this instance still has the tensor files mmap'd.
        """
        self._close_meta()
        for f in self._tensor_files.values():
            if hasattr(f, "close"):
                f.close()
        self._tensor_files.clear()
        self._close_mmaps()

    def clear(self) -> None:
        self.close()
        for child in self.path.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                import shutil

                shutil.rmtree(child)
        self.count = 0
        self.tensor_specs = {}
        os.makedirs(self.path / TENSORS_DIR, exist_ok=True)
        self.init()

    def __len__(self) -> int:
        return self.count

    def _split_item(self, item: dict) -> tuple[dict[str, torch.Tensor | None], dict]:
        tensors: dict[str, torch.Tensor | None] = {}
        meta: dict = {}
        for key, value in item.items():
            if torch.is_tensor(value):
                tensors[key] = value
            else:
                meta[key] = value
        return tensors, meta

    def _row_bytes(self, tensor: torch.Tensor, storage_dtype: torch.dtype) -> bytes:
        buf = tensor.to(storage_dtype).contiguous()
        if storage_dtype == torch.bfloat16:
            return buf.view(torch.uint16).numpy().tobytes()
        return buf.numpy().tobytes()

    def _decode_row(self, data: bytes, shape: tuple[int, ...], storage_dtype: torch.dtype) -> torch.Tensor:
        if storage_dtype == torch.bfloat16:
            t = torch.from_numpy(np.frombuffer(data, dtype=np.uint16).copy()).view(torch.bfloat16)
        else:
            np_dtype = np.dtype(str(storage_dtype).removeprefix("torch."))
            t = torch.from_numpy(np.frombuffer(data, dtype=np_dtype).copy())
        return t.reshape(shape)

    def refresh_reads(self) -> None:
        """Make rows added since the last refresh visible to reads.

        Reads mmap each tensor stack sized at open time and writers are buffered, so a
        cache used as a live read-back store (add, then read older rows, then add more —
        e.g. the text-embedding dedup spill) must flush and drop stale mmaps between
        writes and reads."""
        for f in self._tensor_files.values():
            f.flush()
        if self._meta_con is not None and not self._meta_read_only:
            self._meta_con.commit()
        self._close_mmaps()

    def _grow_tensor_dim0(self, key: str, new_d0: int) -> None:
        spec = self.tensor_specs[key]
        old_shape = tuple(int(x) for x in spec["shape"])
        if new_d0 <= old_shape[0]:
            return
        storage_dtype = _dtype_from_str(spec["storage_dtype"])
        new_shape = (new_d0,) + old_shape[1:]
        path = self.path / TENSORS_DIR / f"{key}.bin"
        if key in self._tensor_files:
            self._tensor_files[key].close()
            del self._tensor_files[key]
        self._mmaps.pop(key, None)
        old_row = int(spec["item_nbytes"])
        new_row = _tensor_nbytes(new_shape, storage_dtype)
        tmp = path.with_suffix(".bin.tmp")
        with tmp.open("wb") as out:
            if path.is_file() and self.count > 0:
                raw = path.read_bytes()
                for i in range(self.count):
                    chunk = raw[i * old_row : (i + 1) * old_row]
                    t = self._decode_row(chunk, old_shape, storage_dtype)
                    padded = torch.zeros(new_shape, dtype=storage_dtype)
                    padded[: old_shape[0]] = t
                    out.write(self._row_bytes(padded, storage_dtype))
        tmp.replace(path)
        spec["shape"] = list(new_shape)
        spec["item_nbytes"] = new_row
        self._write_manifest()
        self._tensor_files[key] = open(path, "ab")  # noqa: SIM115

    def _align_tensor_to_spec(
        self, key: str, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, list[int]]:
        """Pad dim 0 to bucket max; grow stack file when a longer sequence appears."""
        if key not in self.tensor_specs:
            self._register_tensor_key(key, tensor)
        spec_shape = tuple(int(x) for x in self.tensor_specs[key]["shape"])
        actual = tuple(int(x) for x in tensor.shape)
        if len(actual) != len(spec_shape) or actual[1:] != spec_shape[1:]:
            raise ValueError(
                f"Cache tensor {key} shape {actual} incompatible with bucket {spec_shape}"
            )
        if actual[0] > spec_shape[0]:
            # Dim-0 growth is for token sequences only: any 2-D tensor (cosmos TE rows are
            # (L, D)) plus the known sequence keys at other ranks (krea2's compacted stacks
            # are (L, layers, D), masks (L,)). Everything else — e.g. latents, where dim 0
            # is channels — keeps raising: a dim-0 mismatch there is a data bug that padding
            # would silently corrupt.
            if len(spec_shape) != 2 and key not in _SEQUENCE_TENSOR_KEYS:
                raise ValueError(
                    f"Cache tensor {key} shape {actual} incompatible with bucket {spec_shape}"
                )
            self._grow_tensor_dim0(key, actual[0])
            spec_shape = tuple(int(x) for x in self.tensor_specs[key]["shape"])
        storage_dtype = _dtype_from_str(self.tensor_specs[key]["storage_dtype"])
        if actual == spec_shape:
            return tensor.to(storage_dtype).contiguous(), list(actual)
        padded = torch.zeros(spec_shape, dtype=storage_dtype)
        idx_slices = (slice(0, actual[0]),) + (slice(None),) * (len(actual) - 1)
        padded[idx_slices] = tensor.to(storage_dtype)
        return padded, list(actual)

    def _register_tensor_key(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.tensor_specs:
            return
        storage_dtype = _storage_dtype(tensor.dtype)
        self.tensor_specs[key] = {
            "shape": list(tensor.shape),
            "dtype": _dtype_to_str(tensor.dtype),
            "storage_dtype": _dtype_to_str(storage_dtype),
            "item_nbytes": _tensor_nbytes(tensor.shape, storage_dtype),
        }
        path = self.path / TENSORS_DIR / f"{key}.bin"
        f = open(path, "wb")  # noqa: SIM115
        storage_dtype = _dtype_from_str(self.tensor_specs[key]["storage_dtype"])
        zeros = torch.zeros(self.tensor_specs[key]["shape"], dtype=storage_dtype)
        row_bytes = (
            zeros.view(torch.uint16).numpy().tobytes()
            if storage_dtype == torch.bfloat16
            else zeros.numpy().tobytes()
        )
        for _ in range(self.count):
            f.write(row_bytes)
        self._tensor_files[key] = f
        if self.count > 0 and self._meta_con is not None:
            for idx in range(self.count):
                row = self._meta_con.execute(
                    "SELECT payload FROM item_meta WHERE idx=?", (idx,)
                ).fetchone()
                if row is None:
                    continue
                meta = json.loads(row[0])
                nulls = list(meta.get("_null_tensors", []))
                if key not in nulls:
                    nulls.append(key)
                    meta["_null_tensors"] = nulls
                    meta.pop(key, None)
                    self._meta_con.execute(
                        "UPDATE item_meta SET payload=? WHERE idx=?",
                        (json.dumps(meta), idx),
                    )
        self._write_manifest()

    def _init_tensor_specs(self, tensors: dict[str, torch.Tensor | None]) -> None:
        for key, tensor in tensors.items():
            if tensor is None:
                continue
            self._register_tensor_key(key, tensor)

    def _ensure_writable(self) -> None:
        resuming = self._meta_con is None or self._meta_read_only
        if resuming:
            self._open_meta(read_only=False)
            # Resume: a crash may have left tensor rows / meta rows written past the
            # last manifest checkpoint. Drop that tail so manifest count, tensor files,
            # and meta all agree on exactly self.count rows.
            self._meta_con.execute("DELETE FROM item_meta WHERE idx >= ?", (self.count,))
            self._meta_con.commit()
        self._mmaps.clear()
        for key in self.tensor_specs:
            if key in self._tensor_files:
                continue
            path = self.path / TENSORS_DIR / f"{key}.bin"
            if resuming and path.exists():
                f = open(path, "r+b")  # noqa: SIM115
                f.truncate(self.count * self._row_size(key))
                f.seek(0, os.SEEK_END)
                self._tensor_files[key] = f
            else:
                self._tensor_files[key] = open(path, "ab")  # noqa: SIM115

    def add(self, item: dict) -> None:
        tensors, meta = self._split_item(item)
        if self.count == 0:
            self._init_tensor_specs(tensors)
            self._open_meta(read_only=False)
            self._write_manifest()
        else:
            self._ensure_writable()

        for key, tensor in tensors.items():
            if tensor is not None:
                self._register_tensor_key(key, tensor)

        null_tensor_keys: list[str] = []
        for key, spec in self.tensor_specs.items():
            tensor = tensors.get(key)
            storage_dtype = _dtype_from_str(spec["storage_dtype"])
            if tensor is None:
                null_tensor_keys.append(key)
                buf = torch.zeros(spec["shape"], dtype=storage_dtype)
            else:
                buf, actual_shape = self._align_tensor_to_spec(key, tensor)
                if len(actual_shape) == 2 or key in _SEQUENCE_TENSOR_KEYS:
                    meta.setdefault("_tensor_shapes", {})[key] = actual_shape
            if key not in self._tensor_files:
                self._ensure_writable()
            self._tensor_files[key].write(self._row_bytes(buf, storage_dtype))

        if null_tensor_keys:
            meta["_null_tensors"] = null_tensor_keys
        assert self._meta_con is not None
        self._meta_con.execute(
            "INSERT INTO item_meta(idx, payload) VALUES(?, ?)",
            (self.count, json.dumps(meta)),
        )
        self.count += 1
        # Checkpoint periodically (not just at finalize): flush + commit + manifest so
        # an interrupted run resumes from here instead of redoing everything.
        if self.count % CHECKPOINT_EVERY == 0:
            self._checkpoint()

    def finalize_current_shard(self) -> None:
        for f in self._tensor_files.values():
            if hasattr(f, "close"):
                f.close()
        self._tensor_files.clear()
        self._write_manifest()
        # Bucket finished caching: drop its mmaps and SQLite handle so a cached-but-unread
        # cache holds zero fds (reads re-open lazily). Bounds fds across many bucket caches.
        self._close_mmaps()
        self._close_meta()

    def _read_tensor(self, key: str, idx: int) -> torch.Tensor:
        spec = self.tensor_specs[key]
        original_dtype = _dtype_from_str(spec["dtype"])
        storage_dtype = _dtype_from_str(spec["storage_dtype"])
        self._open_mmap(key)
        arr = self._mmaps[key][idx]
        if storage_dtype == torch.bfloat16:
            t = torch.from_numpy(np.asarray(arr).copy()).view(torch.bfloat16)
        else:
            t = torch.from_numpy(np.asarray(arr).copy())
        if original_dtype != storage_dtype:
            t = t.to(original_dtype)
        return t

    def valid_flags(self) -> list[bool]:
        """Per-row 'valid' flag from the JSON meta only (no tensor mmap reads).

        Rows default to valid; a False marks a tombstone (e.g. a corrupt image's
        zero-placeholder latent) that callers should exclude. Cheap O(N) scan."""
        self._ensure_meta_for_read()
        out = [True] * self.count
        for idx, payload in self._meta_con.execute(
            "SELECT idx, payload FROM item_meta ORDER BY idx"
        ):
            if 0 <= idx < self.count:
                out[idx] = bool(json.loads(payload).get("valid", True))
        return out

    def _build_item(self, idx: int) -> dict:
        self._ensure_meta_for_read()
        row = self._meta_con.execute(
            "SELECT payload FROM item_meta WHERE idx=?", (idx,)
        ).fetchone()
        if row is None:
            raise IndexError(f"Cache index {idx} out of range")
        item = json.loads(row[0])
        null_keys = set(item.pop("_null_tensors", []) or [])
        tensor_shapes = item.pop("_tensor_shapes", {}) or {}
        for key in self.tensor_specs:
            if key in null_keys:
                item[key] = None
            else:
                t = self._read_tensor(key, idx)
                actual = tensor_shapes.get(key)
                if actual is not None:
                    t = t[tuple(slice(0, int(s)) for s in actual)]
                item[key] = t
        return item

    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            raise TypeError("Cache index must be int")
        return self._build_item(idx)

    def get_many(self, indices: list[int]) -> list:
        if not indices:
            return []
        return [self._build_item(i) for i in indices]


def reject_legacy_v1(path: str | Path) -> None:
    """Raise if *path* holds a legacy v1 (``metadata.db``) cache; no longer supported."""
    if (Path(path) / "metadata.db").is_file():
        raise ValueError(
            f"Legacy cache v1 at {path}; regenerate cache. "
            "Delete the old cache directory or run with --regenerate_cache."
        )


def open_disk_cache(path: str | Path, fingerprint: str) -> Cache:
    """Return a ``Cache`` for *path* (raises on a legacy v1 cache)."""
    path = Path(path)
    reject_legacy_v1(path)
    return Cache(path, fingerprint)
