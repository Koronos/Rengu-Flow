"""Cache format v2: stacked bf16/fp tensors (mmap) + SQLite for per-item metadata."""

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


class CacheV2:
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
            self._open_meta()
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            print("[CACHE v2] Format version mismatch, clearing")
            self.clear()
            return

        existing_fp = manifest.get("fingerprint")
        if existing_fp != self.fingerprint:
            print(f"[CACHE v2] Fingerprint changed ({existing_fp} -> {self.fingerprint}), clearing")
            self.clear()
            return

        self.count = int(manifest["count"])
        self.tensor_specs = manifest["tensors"]
        self._open_meta(read_only=True)
        self._open_mmaps()

    def _write_manifest(self) -> None:
        payload = {
            "format_version": FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "count": self.count,
            "tensors": self.tensor_specs,
        }
        (self.path / MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _open_meta(self, read_only: bool = False) -> None:
        if self._meta_con is not None:
            if self._meta_read_only == read_only:
                return
            if not self._meta_read_only:
                self._meta_con.commit()
            self._meta_con.close()
            self._meta_con = None
        db_path = self.path / META_DB_NAME
        if read_only:
            self._meta_con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        else:
            self._meta_con = sqlite3.connect(db_path)
            self._meta_con.execute(
                "CREATE TABLE IF NOT EXISTS item_meta(idx INTEGER PRIMARY KEY, payload TEXT)"
            )
            self._meta_con.commit()
        self._meta_read_only = read_only

    def _open_mmaps(self) -> None:
        self._mmaps.clear()
        for key, spec in self.tensor_specs.items():
            path = self.path / TENSORS_DIR / f"{key}.bin"
            if not path.is_file():
                continue
            dtype = _dtype_from_str(spec["storage_dtype"])
            shape = (self.count,) + tuple(spec["shape"])
            if dtype == torch.bfloat16:
                self._mmaps[key] = np.memmap(path, dtype=np.uint16, mode="r", shape=shape)
            else:
                np_dtype = np.dtype(str(dtype).removeprefix("torch."))
                self._mmaps[key] = np.memmap(path, dtype=np_dtype, mode="r", shape=shape)

    def clear(self) -> None:
        if self._meta_con is not None:
            if not self._meta_read_only:
                self._meta_con.commit()
            self._meta_con.close()
            self._meta_con = None
            self._meta_read_only = False
        for f in self._tensor_files.values():
            if hasattr(f, "close"):
                f.close()
        self._tensor_files.clear()
        self._mmaps.clear()
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
                f"Cache v2 tensor {key} shape {actual} incompatible with bucket {spec_shape}"
            )
        if actual[0] > spec_shape[0]:
            if len(spec_shape) != 2:
                raise ValueError(
                    f"Cache v2 tensor {key} shape {actual} incompatible with bucket {spec_shape}"
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
        if self._meta_read_only:
            self._open_meta(read_only=False)
        self._mmaps.clear()
        for key in self.tensor_specs:
            if key in self._tensor_files:
                continue
            path = self.path / TENSORS_DIR / f"{key}.bin"
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
                if len(actual_shape) == 2:
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

    def finalize_current_shard(self) -> None:
        for f in self._tensor_files.values():
            if hasattr(f, "close"):
                f.close()
        self._tensor_files.clear()
        if self._meta_con is not None:
            self._meta_con.commit()
        self._write_manifest()
        self._open_mmaps()

    def _read_tensor(self, key: str, idx: int) -> torch.Tensor:
        spec = self.tensor_specs[key]
        original_dtype = _dtype_from_str(spec["dtype"])
        storage_dtype = _dtype_from_str(spec["storage_dtype"])
        if key not in self._mmaps:
            self._open_mmaps()
        arr = self._mmaps[key][idx]
        if storage_dtype == torch.bfloat16:
            t = torch.from_numpy(np.asarray(arr).copy()).view(torch.bfloat16)
        else:
            t = torch.from_numpy(np.asarray(arr).copy())
        if original_dtype != storage_dtype:
            t = t.to(original_dtype)
        return t

    def _build_item(self, idx: int) -> dict:
        assert self._meta_con is not None
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
