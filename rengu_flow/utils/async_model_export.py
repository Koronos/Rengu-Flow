"""Background disk write for model export after a CPU snapshot (rank 0 only, POC)."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

_ASYNC_EXPORT_TIMEOUT_SEC = 600


def _tensor_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def estimate_state_dict_bytes(
    state_dict: dict[str, torch.Tensor],
    save_dtype: torch.dtype | None = None,
) -> int:
    """Estimated CPU snapshot size without copying tensors."""
    total = 0
    for tensor in state_dict.values():
        dtype = save_dtype if save_dtype is not None else tensor.dtype
        total += tensor.numel() * _tensor_element_size(dtype)
    return total


def _available_ram_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available


def async_snapshot_fits_in_ram(
    state_dict: dict[str, torch.Tensor],
    save_dtype: torch.dtype | None,
) -> tuple[bool, int, int | None]:
    """Return (fits, needed_bytes, available_ram_bytes_or_none_if_unknown)."""
    needed = estimate_state_dict_bytes(state_dict, save_dtype)
    available = _available_ram_bytes()
    if available is None:
        return True, needed, None
    return needed <= available, needed, available


def async_snapshot_fits_from_config(
    state_dict: dict[str, torch.Tensor],
    save_dtype: torch.dtype | None,
    config: dict,
) -> tuple[bool, int, int | None]:
    return async_snapshot_fits_in_ram(state_dict, save_dtype)


def format_byte_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.1f} MiB"
    return f"{num_bytes} B"


def clone_state_dict_to_cpu(
    state_dict: dict[str, torch.Tensor],
    save_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Detach, copy to CPU, optionally cast. Synchronizes CUDA when available."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        t = tensor.detach()
        if t.device.type == "cpu" and (save_dtype is None or t.dtype == save_dtype):
            out[key] = t.clone()
        elif save_dtype is not None:
            out[key] = t.to(device="cpu", dtype=save_dtype)
        else:
            out[key] = t.to(device="cpu")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out


@dataclass(frozen=True)
class ModelExportJob:
    name: str
    save_dir: Path
    state_dict: dict[str, torch.Tensor]
    is_adapter: bool
    config_path: str


class AsyncModelExportWriter:
    """Single background thread; one export on disk at a time."""

    def __init__(self, write_fn: Callable[[ModelExportJob], None]) -> None:
        self._write_fn = write_fn
        self._work: queue.Queue[ModelExportJob | None] = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="async-model-export", daemon=False)
        self._worker.start()

    def _reraise_if_failed(self, message: str) -> None:
        with self._error_lock:
            if self._error is not None:
                raise RuntimeError(message) from self._error

    def submit(self, job: ModelExportJob) -> None:
        self.wait_done()
        self._reraise_if_failed("Previous async model export failed")
        self._idle.clear()
        self._work.put(job)

    def wait_done(self) -> None:
        if not self._idle.wait(timeout=_ASYNC_EXPORT_TIMEOUT_SEC):
            raise TimeoutError(
                f"Async model export did not finish within {_ASYNC_EXPORT_TIMEOUT_SEC}s"
            )
        self._reraise_if_failed("Async model export failed")

    def shutdown(self) -> None:
        self.wait_done()
        self._work.put(None)
        self._worker.join(timeout=_ASYNC_EXPORT_TIMEOUT_SEC)
        if self._worker.is_alive():
            raise TimeoutError(
                f"Async model export worker did not finish within {_ASYNC_EXPORT_TIMEOUT_SEC}s"
            )
        self._reraise_if_failed("Async model export failed")

    def _run(self) -> None:
        while True:
            job = self._work.get()
            try:
                if job is None:
                    return
                t0 = time.perf_counter()
                self._write_fn(job)
                elapsed = time.perf_counter() - t0
                print(f"[async_export] finished {job.name} disk write in {elapsed:.2f}s")
            except Exception as exc:
                with self._error_lock:
                    self._error = exc
                print(f"[async_export] failed while writing {job.name}: {exc}")
                traceback.print_exc()
            finally:
                self._idle.set()
