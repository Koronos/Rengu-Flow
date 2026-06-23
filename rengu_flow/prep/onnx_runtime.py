"""On-demand CUDA 12 runtime for the ONNX prep models on native Windows.

``onnxruntime-gpu`` is built against CUDA 12, but rengu's torch is cu13 — so on native Windows the
cu12 DLLs the CUDAExecutionProvider needs (``cublasLt64_12``, ``cudnn*``, ``cudart*``) are absent and
it silently falls back to CPU. We install the cu12 runtime on demand (only when an ONNX model is
about to run) and add its DLL directories to the loader path. No-op off Windows: Linux/WSL provides
the cu12 runtime via the system CUDA toolkit / manylinux wheels.

Call ``ensure_onnx_cuda_runtime()`` right before importing onnxruntime / creating an InferenceSession.
"""

from __future__ import annotations

import os
from pathlib import Path

from rengu_flow.platform_compat import PLATFORM
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

_prepared = False


def ensure_onnx_cuda_runtime() -> None:
    """Install (once, on demand) and expose the cu12 runtime so ONNX runs on GPU. Windows-only."""
    global _prepared
    if _prepared or not PLATFORM.is_windows:
        return
    from rengu_flow.install.manager import ensure_profiles

    ensure_profiles(["onnx-cuda"], reason="ONNX GPU inference")
    _add_nvidia_dll_dirs()
    _prepared = True


def _add_nvidia_dll_dirs() -> None:
    """Expose the cu12 runtime DLLs so onnxruntime's CUDA provider can load them.

    Each ``nvidia-*-cu12`` wheel ships its DLLs under ``site-packages/nvidia/<lib>/bin``. Adding those
    dirs with ``os.add_dll_directory`` is not enough: Python 3.8+ removed PATH from the default
    Windows DLL search, and onnxruntime resolves its provider's *transitive* deps (``cublasLt64_12``,
    ``cudnn*``, …) without the user-added dirs. So we also preload every DLL by full path — once a
    module is resident the loader reuses it by base name. Two tolerant passes settle inter-DLL order.
    """
    import ctypes

    try:
        import nvidia  # namespace package provided by the nvidia-*-cu12 wheels
    except ImportError:
        logger.warning("ONNX GPU: nvidia cu12 runtime not importable after install — staying on CPU.")
        return
    bin_dirs = [
        sub / "bin"
        for base in nvidia.__path__
        for sub in Path(base).iterdir()
        if (sub / "bin").is_dir()
    ]
    for bin_dir in bin_dirs:
        os.add_dll_directory(str(bin_dir))
    dlls = [p for bin_dir in bin_dirs for p in bin_dir.glob("*.dll")]
    for _ in range(2):
        for dll in dlls:
            try:
                ctypes.WinDLL(str(dll))
            except OSError:
                pass
    logger.info(
        "ONNX GPU: exposed %d nvidia cu12 DLL dir(s), preloaded %d DLL(s).", len(bin_dirs), len(dlls)
    )
