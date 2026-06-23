"""On-demand, Windows-gated CUDA 12 runtime for the ONNX prep models."""

from __future__ import annotations

import rengu_flow.prep.onnx_runtime as onnx_rt
from rengu_flow.install.manager import _importable
from rengu_flow.install.profiles import (
    ALL_PROFILE_NAMES,
    PROFILE_EXTRAS,
    PROFILE_IMPORT_CHECKS,
)


def test_onnx_cuda_profile_registered_but_not_in_all():
    # Resolvable as a profile (so ensure_profiles can install it) ...
    assert PROFILE_EXTRAS["onnx-cuda"] == "onnx-cuda"
    assert PROFILE_IMPORT_CHECKS["onnx-cuda"]
    # ... but excluded from `all`: off Windows the extra is empty and the import-check would fail.
    assert "onnx-cuda" not in ALL_PROFILE_NAMES


def test_importable_handles_missing_dotted_parent():
    # find_spec raises ModuleNotFoundError for a dotted name whose parent is absent; _importable
    # must swallow that and return False (this is what the nvidia.cudnn probe relies on).
    assert _importable("nvidia.totally_missing_subpackage_xyz") is False
    assert _importable("json") is True


def test_ensure_onnx_cuda_runtime_noop_off_windows(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(onnx_rt, "_prepared", False)
    monkeypatch.setattr(onnx_rt, "PLATFORM", SimpleNamespace(is_windows=False))
    called = {"n": 0}

    def _boom(*a, **k):  # would run uv sync — must NOT be reached off Windows
        called["n"] += 1

    monkeypatch.setattr("rengu_flow.install.manager.ensure_profiles", _boom)
    onnx_rt.ensure_onnx_cuda_runtime()
    assert called["n"] == 0
