"""Helpers of the contract suite (imported by its conftest and test modules)."""

from __future__ import annotations

import importlib
import math


def import_or_skip(module: str):
    """Import ``module``, skipping the test only when an *external* dependency is missing (an
    extra not installed). A missing ``rengu_flow`` module is a bug, never a skip: it re-raises."""
    import pytest

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if not missing or missing == "rengu_flow" or missing.startswith("rengu_flow."):
            raise
        pytest.skip(f"{module} needs {missing!r}, which is not installed")


def tensor_summary(t) -> dict:
    """Shape, sum, abs-sum and 16 strided samples of a tensor (float64 on CPU)."""
    import torch

    flat = t.detach().to("cpu", torch.float64).flatten()
    idx = torch.linspace(0, flat.numel() - 1, min(16, flat.numel())).long()
    return {
        "shape": list(t.shape),
        "sum": float(flat.sum()),
        "abs_sum": float(flat.abs().sum()),
        "samples": [float(v) for v in flat[idx]],
    }


def image_summary(image) -> dict:
    """Size, mode, mean and 16 strided pixel values of a PIL image."""
    import numpy as np

    arr = np.asarray(image, dtype=np.float64).flatten()
    idx = np.linspace(0, arr.size - 1, 16).astype(int)
    return {"size": list(image.size), "mode": image.mode, "mean": float(arr.mean()), "pixels": [float(v) for v in arr[idx]]}


def close(a, b, rel: float, abs_: float) -> bool:
    """Structural JSON comparison with float tolerances."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], rel, abs_) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y, rel, abs_) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
    return a == b


def materialize(submodel) -> None:
    """Put a frozen submodel on the CPU the way ``DatasetManager`` does before it encodes (a lazy
    streamed encoder loads itself)."""
    submodel.to("cpu")
    if next(submodel.parameters()).device.type == "meta" and callable(getattr(submodel, "load", None)):
        submodel.load()


def inter_layer_values(layers, features) -> tuple[object, list]:
    """``(output, [value after each layer but the last])``."""
    seen = []
    x = features
    for layer in layers:
        x = layer(x)
        seen.append(x)
    return x, seen[:-1]


def block_module_ids(pipeline) -> set[int]:
    """``id`` of every module inside the pipeline's transformer blocks (its block-swap units)."""
    return {id(m) for block in pipeline.get_block_swap_modules() for m in block.modules()}
