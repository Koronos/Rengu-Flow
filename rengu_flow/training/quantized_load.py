"""Helpers for loading weights in reduced precision (fp8, etc.)."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

FP8_DTYPE_NAMES = frozenset({"float8", "float8_e4m3fn", "float8_e5m2"})


def is_fp8_dtype(dtype: Any) -> bool:
    if torch is None or dtype is None:
        return False
    if isinstance(dtype, str):
        return dtype.lower() in FP8_DTYPE_NAMES
    return dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )


def bulk_dtype_for_load(
    param_name: str,
    *,
    default_dtype: Any,
    bulk_dtype: Any,
    high_precision_keywords: tuple[str, ...] = (),
) -> Any:
    """Pick load dtype for one parameter (Cosmos-style bulk vs embedder precision)."""
    if any(kw in param_name for kw in high_precision_keywords):
        return default_dtype
    return bulk_dtype
