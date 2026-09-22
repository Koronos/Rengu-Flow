"""Preview-sampling helpers shared by the DiT pipelines."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def preview_compute_dtype(pipeline) -> torch.dtype:
    """The pipeline's compute dtype (``model.dtype``; strings resolved to ``torch`` dtypes)."""
    dtype = pipeline.model_config.get("dtype", torch.bfloat16)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return dtype


def preview_autocast(pipeline):
    """CUDA autocast at the pipeline's compute dtype; a no-op context without CUDA."""
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=preview_compute_dtype(pipeline))
    return nullcontext()
