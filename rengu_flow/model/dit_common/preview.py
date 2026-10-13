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


def image_to_pil(image: torch.Tensor):
    """``(3, H, W)`` decoded image in ``[-1, 1]`` -> ``PIL.Image`` (RGB, uint8)."""
    from PIL import Image

    pixels = ((image.float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())


def preview_autocast(pipeline):
    """CUDA autocast at the pipeline's compute dtype; a no-op context without CUDA."""
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=preview_compute_dtype(pipeline))
    return nullcontext()
