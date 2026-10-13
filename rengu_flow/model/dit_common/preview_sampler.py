"""Preview-sampling utilities shared by the DiT pipelines.

Utilities only: each model keeps its own Euler loop (~40 lines). A template sampler would force
one dtype policy (krea2 iterates in the parameter dtype, qwen_image21 in fp32) and change the
numerics; the CFG formula is data (``apply_cfg``'s ``base``), never unified.
"""

from __future__ import annotations

import math

import torch

from rengu_flow.model.dit_common.preview import image_to_pil
from rengu_flow.utils.common import round_to_nearest_multiple

# CFG bases: krea2 extrapolates from the conditional prediction, the others from the
# unconditional one (classic classifier-free guidance).
CFG_FROM_COND = "cond"  # cond + g * (cond - uncond)
CFG_FROM_UNCOND = "uncond"  # uncond + g * (cond - uncond)


def preview_size(preview_cfg: dict, multiple: int, *, default: int = 1024, minimum: int = 0) -> tuple[int, int]:
    """``(width, height)`` of ``preview_cfg`` rounded to the nearest ``multiple`` (at least
    ``minimum``)."""
    return tuple(
        max(minimum, round_to_nearest_multiple(int(preview_cfg.get(key, default)), multiple))
        for key in ("width", "height")
    )


def preview_settings(pipeline, preview_cfg: dict, *, negative_prompt: str = "") -> tuple[int, float, str]:
    """``(num_inference_steps, guidance_scale, negative_prompt)`` of ``preview_cfg``; missing
    steps/guidance fall back to the model spec's ``preview_spec`` defaults."""
    spec = pipeline.spec.preview_spec
    return (
        int(preview_cfg.get("num_inference_steps", spec.steps)),
        float(preview_cfg.get("guidance_scale", spec.guidance)),
        str(preview_cfg.get("negative_prompt", negative_prompt)),
    )


def shifted_sigmas(num_steps: int, mu: float, *, shift_terminal: float | None = None, device="cpu") -> torch.Tensor:
    """The ``num_steps + 1`` float32 sigmas (terminal 0 appended) of diffusers'
    ``FlowMatchEulerDiscreteScheduler`` with the exponential dynamic shift at ``mu``:
    ``linspace(1, 1/n, n)`` shifted, then (``shift_terminal``) stretched so the last sigma lands
    on it. Computed in float64 on ``device``."""
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float64, device=device)
    sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
    if shift_terminal is not None:
        one_minus = 1 - sigmas
        sigmas = 1 - one_minus / (one_minus[-1] / (1 - shift_terminal))
    return torch.cat([sigmas, sigmas.new_zeros(1)]).float()


def apply_cfg(cond: torch.Tensor, uncond: torch.Tensor, guidance: float, base: str) -> torch.Tensor:
    """Classifier-free guidance ``base + g * (cond - uncond)``; ``base`` is ``CFG_FROM_COND`` or
    ``CFG_FROM_UNCOND`` (the model's formula, as data)."""
    if base == CFG_FROM_COND:
        return cond + guidance * (cond - uncond)
    if base == CFG_FROM_UNCOND:
        return uncond + guidance * (cond - uncond)
    raise ValueError(f"unknown CFG base {base!r}")


def seeded_noise(shape, seed: int, device, *, dtype=torch.float32, generator_device=None) -> torch.Tensor:
    """Initial preview noise from ``seed``: drawn on ``generator_device`` (default ``device``;
    e.g. ``"cpu"`` to match ComfyUI) and moved to ``device``."""
    gen_device = device if generator_device is None else generator_device
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=gen_device, dtype=dtype).to(device)


def run_blocks(pipeline, blocks, hidden, run):
    """``hidden = run(i, block, hidden)`` over ``blocks``, honoring the preview block-swap
    offloader (``pipeline._preview_offloader``, set by ``prepare_preview_memory``)."""
    offloader = getattr(pipeline, "_preview_offloader", None)
    for i, block in enumerate(blocks):
        if offloader is not None:
            offloader.wait_for_block(i)
        hidden = run(i, block, hidden)
        if offloader is not None:
            offloader.submit_move_blocks_forward(i)
    return hidden


def decode_single_frame(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode ``(B, C, h, w)`` latents with a video-shaped (``T = 1``) image VAE."""
    return vae.decode(latents.unsqueeze(2)).sample.squeeze(2)


def decode_to_pil(pipeline, latents: torch.Tensor, device, *, decode=decode_single_frame, to_pil=image_to_pil):
    """Normalized latents -> ``PIL.Image``: move the VAE to ``device``, undo the cache
    normalization (``pipeline._latent_stats``), ``decode(vae, latents)`` and convert the first
    image (``to_pil``; default ``[-1, 1]`` RGB)."""
    vae = pipeline.vae.to(device)
    mean, std = pipeline._latent_stats(device, vae.dtype)
    image = decode(vae, latents.to(vae.dtype) * std + mean)
    return to_pil(image[0])
