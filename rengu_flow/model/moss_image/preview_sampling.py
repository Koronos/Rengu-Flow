"""Euler flow-matching preview sampling for MOSS-Image: the reference ``MOSSImagePipeline.__call__``.

* sigmas: ``MOSSImageScheduler.set_timesteps`` — a fixed shift 5.0 (not resolution dependent) over
  ``linspace(1, lower, n)``, every sigma snapped to the nearest point of the 1000-step training grid
  (consecutive duplicates removed), terminal 0 appended; the model time input is ``sigma * 1000``;
* true CFG ``uncond + g * (cond - uncond)`` in fp32 (two forwards per step, batch 1 each, no text
  padding anywhere), the negative prompt encoded through the same chat template (empty by default),
  and the negative branch skipping block 22 (``uncond_skip_layers = (22,)``, the reference default;
  ignored for a DiT shallower than that);
* fp32 noise, fp32 Euler update, FLUX.2 VAE decode in fp32 (``bn`` de-normalization + unpatchify).
"""

from __future__ import annotations

import torch

from rengu_flow.model.dit_common import preview_autocast, preview_compute_dtype
from rengu_flow.model.dit_common.preview import image_to_pil
from rengu_flow.model.dit_common.preview_sampler import (
    CFG_FROM_UNCOND,
    apply_cfg,
    decode_to_pil,
    preview_settings,
    preview_size,
    run_blocks,
    seeded_noise,
)

SHIFT = 5.0
NUM_TRAIN_TIMESTEPS = 1000
UNCOND_SKIP_LAYERS = (22,)
# Preview VAE decode tiling (pixels): an untiled fp32 decode of a 1024 px image needs several GB.
PREVIEW_TILE = 512


def _grid(count: int, shift: float, total: int = NUM_TRAIN_TIMESTEPS) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count)
    return shift * values / (1 + (shift - 1) * values)


def reference_sigmas(num_steps: int, device="cpu", shift: float = SHIFT) -> torch.Tensor:
    """The float32 sigmas of ``MOSSImageScheduler.set_timesteps(num_steps)`` with terminal 0 appended
    (``len == len(timesteps) + 1``; snapping can drop duplicates, so it may be shorter than steps + 1)."""
    total = NUM_TRAIN_TIMESTEPS
    sigmas = _grid(num_steps, shift)
    training = _grid(total, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)]).to(device)


def velocity(pipeline, latents: torch.Tensor, sigma: torch.Tensor, row: torch.Tensor, skip=(), compute_dtype=None):
    """One fp32 velocity ``(1, 128, h, w)`` for ``latents`` (fp32) given the prompt row ``(L, 4096)``,
    honoring the preview block-swap offloader; ``skip`` = block indices left out (identity)."""
    transformer = pipeline.transformer
    h, w = latents.shape[-2:]
    embeds = row.to(latents.device, compute_dtype)[None]
    x, mod, head, rope, _ = transformer.embed(latents, sigma * NUM_TRAIN_TIMESTEPS, embeds, None)
    skip = frozenset(skip)
    x = run_blocks(pipeline, transformer.layers, x, lambda i, block, hidden: hidden if i in skip else block(hidden, rope, mod))
    return transformer.final(x, head, h, w).float()


@torch.no_grad()
def generate_preview_image(pipeline, preview_cfg: dict, prompt: str, step: int, seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width, height = preview_size(preview_cfg, 16)
    num_steps, guidance, negative_prompt = preview_settings(pipeline, preview_cfg)
    do_cfg = guidance != 1.0

    pipeline.ensure_vae_for_preview()
    prompts = [prompt, negative_prompt] if do_cfg else [prompt]
    rows = pipeline.preview_prompt_embeds(prompts, preview_cfg, device)

    transformer = pipeline.transformer
    compute_dtype = preview_compute_dtype(pipeline)
    grid_h, grid_w = height // 16, width // 16
    latents = seeded_noise((1, transformer.config.latent_channels, grid_h, grid_w), seed, device)
    sigmas = reference_sigmas(num_steps, device)
    depth = len(transformer.layers)
    skip = tuple(i for i in UNCOND_SKIP_LAYERS if i < depth)

    with preview_autocast(pipeline):
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            prediction = velocity(pipeline, latents, sigma, rows[0], compute_dtype=compute_dtype)
            if do_cfg:
                uncond = velocity(pipeline, latents, sigma, rows[1], skip=skip, compute_dtype=compute_dtype)
                prediction = apply_cfg(prediction, uncond, guidance, CFG_FROM_UNCOND)
            latents = latents + (sigmas[i + 1] - sigma) * prediction
        return decode_to_pil(pipeline, latents, device, decode=_decode, to_pil=image_to_pil)


def _decode(vae, z: torch.Tensor) -> torch.Tensor:
    """De-normalized ``(B, 128, h, w)`` latents -> image in ``[-1, 1]`` (unpatchify + VAE decode,
    tiled above ``PREVIEW_TILE`` px)."""
    from rengu_flow.model.moss_image.pipeline import unpatchify

    z = unpatchify(z)
    tiling = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)
    vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size = True, PREVIEW_TILE, PREVIEW_TILE // 8
    try:
        return vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    finally:
        vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size = tiling
