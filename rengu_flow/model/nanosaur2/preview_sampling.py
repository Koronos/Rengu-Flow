"""Nanosaur2 preview sampling: Euler over ComfyUI's "simple" schedule of a shift-3 discrete flow
(``ModelSamplingDiscreteFlow``, 1000 steps), with the reference guidance modes:

* ``cfg``       — classifier-free guidance on every step (negative prompt, full model);
* ``path_drop`` — the unconditional pass is the negative prompt *without* the sparse middle
  blocks (SPRINT path-drop guidance) on every step;
* ``alternate`` — ``cfg`` on even steps, ``path_drop`` on odd ones (the reference default).
"""

from __future__ import annotations

import torch

from rengu_flow.model.dit_common import preview_autocast as _autocast
from rengu_flow.model.dit_common.preview import image_to_pil
from rengu_flow.model.dit_common.preview_sampler import (
    CFG_FROM_UNCOND,
    apply_cfg,
    preview_settings,
    preview_size,
    seeded_noise,
)

GUIDANCE_MODES = ("alternate", "cfg", "path_drop")
FLOW_SHIFT = 3.0
NUM_TRAIN_TIMESTEPS = 1000


def flow_sigmas(num_steps: int, shift: float = FLOW_SHIFT, multiplier: float = 1.0) -> torch.Tensor:
    """ComfyUI ``simple_scheduler`` over ``ModelSamplingDiscreteFlow(shift)`` sigmas, terminal 0
    appended: ``num_steps + 1`` float32 values from ~1 down to 0."""
    ts = torch.arange(1, NUM_TRAIN_TIMESTEPS + 1, dtype=torch.float32) / NUM_TRAIN_TIMESTEPS * multiplier
    t = ts / multiplier
    table = t if shift == 1.0 else shift * t / (1 + (shift - 1) * t)
    stride = len(table) / num_steps
    sigmas = [float(table[-(1 + int(i * stride))]) for i in range(num_steps)]
    return torch.tensor(sigmas + [0.0], dtype=torch.float32)


def step_uses_path_drop(mode: str, step: int) -> bool:
    if mode not in GUIDANCE_MODES:
        raise ValueError(f"preview guidance_mode must be one of {GUIDANCE_MODES}, got {mode!r}")
    return mode == "path_drop" or (mode == "alternate" and step % 2 == 1)


def guided_velocity(transformer, x, sigma, context, weights, guidance: float, path_drop: bool):
    """One guided velocity. ``context`` / ``weights`` hold ``[cond, uncond]`` rows (``[cond]``
    only when ``guidance == 1``); the uncond row skips the sparse blocks on path-drop steps."""
    if guidance == 1.0:
        return transformer(x, sigma.expand(1), context[:1], weights[:1])
    v = transformer(
        x.repeat(2, 1, 1, 1), sigma.expand(2), context, weights, sparse_skip=[False, path_drop]
    )
    cond, uncond = v.chunk(2)
    return apply_cfg(cond, uncond, guidance, CFG_FROM_UNCOND)


@torch.no_grad()
def generate_preview_image(pipeline, preview_cfg: dict, prompt: str, step: int, seed: int):
    from rengu_flow.model.nanosaur2.pipeline import pad_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    width, height = preview_size(preview_cfg, 16)
    num_steps, guidance, negative_prompt = preview_settings(pipeline, preview_cfg)
    mode = str(preview_cfg.get("guidance_mode", "alternate"))
    step_uses_path_drop(mode, 0)  # validate the mode before any work

    pipeline.ensure_vae_for_preview()
    prompts = [prompt] if guidance == 1.0 else [prompt, negative_prompt]
    rows = pipeline.preview_prompt_embeds(prompts, preview_cfg, device)
    context, weights, _ = pad_text([e for e, _ in rows], [w for _, w in rows])

    transformer = pipeline.transformer
    channels = transformer.in_channels
    # ComfyUI draws the initial noise on the CPU from the seed.
    noise = seeded_noise((1, channels, height // 16, width // 16), seed, device, generator_device="cpu")
    sigmas = flow_sigmas(num_steps).to(device)
    x = (noise * sigmas[0]).float()

    with _autocast(pipeline):
        param_dtype = next(transformer.parameters()).dtype
        context = context.to(device, param_dtype)
        weights = weights.to(device, param_dtype)
        for i in range(num_steps):
            sigma = sigmas[i : i + 1]
            v = guided_velocity(
                transformer, x.to(param_dtype), sigma, context, weights, guidance, step_uses_path_drop(mode, i)
            )
            x = x + (sigmas[i + 1] - sigmas[i]) * v.float()

        vae = pipeline.vae.to(device)
        image = vae.decode(x.to(vae.dtype))
    return image_to_pil(image[0])
