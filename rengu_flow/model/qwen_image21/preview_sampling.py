"""Euler flow-matching preview sampling for Qwen-Image 2.1 (text-to-image and edit).

Mirrors the reference pipeline's sampler: sigmas ``linspace(1, 1/n, n)`` under the scheduler's
resolution-aware exponential shift, stretched so the last one lands on ``shift_terminal``
(``FlowMatchEulerDiscreteScheduler`` with the checkpoint's ``scheduler_config.json``), Euler
steps ``x += (sigma_next - sigma) * v``, and the prefix KV cache (text keys/values are
step-independent under ``causal_condition``: extracted on the first step, reused after). CFG is
off by default (``guidance_scale = 1``, the reference's ``true_cfg_scale``); above 1 the
negative prompt is batched with the positive one and ``v = v_neg + g * (v_pos - v_neg)``.

Edit previews (a prompt with ``control_images``): each condition image is resized with the
training helper (``rengu_flow.data.control.load_control_image``: own aspect, area
``control_resolution ** 2``, floored to 32 px), read by the text encoder as vision context and
VAE-encoded (mode) into the latent blocks that precede the noise, like the reference. The output
takes the last condition image's aspect at the ``width * height`` area (the reference's default
when no size is given). The KV-cache prefix then holds the prompt and the condition images.
"""

from __future__ import annotations

import math

import torch

from rengu_flow.model.dit_common import pad_text_embeddings, preview_autocast, preview_compute_dtype
from rengu_flow.model.qwen_image21.dit import (
    QwenImage21KVCache,
    build_t2i_img_mask,
    pack_latents,
    t2i_img_shapes,
    unpack_latents,
)
from rengu_flow.utils.common import round_to_nearest_multiple

# Condition-image and output sizes are multiples of 32 px (16x VAE, 2x2 latents per vision slot).
CONDITION_MULTIPLE = 32

# Qwen/Qwen-Image-2.1 scheduler/scheduler_config.json.
SHIFT_TERMINAL = 0.02
# Preview VAE decode tiling (pixels): tiles kick in above 512 px on either side.
PREVIEW_TILE = 512
PREVIEW_TILE_STRIDE = 448


def shifted_sigmas(num_steps: int, image_seq_len: int, device="cpu") -> torch.Tensor:
    """The ``num_steps + 1`` sigmas (terminal 0 appended) of the reference scheduler."""
    from rengu_flow.model.qwen_image21.pipeline import calculate_shift

    mu = calculate_shift(image_seq_len)
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float64)
    sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
    one_minus = 1 - sigmas
    sigmas = 1 - one_minus / (one_minus[-1] / (1 - SHIFT_TERMINAL))
    return torch.cat([sigmas, sigmas.new_zeros(1)]).float().to(device)


def denoise_step(
    pipeline,
    hidden_states: torch.Tensor,
    embeds: torch.Tensor,
    timestep: torch.Tensor,
    img_shapes,
    img_mask: torch.Tensor,
    encoder_mask: torch.Tensor | None,
    kv_cache: QwenImage21KVCache | None = None,
    kv_cache_mode: str | None = None,
    target_len: int | None = None,
) -> torch.Tensor:
    """One velocity prediction for the target tokens ``(B, h*w, C)``, honoring the preview
    block-swap offloader. Equals ``transformer(...)[:, -h*w:]``. ``hidden_states`` is
    ``[condition latents | target]``; ``target_len`` (default: all of it) is the target's
    ``h*w``."""
    transformer = pipeline.transformer
    offloader = getattr(pipeline, "_preview_offloader", None)
    prepared = transformer.prepare_inputs(
        hidden_states, embeds, timestep, img_shapes, img_mask, encoder_mask, kv_cache, kv_cache_mode
    )
    hidden = prepared.hidden_states
    for i, block in enumerate(transformer.transformer_blocks):
        if offloader is not None:
            offloader.wait_for_block(i)
        hidden = block(
            hidden,
            prepared.modulation,
            prepared.rotary_emb,
            prepared.attention_mask,
            prepared.modulation_mask,
            kv_cache.get_layer(i) if kv_cache is not None else None,
            kv_cache_mode,
            prepared.cache_write_slice,
            prepared.segments,
            prepared.key_valid,
        )
        if offloader is not None:
            offloader.submit_move_blocks_forward(i)
    n = hidden_states.shape[1] if target_len is None else target_len
    mask = None if prepared.modulation_mask is None else prepared.modulation_mask[-n:]
    return transformer.finalize(hidden[:, -n:], prepared.temb, mask)


def preview_control_resolution(preview_cfg: dict) -> int:
    """Side of the condition images' target area: ``control_resolution`` when set, else the side
    of the preview's ``width * height`` area (the reference sizes both from one resolution)."""
    if preview_cfg.get("control_resolution"):
        return int(preview_cfg["control_resolution"])
    return round(math.sqrt(int(preview_cfg.get("width", 1024)) * int(preview_cfg.get("height", 1024))))


def _image_to_tensor(img) -> torch.Tensor:
    """PIL RGB/RGBA -> ``(1, C, H, W)`` float in ``[-1, 1]``."""
    import numpy as np

    arr = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)
    return (arr / 127.5 - 1.0).unsqueeze(0)


@torch.no_grad()
def generate_preview_image(
    pipeline, preview_cfg: dict, prompt: str, step: int, seed: int, control_images: list | None = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 16x VAE and 2x2 latent groups per vision slot: pixels must be multiples of 32.
    width = round_to_nearest_multiple(int(preview_cfg.get("width", 1024)), 32)
    height = round_to_nearest_multiple(int(preview_cfg.get("height", 1024)), 32)
    num_steps = int(preview_cfg.get("num_inference_steps", 28))
    guidance = float(preview_cfg.get("guidance_scale", 1.0))
    negative_prompt = str(preview_cfg.get("negative_prompt", ""))
    do_cfg = guidance > 1.0

    pipeline.ensure_vae_for_preview()
    prompts = [prompt, negative_prompt] if do_cfg else [prompt]
    conditions = None
    if control_images:
        from rengu_flow.data.control import control_size, load_control_image, read_control_dims

        resolution = preview_control_resolution(preview_cfg)
        conditions = [load_control_image(path, resolution, CONDITION_MULTIPLE) for path in control_images]
        # The output takes the last condition image's (source) aspect at the preview's area.
        width, height = control_size(
            *read_control_dims(control_images[-1]), math.sqrt(width * height), CONDITION_MULTIPLE
        )
        per_prompt = pipeline.preview_edit_prompt_embeds(
            prompts, control_images, conditions, resolution, preview_cfg, device
        )
        embeds, text_mask = pad_text_embeddings(
            [e for e, _ in per_prompt], [torch.ones(e.shape[0], dtype=torch.bool) for e, _ in per_prompt]
        )
        image_pad_mask, _ = pad_text_embeddings([m for _, m in per_prompt], [m for _, m in per_prompt])
        image_pad_mask = image_pad_mask.bool()
        if not bool((image_pad_mask == image_pad_mask[:1]).all()):
            raise ValueError("qwen_image21 preview: positive and negative prompts disagree on the image layout")
    else:
        per_prompt = pipeline.preview_prompt_embeds(prompts, preview_cfg, device)
        embeds, text_mask = pad_text_embeddings(
            [e for e, _ in per_prompt], [torch.ones(e.shape[0], dtype=torch.bool) for e, _ in per_prompt]
        )

    transformer = pipeline.transformer
    compute_dtype = preview_compute_dtype(pipeline)
    grid_h, grid_w = height // 16, width // 16
    batch = len(prompts)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, transformer.config.in_channels, grid_h, grid_w),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    packed = pack_latents(latents)
    sigmas = shifted_sigmas(num_steps, grid_h * grid_w, device)

    embeds = embeds.to(device, compute_dtype)
    text_mask = text_mask.to(device)
    encoder_mask = None if bool(text_mask.all()) else text_mask
    condition_packed = None
    if conditions is None:
        img_mask = build_t2i_img_mask(embeds.shape[1], grid_h, grid_w, batch, device=device)
        img_shapes = t2i_img_shapes(grid_h, grid_w, batch)
    else:
        vae = pipeline.vae.to(device)
        was_tiling = vae.use_tiling
        vae.enable_tiling(PREVIEW_TILE, PREVIEW_TILE, PREVIEW_TILE_STRIDE, PREVIEW_TILE_STRIDE)
        try:
            condition_latents = [
                pipeline.encode_condition_latents(vae, _image_to_tensor(img).to(device)) for img in conditions
            ]
        finally:
            vae.use_tiling = was_tiling
        condition_packed = torch.cat([pack_latents(c) for c in condition_latents], dim=1)
        img_shapes = [[*[(1, c.shape[-2], c.shape[-1]) for c in condition_latents], (1, grid_h, grid_w)]] * batch
        target_slots = torch.ones(batch, grid_h * grid_w // 4, dtype=torch.bool)
        img_mask = torch.cat([image_pad_mask, target_slots], dim=1).to(device)
    kv_cache = QwenImage21KVCache(len(transformer.transformer_blocks))

    with preview_autocast(pipeline):
        for i in range(num_steps):
            # The reference feeds the scheduler timestep (sigma * 1000) in the latents' dtype,
            # divided by 1000.
            timestep = (sigmas[i] * 1000).expand(batch).to(compute_dtype) / 1000
            hidden = packed.expand(batch, -1, -1).to(compute_dtype)
            if condition_packed is not None:
                hidden = torch.cat([condition_packed.expand(batch, -1, -1).to(compute_dtype), hidden], dim=1)
            velocity = denoise_step(
                pipeline,
                hidden,
                embeds,
                timestep,
                img_shapes,
                img_mask,
                encoder_mask,
                kv_cache,
                "extract" if i == 0 else "cached",
                target_len=packed.shape[1],
            ).float()
            if do_cfg:
                cond, uncond = velocity[:1], velocity[1:]
                velocity = uncond + guidance * (cond - uncond)
            packed = packed + (sigmas[i + 1] - sigmas[i]) * velocity
        del kv_cache

        latents = unpack_latents(packed, grid_h, grid_w)
        vae = pipeline.vae.to(device)
        mean, std = pipeline._latent_stats(device, vae.dtype)
        latents = latents.to(vae.dtype) * std + mean
        # An untiled 1024x1024 decode needs ~6.6 GB of activations (measured, bf16) — more
        # than an 8 GB card has left next to the DiT. 512 px tiles with a 64 px blended overlap
        # cap it at ~1.7 GB; smaller images decode in one piece exactly as before.
        was_tiling = vae.use_tiling
        vae.enable_tiling(PREVIEW_TILE, PREVIEW_TILE, PREVIEW_TILE_STRIDE, PREVIEW_TILE_STRIDE)
        try:
            image = vae.decode(latents.unsqueeze(2)).sample[:, :, 0]
        finally:
            vae.use_tiling = was_tiling

    rgba = (image[0].float().clamp(-1, 1) + 1) / 2  # (4, H, W) in [0, 1]
    # The VAE decodes RGBA; show it composited over white (what an image viewer does).
    rgb = rgba[:3] * rgba[3:4] + (1 - rgba[3:4])
    rgb = (rgb * 255).round().to(torch.uint8)
    from PIL import Image

    return Image.fromarray(rgb.permute(1, 2, 0).cpu().numpy())
