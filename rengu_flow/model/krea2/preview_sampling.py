"""Euler flow-matching preview sampling for Krea 2."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from rengu_flow.model.krea2.dit import pack_latents, prepare_position_ids, unpack_latents
from rengu_flow.model.krea2.text import compact_text_embeddings, encode_prompts
from rengu_flow.utils.common import round_to_nearest_multiple


def _autocast(pipeline):
    if torch.cuda.is_available():
        dtype = pipeline.model_config.get("dtype", torch.bfloat16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _shifted_sigmas(num_steps: int, image_seq_len: int, device) -> torch.Tensor:
    """linspace(1, 1/steps) grid under the Krea 2 resolution-aware exponential time shift,
    with the terminal 0.0 appended."""
    from rengu_flow.model.krea2.pipeline import calculate_shift

    mu = calculate_shift(image_seq_len)
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float64, device=device)
    sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
    return torch.cat([sigmas, sigmas.new_zeros(1)]).float()


def _denoise_step(pipeline, latents, text_states, temb_t, attn_mask, rope, text_seq_len):
    """One transformer velocity prediction, honoring the preview block-swap offloader."""
    transformer = pipeline.transformer
    offloader = getattr(pipeline, "_preview_offloader", None)
    temb_mod = transformer.time_mod_proj(F.gelu(temb_t, approximate="tanh"))
    hidden = torch.cat([text_states, transformer.img_in(latents)], dim=1)
    for i, block in enumerate(transformer.transformer_blocks):
        if offloader is not None:
            offloader.wait_for_block(i)
        hidden = block(hidden, temb_mod, rope, attn_mask)
        if offloader is not None:
            offloader.submit_move_blocks_forward(i)
    return transformer.final_layer(hidden[:, text_seq_len:], temb_t)


@torch.no_grad()
def generate_preview_image(pipeline, preview_cfg: dict, prompt: str, step: int, seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width = round_to_nearest_multiple(int(preview_cfg.get("width", 1024)), 16)
    height = round_to_nearest_multiple(int(preview_cfg.get("height", 1024)), 16)
    num_steps = int(preview_cfg.get("num_inference_steps", 28))
    guidance = float(preview_cfg.get("guidance_scale", 4.5))
    negative_prompt = str(preview_cfg.get("negative_prompt", ""))
    do_cfg = guidance > 0

    pipeline.ensure_vae_for_preview()
    pipeline.ensure_text_encoder_for_preview(device)

    with _autocast(pipeline):
        prompts = [prompt, negative_prompt] if do_cfg else [prompt]
        embeds, text_mask = encode_prompts(
            pipeline.text_encoder,
            pipeline.tokenizer,
            prompts,
            select_layers=pipeline.select_layers,
            max_sequence_length=pipeline.max_sequence_length,
            device=device,
        )
        embeds, text_mask = compact_text_embeddings(embeds, text_mask)
    pipeline.offload_text_encoder_after_encode(preview_cfg)

    transformer = pipeline.transformer
    grid_h, grid_w = height // 16, width // 16
    image_seq_len = grid_h * grid_w
    text_seq_len = text_mask.shape[1]

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, len(pipeline.vae.config.latents_mean), grid_h * 2, grid_w * 2),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    packed = pack_latents(latents)
    if do_cfg:
        packed = packed.repeat(2, 1, 1)

    sigmas = _shifted_sigmas(num_steps, image_seq_len, device)

    with _autocast(pipeline):
        param_dtype = next(transformer.parameters()).dtype
        embeds = embeds.to(device, param_dtype)
        packed = packed.to(param_dtype)
        _, attn_mask = transformer.build_attention_masks(text_mask.to(device), image_seq_len)
        position_ids = prepare_position_ids(text_seq_len, grid_h, grid_w, device)
        rope = transformer.rotary_emb(position_ids)
        # The text branch does not depend on the timestep: fuse once, reuse every step.
        text_states = transformer.txt_in(
            transformer.text_fusion(embeds, attention_mask=text_mask.to(device)[:, None, None, :])
        )

        for i in range(num_steps):
            timestep = sigmas[i].expand(packed.shape[0]).to(param_dtype)
            temb_t = transformer.time_embed(timestep, dtype=packed.dtype)
            velocity = _denoise_step(
                pipeline, packed, text_states, temb_t, attn_mask, rope, text_seq_len
            )
            if do_cfg:
                cond, uncond = velocity.chunk(2)
                velocity = cond + guidance * (cond - uncond)
                velocity = velocity.repeat(2, 1, 1)
            packed = packed + (sigmas[i + 1] - sigmas[i]).to(packed.dtype) * velocity

        packed = packed[:1]
        latents = unpack_latents(packed.float(), grid_h, grid_w)

        vae = pipeline.vae.to(device)
        mean, std = pipeline._latent_stats(device, vae.dtype)
        latents = latents.to(vae.dtype) * std + mean
        image = vae.decode(latents.unsqueeze(2)).sample.squeeze(2)

    image = ((image[0].float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    import numpy as np
    from PIL import Image

    return Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
