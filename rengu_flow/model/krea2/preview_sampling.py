"""Euler flow-matching preview sampling for Krea 2."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rengu_flow.model.dit_common import preview_autocast as _autocast
from rengu_flow.model.dit_common.preview_sampler import (
    CFG_FROM_COND,
    apply_cfg,
    decode_to_pil,
    preview_settings,
    preview_size,
    run_blocks,
    seeded_noise,
    shifted_sigmas,
)
from rengu_flow.model.krea2.dit import pack_latents, prepare_position_ids, unpack_latents
from rengu_flow.model.krea2.text import pad_text_embeddings


def _shifted_sigmas(num_steps: int, image_seq_len: int, device) -> torch.Tensor:
    """linspace(1, 1/steps) grid under the Krea 2 resolution-aware exponential time shift,
    with the terminal 0.0 appended."""
    from rengu_flow.model.krea2.pipeline import calculate_shift

    return shifted_sigmas(num_steps, calculate_shift(image_seq_len), device=device)


def _denoise_step(pipeline, latents, text_states, temb_t, attn_mask, rope, text_seq_len):
    """One transformer velocity prediction, honoring the preview block-swap offloader."""
    transformer = pipeline.transformer
    temb_mod = transformer.time_mod_proj(F.gelu(temb_t, approximate="tanh"))
    hidden = torch.cat([text_states, transformer.img_in(latents)], dim=1)
    hidden = run_blocks(
        pipeline, transformer.transformer_blocks, hidden, lambda _i, block, h: block(h, temb_mod, rope, attn_mask)
    )
    return transformer.final_layer(hidden[:, text_seq_len:], temb_t)


@torch.no_grad()
def generate_preview_image(pipeline, preview_cfg: dict, prompt: str, step: int, seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width, height = preview_size(preview_cfg, 16)
    num_steps, guidance, negative_prompt = preview_settings(pipeline, preview_cfg)
    do_cfg = guidance > 0  # Krea 2's CFG: cond + g * (cond - uncond), on for any g > 0

    pipeline.ensure_vae_for_preview()

    prompts = [prompt, negative_prompt] if do_cfg else [prompt]
    rows = pipeline.preview_prompt_embeds(prompts, preview_cfg, device)
    embeds, text_mask = pad_text_embeddings(rows, [row.new_ones(row.shape[0], dtype=torch.bool) for row in rows])

    transformer = pipeline.transformer
    grid_h, grid_w = height // 16, width // 16
    image_seq_len = grid_h * grid_w
    text_seq_len = text_mask.shape[1]

    latents = seeded_noise((1, len(pipeline.vae.config.latents_mean), grid_h * 2, grid_w * 2), seed, device)
    packed = pack_latents(latents)
    if do_cfg:
        packed = packed.repeat(2, 1, 1)

    sigmas = _shifted_sigmas(num_steps, image_seq_len, device)

    with _autocast(pipeline):
        # Iterates in the DiT's parameter dtype (not fp32): what the reference sampler does.
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
                velocity = apply_cfg(*velocity.chunk(2), guidance, CFG_FROM_COND).repeat(2, 1, 1)
            packed = packed + (sigmas[i + 1] - sigmas[i]).to(packed.dtype) * velocity

        latents = unpack_latents(packed[:1].float(), grid_h, grid_w)
        return decode_to_pil(pipeline, latents, device)
