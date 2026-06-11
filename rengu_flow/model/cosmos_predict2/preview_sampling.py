"""Euler flow-matching preview sampling for Cosmos Predict2 (image, T=1)."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from rengu_flow.model.cosmos_predict2.layers import NoopOffloader
from rengu_flow.model.cosmos_predict2.text import compute_text_embeddings, tokenize
from rengu_flow.model.cosmos_predict2.vae import vae_decode_tiled
from rengu_flow.utils.common import round_to_nearest_multiple

# Wan VAE spatial compression (three stride-2 downsamples in the encoder).
WAN_VAE_SPATIAL_FACTOR = 8


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


@dataclass
class PreviewPromptData:
    crossattn_emb: torch.Tensor
    attn_mask: torch.Tensor
    t5_input_ids: torch.Tensor
    t5_attn_mask: torch.Tensor


def _model_compute_dtype(pipeline) -> torch.dtype:
    dtype = pipeline.model_config.get("dtype", torch.bfloat16)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return dtype


def _preview_autocast(pipeline):
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=_model_compute_dtype(pipeline))
    return nullcontext()


def round_preview_pixels(value: int, multiple: int = 16) -> int:
    return max(multiple, round_to_nearest_multiple(int(value), multiple))


def apply_timestep_shift(t: torch.Tensor, h_lat: int, w_lat: int, model_config: dict) -> torch.Tensor:
    if shift := model_config.get("shift"):
        return (t * shift) / (1 + (shift - 1) * t)
    if model_config.get("flux_shift", False):
        mu = get_lin_function(y1=0.5, y2=1.15)((h_lat // 2) * (w_lat // 2))
        return time_shift(mu, 1.0, t)
    return t


def build_timestep_schedule(
    num_steps: int,
    h_lat: int,
    w_lat: int,
    model_config: dict,
    device: torch.device,
) -> torch.Tensor:
    """Monotonic schedule from t=1 (noise) to t=0 (data), length num_steps + 1."""
    raw = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return apply_timestep_shift(raw, h_lat, w_lat, model_config)


def latent_shape_from_pixels(
    pipeline, width: int, height: int, device: torch.device
) -> tuple[int, int, int, int]:
    """Return (C, T, H_lat, W_lat) from Wan VAE geometry (no encode pass)."""
    _ = device
    z_dim = getattr(getattr(pipeline, "vae", None), "model", None)
    channels = getattr(z_dim, "z_dim", 16) if z_dim is not None else 16
    h_lat = max(1, height // WAN_VAE_SPATIAL_FACTOR)
    w_lat = max(1, width // WAN_VAE_SPATIAL_FACTOR)
    return channels, 1, h_lat, w_lat


def encode_preview_prompt(pipeline, caption: str, device: torch.device) -> PreviewPromptData:
    batch_encoding = tokenize(pipeline.tokenizer, [caption])
    t5_batch_encoding = tokenize(pipeline.t5_tokenizer, [caption])
    input_ids = batch_encoding.input_ids.to(device)
    attn_mask = batch_encoding.attention_mask.to(device)
    t5_input_ids = t5_batch_encoding.input_ids.to(device)
    t5_attn_mask = t5_batch_encoding.attention_mask.to(device)
    with torch.no_grad():
        crossattn_emb = compute_text_embeddings(pipeline.text_encoder, input_ids, attn_mask)
    return PreviewPromptData(
        crossattn_emb=crossattn_emb,
        attn_mask=attn_mask,
        t5_input_ids=t5_input_ids,
        t5_attn_mask=t5_attn_mask,
    )


def _apply_llm_adapter(pipeline, prompt_data: PreviewPromptData) -> torch.Tensor:
    crossattn_emb = prompt_data.crossattn_emb
    if pipeline.use_llm_adapter:
        crossattn_emb = pipeline.transformer.llm_adapter(
            source_hidden_states=crossattn_emb,
            target_input_ids=prompt_data.t5_input_ids,
            target_attention_mask=prompt_data.t5_attn_mask,
            source_attention_mask=prompt_data.attn_mask,
        )
        crossattn_emb = crossattn_emb.clone()
        crossattn_emb[~prompt_data.t5_attn_mask.bool()] = 0
    return crossattn_emb


def forward_transformer(
    pipeline,
    x: torch.Tensor,
    timesteps_B_T: torch.Tensor,
    crossattn_emb: torch.Tensor,
) -> torch.Tensor:
    transformer = pipeline.transformer
    offloader = getattr(pipeline, "_preview_offloader", None)
    with _preview_autocast(pipeline):
        if offloader is None or isinstance(offloader, NoopOffloader):
            padding_mask = torch.zeros(
                x.shape[0], 1, x.shape[3], x.shape[4], dtype=x.dtype, device=x.device
            )
            return transformer(x, timesteps_B_T, crossattn_emb, fps=None, padding_mask=padding_mask)

        padding_mask = torch.zeros(
            x.shape[0], 1, x.shape[3], x.shape[4], dtype=x.dtype, device=x.device
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = transformer.prepare_embedded_sequence(
            x, fps=None, padding_mask=padding_mask,
        )
        assert extra_pos_emb is None

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = transformer.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = transformer.t_embedding_norm(t_embedding_B_T_D)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
        }
        for block_idx, block in enumerate(transformer.blocks):
            offloader.wait_for_block(block_idx)
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )
            offloader.submit_move_blocks_forward(block_idx)

        x_B_T_H_W_O = transformer.final_layer(
            x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D
        )
        return transformer.unpatchify(x_B_T_H_W_O)


def _predict_velocity(
    pipeline,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    crossattn_emb: torch.Tensor,
    crossattn_uncond: torch.Tensor | None,
    guidance_scale: float,
) -> torch.Tensor:
    if crossattn_uncond is None or guidance_scale == 1.0:
        return forward_transformer(pipeline, x, t_batch, crossattn_emb)
    x2 = torch.cat([x, x], dim=0)
    t2 = t_batch.expand(2, -1) if t_batch.shape[0] == 1 else t_batch
    emb2 = torch.cat([crossattn_emb, crossattn_uncond], dim=0)
    v2 = forward_transformer(pipeline, x2, t2, emb2)
    v_cond, v_uncond = v2.chunk(2, dim=0)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


def euler_sample_latents(
    pipeline,
    prompt_data: PreviewPromptData,
    width: int,
    height: int,
    num_steps: int,
    seed: int,
    device: torch.device,
    *,
    uncond_prompt_data: PreviewPromptData | None = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    pipeline.ensure_transformer_for_preview(device)
    dtype = _model_compute_dtype(pipeline)
    c, t_frames, h_lat, w_lat = latent_shape_from_pixels(pipeline, width, height, device)
    schedule = build_timestep_schedule(num_steps, h_lat, w_lat, pipeline.model_config, device)
    crossattn_emb = _apply_llm_adapter(pipeline, prompt_data)
    crossattn_uncond = None
    if uncond_prompt_data is not None and guidance_scale != 1.0:
        crossattn_uncond = _apply_llm_adapter(pipeline, uncond_prompt_data)

    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, c, t_frames, h_lat, w_lat, generator=generator, device=device, dtype=dtype)

    with torch.inference_mode(), _preview_autocast(pipeline):
        for i in range(num_steps):
            t_val = schedule[i]
            t_next = schedule[i + 1]
            dt = (t_val - t_next).to(dtype=dtype)
            t_batch = t_val.view(1, 1).to(dtype=dtype)
            v_pred = _predict_velocity(
                pipeline, x, t_batch, crossattn_emb, crossattn_uncond, guidance_scale
            )
            x = x - dt * v_pred
            if (i + 1) % 5 == 0 or i + 1 == num_steps:
                print(f"rengu_flow: preview Euler {i + 1}/{num_steps}", flush=True)

    return x


def decode_latents_to_pil(pipeline, latents: torch.Tensor):
    """Decode (1, C, T, H, W) latents to a PIL RGB image (first frame)."""
    from PIL import Image

    device = latents.device
    dtype = _model_compute_dtype(pipeline)
    pipeline.ensure_vae_for_preview()
    pipeline.vae.model.to(device)
    latent = latents[0].to(dtype=dtype)
    with torch.inference_mode():
        # Tiled decode bounds the conv3d activation peak so the decode fits next to the
        # resident DiT + training state (the DiT can no longer be offloaded for the decode —
        # see offload_transformer_for_decode). Single-tile latents decode exactly as before.
        decoded = vae_decode_tiled(latent, pipeline.vae)
    frame = decoded[:, 0].cpu()
    frame = (frame + 1.0) * 0.5
    frame = (frame * 255.0).round().byte()
    arr = frame.permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def generate_preview_image(
    pipeline,
    preview_cfg: dict,
    prompt: str,
    step: int,
    seed: int,
) -> Any:
    device = torch.device("cuda", index=0)
    width = round_preview_pixels(preview_cfg.get("width", 1024), pipeline.pixels_round_to_multiple)
    height = round_preview_pixels(preview_cfg.get("height", 1024), pipeline.pixels_round_to_multiple)
    num_steps = int(preview_cfg.get("num_inference_steps", 20))
    guidance_scale = float(preview_cfg.get("guidance_scale", 4.0))
    negative_prompt = preview_cfg.get("negative_prompt", "")

    print("rengu_flow: encoding preview prompt...", flush=True)
    pipeline.ensure_text_encoder_for_preview(device)
    prompt_data = encode_preview_prompt(pipeline, prompt, device)
    uncond_data = None
    if guidance_scale != 1.0:
        uncond_data = encode_preview_prompt(pipeline, negative_prompt or "", device)
    pipeline.offload_text_encoder_after_encode(preview_cfg)
    print(f"rengu_flow: Euler sampling ({num_steps} steps, cfg={guidance_scale})...", flush=True)

    latents = euler_sample_latents(
        pipeline,
        prompt_data,
        width,
        height,
        num_steps,
        seed,
        device,
        uncond_prompt_data=uncond_data,
        guidance_scale=guidance_scale,
    )
    print("rengu_flow: VAE decode for preview...", flush=True)
    # Free the DiT from VRAM before decoding — the VAE decoder's conv3d needs a large contiguous
    # block and the DiT is not used here. Off by default (opt-in via preview_offload_dit_for_decode):
    # the CPU<->GPU round-trip corrupts DeepSpeed/compiled param storage. No-op when block swap
    # manages residency.
    pipeline.offload_transformer_for_decode(preview_cfg)
    image = decode_latents_to_pil(pipeline, latents)
    print("rengu_flow: preview image ready.", flush=True)
    return image
