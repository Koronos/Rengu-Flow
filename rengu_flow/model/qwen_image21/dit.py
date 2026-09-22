# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen-Image 2.1 transformer (single-stream, block-causal), vendored from diffusers `main`
(commit 6256aa7, PR #14804 — ``models/transformers/transformer_qwenimage21.py``,
``QwenImage21Transformer2DModel``).

Vendored because the class only exists in unreleased diffusers and the pinned release is
0.37.1. Adapted for training here:

- The attention-processor indirection is removed: ``QwenImage21Attention`` runs the exact
  multi-pass SDPA prefill of upstream ``QwenImage21AttnProcessor`` inline, through
  ``F.scaled_dot_product_attention``. The ``flex_attention`` processor and its ``BlockMask``
  builder are dropped (they need a compiled model to be usable; the SDPA path is exact).
- The PEFT / LoRA-scale / cache / original-model mixins are removed (rengu attaches adapters
  through its own networks seam).
- ``forward`` is split into ``prepare_inputs`` / blocks / ``finalize`` so the training pipeline
  can partition it into layers; ``forward`` is their composition.
- RoPE positions skip padded text tokens when ``encoder_hidden_states_mask`` marks padding
  (upstream advances the position over padding too, so a padded prompt shifted the target
  image's rotary offsets relative to the unpadded one). With no padding the result is
  bit-identical to upstream.

The prefix KV cache (``QwenImage21KVCache``, ``kv_cache_mode="extract"|"cached"``) is kept for
the preview sampler. Module and parameter names are unchanged, so the official checkpoints
load verbatim via ``from_pretrained``. See NOTICE.md / THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm

# Each vision-language image slot represents a 2x2 group of latent tokens.
IMG_TOKENS_PER_SLOT = 4


class QwenImage21KVLayerCache:
    """Per-layer KV cache for text and condition-image prefix tokens.

    Stores K and V projections (post-RoPE) for the prefix extracted during the first denoising
    step. Tensor format: ``(batch_size, num_prefix_tokens, num_heads, head_dim)``.
    """

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def store(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k, self.v


class QwenImage21KVCache:
    """Container for all transformer blocks' prefix KV caches."""

    def __init__(self, num_layers: int):
        self.layer_caches = [QwenImage21KVLayerCache() for _ in range(num_layers)]

    def get_layer(self, layer_idx: int) -> QwenImage21KVLayerCache:
        return self.layer_caches[layer_idx]


def apply_rotary_emb_qwen(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Complex RoPE (upstream ``apply_rotary_emb_qwen(use_real=False)``).

    ``x``: ``(B, S, H, D)``. ``freqs_cis``: complex ``(S, D/2)`` shared by the batch, or
    ``(B, S, D/2)`` per sample.
    """
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(-2)  # broadcast over heads
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenImage21TemporalTimesteps(nn.Module):
    r"""Sinusoidal timestep embedding. `cos` occupies the first half of the channels and `sin` the second."""

    def __init__(self, timestep_dim: int, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor

        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.freqs[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)


class QwenImage21TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False
        )

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    r"""
    RMSNorm whose learnable weight is stored zero-centered: the effective scale is `weight + 1`, computed in fp32.
    Checkpoints therefore store `scale - 1`.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)


class QwenImage21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states))


class QwenImage21AdaLayerNormContinuous(nn.Module):
    r"""
    Final adaptive norm. Scale only — this variant emits no shift, so `linear` maps to `embedding_dim` rather than `2 *
    embedding_dim`.
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        scale = _select_modulation_rows(scale, target_token_mask)
        return self.norm(hidden_states) * (1 + scale)


def _select_modulation_rows(params: torch.Tensor, target_token_mask: torch.Tensor | None) -> torch.Tensor:
    r"""
    Broadcast per-sample modulation `params` over the token axis.

    With `causal_condition`, `params` holds `batch_size + 1` rows: rows `[0, batch_size)` come from the real timestep
    and the trailing row from `t = 0`. Text and condition-image tokens take the `t = 0` row, target-image tokens take
    their own sample's row. `target_token_mask` `(seq_len,)` bool; `None` disables the split.
    """
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


def _qwenimage21_prefix_segments(image_ids: torch.Tensor, prefix_len: int) -> list[tuple[int, int, bool]]:
    """Split the prefix into `(start, end, is_text)` runs of equal `image_ids` (the block-causal
    structure in the form the SDPA prefill consumes). `tolist()` is a device sync, so the model
    derives it once per forward rather than once per layer."""
    prefix_ids = image_ids[:prefix_len].tolist()
    segments = []
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or prefix_ids[index] != prefix_ids[start]:
            segments.append((start, index, prefix_ids[start] < 0))
            start = index
    return segments


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor | None):
    """SDPA over ``(B, S, H, D)`` tensors (the layout diffusers' native backend uses)."""
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attn_mask
    )
    return out.transpose(1, 2)


class QwenImage21Attention(nn.Module):
    r"""
    Attention module for [`QwenImage21TransformerBlock`]. Projection layout matches the legacy diffusers
    `Attention` so Qwen-Image 2.x checkpoints load into it unchanged.

    Prefill (``segments`` given): every prefix segment attends to the keys ``[0, end)`` (everything
    before it plus its own block); text segments additionally get a causal triangle over their own
    keys; the target image attends to everything. Padded text keys (``key_valid``) are dropped on
    every path. Decode (``segments`` is None): full attention over ``[cached prefix, target]``
    with ``attention_mask`` as the padding mask.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head
        self.use_bias = False

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def _prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        layer_cache: QwenImage21KVLayerCache | None,
        kv_cache_mode: str | None,
        cache_write_slice: slice | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        query = self.norm_q(query).to(value.dtype)
        key = self.norm_k(key).to(value.dtype)

        if rotary_emb is not None:
            query = apply_rotary_emb_qwen(query, rotary_emb)
            key = apply_rotary_emb_qwen(key, rotary_emb)

        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                # `clone()`, not `contiguous()`: at batch size 1 the prefix slice already counts as
                # contiguous, so `contiguous()` would return a view pinning the whole prefill K/V.
                layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get()
                key = torch.cat([cached_k, key], dim=1)
                value = torch.cat([cached_v, value], dim=1)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = self._prepare_qkv(
            hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )
        seq_len_q = query.shape[1]

        if segments is None:
            hidden_states = _sdpa(query, key, value, attention_mask)
        else:
            prefix_len = segments[-1][1] if segments else 0
            outputs = []
            for start, end, is_text in segments:
                seg_mask = None
                if is_text:
                    seg_len = end - start
                    seg_mask = torch.cat(
                        [
                            torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                            torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                        ],
                        dim=1,
                    )[None, None]
                if key_valid is not None:
                    seg_key_valid = key_valid[:, None, None, :end]
                    seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
                outputs.append(_sdpa(query[:, start:end], key[:, :end], value[:, :end], seg_mask))
            outputs.append(
                _sdpa(
                    query[:, prefix_len:],
                    key,
                    value,
                    None if key_valid is None else key_valid[:, None, None, :],
                )
            )
            hidden_states = torch.cat(outputs, dim=1)
        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class QwenImage21TransformerBlock(nn.Module):
    r"""
    Single-stream block. Modulation is not learned per block — the parent model computes one shared `modulation` tensor
    and every block splits its scales and gates out of it.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        mod_params: torch.Tensor,
        target_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attn_output = self.attn(
            hidden_states=img_modulated,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )
        hidden_states = hidden_states + img_gate1.tanh() * attn_output

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class QwenImage21Rope(nn.Module):
    r"""
    3-axis (frame, height, width) rotary embedding over the joint text/image sequence.

    Text tokens advance a shared position on all three axes. Every image block freezes the frame axis at the position
    reached by the preceding text and lays its tokens out on a height/width grid centred on zero, so a block's spatial
    positions do not depend on where it sits in the sequence.
    """

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        pos_index = torch.arange(8192)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.freqs = [
            torch.cat([self.rope_params(pos_index, dim, theta), self.rope_params(neg_index, dim, theta)], dim=0)
            for dim in axes_dim
        ]

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        img_shapes: list[tuple[int, int, int]],
        image_pad_mask: torch.Tensor,
        device: torch.device,
        token_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Complex ``(seq_len, sum(axes_dim)/2)`` frequencies.

        ``token_valid`` (``(seq_len,)`` bool, optional): padded text positions (``False``) do not
        advance the text position (rengu deviation; ``None`` or all-``True`` is upstream-exact).
        """
        self.freqs = [freq.to(device) for freq in self.freqs]

        frame_index, image_height_index, image_width_index = [], [], []
        cursor, position = 0, 0
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()
        valid = None if token_valid is None else token_valid.tolist()

        def text_run(start: int, end: int) -> None:
            nonlocal position
            if valid is None:
                frame_index.extend(range(position, position + end - start))
                position += end - start
                return
            for index in range(start, end):
                frame_index.append(position)
                position += int(valid[index])

        for _, height, width in img_shapes:
            block_start = is_image_token.index(True, cursor)
            text_run(cursor, block_start)

            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)

            image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

        if cursor < total_len:
            text_run(cursor, total_len)

        frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_index = frame_index.clone()
        width_index = frame_index.clone()
        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)

        return torch.cat([self.freqs[0][frame_index], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1)


class QwenImage21PreparedInputs(NamedTuple):
    """Everything the transformer blocks need, produced by ``prepare_inputs``."""

    hidden_states: torch.Tensor  # (B, S, inner_dim) joint sequence (only the target in "cached" mode)
    temb: torch.Tensor  # (B[+1], inner_dim) timestep embedding (+ t=0 row with causal_condition)
    modulation: torch.Tensor  # (B[+1], 4 * inner_dim) shared block modulation
    rotary_emb: torch.Tensor  # complex (S, D/2), or (B, S, D/2) when padding differs per sample
    modulation_mask: torch.Tensor | None  # (S,) bool target-token mask, None without causal_condition
    attention_mask: torch.Tensor | None  # decode-path padding mask (B, 1, 1, S_kv)
    cache_write_slice: slice | None
    segments: list[tuple[int, int, bool]] | None  # prefill structure; None in "cached" mode
    key_valid: torch.Tensor | None  # (B, S) bool, prefill padding mask
    prefix_len: int


class QwenImage21Transformer2DModel(ModelMixin, ConfigMixin):
    r"""
    The single-stream Transformer used by Qwen-Image 2.1.

    Text and image latents share one sequence: condition-image tokens are substituted into the text stream at the
    positions the vision-language encoder reserved for them, and the target image's tokens are appended. A single
    shared `modulation` projection feeds every block, so blocks hold no modulation parameters of their own.

    - **Block-causal attention** — attention follows `(q_idx >= kv_idx) or same_image_block`, so the sequence is causal
      while each image block stays internally bidirectional (exact multi-pass SDPA prefill).
    - `causal_condition` — text and condition-image tokens are modulated from `t = 0` instead of the sampled timestep,
      which also makes their activations timestep-independent and so cacheable across denoising steps.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImage21TransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImage21TransformerBlock"]
    _skip_keys = ["kv_cache"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=False)

        # One shared modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate].
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False

    @staticmethod
    def build_token_metadata(
        image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Label every token of the joint sequence with the image block it belongs to.

        Block boundaries come from the token counts in `img_shapes`, not from runs of `True` in `image_pad_mask`: two
        adjacent condition images must stay separate blocks.

        Returns `image_ids` `(seq_len,)` (`-1` at text positions, a unique id per image block) and
        `target_token_mask` `(seq_len,)` marking the target image's tokens.
        """
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        block_lengths = [math.prod(shape) for shape in img_shapes]
        if sum(block_lengths) != image_positions.numel():
            raise ValueError(
                f"img_shapes accounts for {sum(block_lengths)} image tokens but image_pad_mask marks "
                f"{image_positions.numel()}."
            )

        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        block_ids = torch.repeat_interleave(
            torch.arange(len(block_lengths), device=image_pad_mask.device),
            torch.tensor(block_lengths, device=image_pad_mask.device),
        )
        image_ids[image_positions] = block_ids

        target_token_mask = torch.zeros_like(image_pad_mask)
        target_token_mask[image_positions[-block_lengths[-1] :]] = True
        return image_ids, target_token_mask

    def _rotary_emb(
        self,
        img_shapes: list[tuple[int, int, int]],
        image_pad_mask: torch.Tensor,
        joint_key_valid: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if joint_key_valid is None:
            return self.pos_embed(img_shapes, image_pad_mask, device=device)
        rows = joint_key_valid.cpu()
        unique_rows, inverse = torch.unique(rows, dim=0, return_inverse=True)
        if unique_rows.shape[0] == 1:
            return self.pos_embed(img_shapes, image_pad_mask, device=device, token_valid=unique_rows[0])
        per_row = torch.stack(
            [self.pos_embed(img_shapes, image_pad_mask, device=device, token_valid=row) for row in unique_rows]
        )
        return per_row[inverse.to(device)]

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        kv_cache: QwenImage21KVCache | None = None,
        kv_cache_mode: str | None = None,
    ) -> QwenImage21PreparedInputs:
        """Embed latents/text/timestep and build the joint sequence, RoPE and attention metadata.
        Arguments as in ``forward``."""
        batch_size = hidden_states.shape[0]
        if kv_cache is not None and not self.config.causal_condition:
            raise ValueError(
                "kv_cache requires `causal_condition=True`. The cache is only valid because text and condition-image "
                "tokens modulate from t=0, which makes their activations independent of the denoising step."
            )
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError(
                f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}."
            )
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError(f"kv_cache_mode is {kv_cache_mode!r} but no kv_cache was passed to hold the prefix.")

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Each vision-language image slot stands for 2x2 latent tokens, so expand those positions four-fold and drop
        # the actual latents into them. Samples share a layout, hence the single row.
        repeats = torch.where(img_mask, IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

        target_tokens = math.prod(img_shapes[0][-1])
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2]),
            ],
            dim=1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states

        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

        timestep = timestep.to(hidden_states.dtype)
        if self.config.causal_condition:
            # Extra t=0 row; text and condition-image tokens modulate from it.
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        # Padded prompt positions must never be attended to, on any path. Text positions of the joint sequence line up,
        # in order, with the non-image positions of the vision-language sequence.
        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(
                batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

        rotary_emb = self._rotary_emb(img_shapes[0], image_pad_mask, joint_key_valid, hidden_states.device)
        prefix_len = int((~target_token_mask).sum())

        if kv_cache_mode == "cached":
            # decode: only the target image's queries are recomputed; target rows see the entire prefix + their own
            # block, so only the padding mask is needed.
            return QwenImage21PreparedInputs(
                hidden_states=joint_hidden_states[:, prefix_len:],
                temb=temb,
                modulation=modulation,
                rotary_emb=rotary_emb[..., prefix_len:, :],
                modulation_mask=None if modulation_mask is None else modulation_mask[prefix_len:],
                attention_mask=None if joint_key_valid is None else joint_key_valid[:, None, None, :],
                cache_write_slice=None,
                segments=None,
                key_valid=None,
                prefix_len=prefix_len,
            )
        return QwenImage21PreparedInputs(
            hidden_states=joint_hidden_states,
            temb=temb,
            modulation=modulation,
            rotary_emb=rotary_emb,
            modulation_mask=modulation_mask,
            attention_mask=None,
            cache_write_slice=slice(0, prefix_len) if kv_cache_mode == "extract" else None,
            segments=_qwenimage21_prefix_segments(image_ids, prefix_len),
            key_valid=joint_key_valid,
            prefix_len=prefix_len,
        )

    def finalize(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, modulation_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Final adaptive norm + output projection over the (joint or target-only) sequence."""
        return self.proj_out(self.norm_out(hidden_states, temb, modulation_mask))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        kv_cache: QwenImage21KVCache | None = None,
        kv_cache_mode: str | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states: `(batch_size, image_sequence_length, in_channels)` packed latents,
                condition images first and the target image last.
            encoder_hidden_states: `(batch_size, text_sequence_length, context_in_dim)` text
                embeddings from the vision-language encoder (right-padded).
            timestep: `(batch_size,)` current denoising step, scaled to `[0, 1]`.
            img_shapes: per-sample list of `(frame, height, width)` in latent tokens, condition
                images first and the target image last. All samples must share a layout.
            img_mask: `(batch_size, vlm_sequence_length)` bool, `True` at the vision-language
                encoder's image slots (each a `2x2` group of latent tokens), including the
                appended target slots — see ``build_t2i_img_mask``.
            encoder_hidden_states_mask: optional `(batch_size, text_sequence_length)` bool marking
                valid text tokens. Padding must be on the right.
            kv_cache / kv_cache_mode: prefix KV caching (`"extract"` on the first denoising step,
                `"cached"` afterwards). Requires `causal_condition=True`.

        Returns:
            `(batch_size, joint_sequence_length, out_channels)` (only the target tokens in
            `"cached"` mode). The target image is the trailing `h * w` tokens.
        """
        inputs = self.prepare_inputs(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_shapes,
            img_mask,
            encoder_hidden_states_mask,
            kv_cache,
            kv_cache_mode,
        )
        joint_hidden_states = inputs.hidden_states
        for index_block, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache.get_layer(index_block) if kv_cache is not None else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                joint_hidden_states = self._gradient_checkpointing_func(
                    block,
                    joint_hidden_states,
                    inputs.modulation,
                    inputs.rotary_emb,
                    inputs.attention_mask,
                    inputs.modulation_mask,
                    layer_cache,
                    kv_cache_mode,
                    inputs.cache_write_slice,
                    inputs.segments,
                    inputs.key_valid,
                )
            else:
                joint_hidden_states = block(
                    hidden_states=joint_hidden_states,
                    modulation=inputs.modulation,
                    rotary_emb=inputs.rotary_emb,
                    attention_mask=inputs.attention_mask,
                    target_token_mask=inputs.modulation_mask,
                    layer_cache=layer_cache,
                    kv_cache_mode=kv_cache_mode,
                    cache_write_slice=inputs.cache_write_slice,
                    segments=inputs.segments,
                    key_valid=inputs.key_valid,
                )

        return self.finalize(joint_hidden_states, inputs.temb, inputs.modulation_mask)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """`(B, C, H, W)` or `(B, C, 1, H, W)` latents -> `(B, H*W, C)` token sequence (2.1 is
    unpatched: packing is a plain spatial flatten, as in the upstream pipeline)."""
    if latents.ndim == 5:
        latents = latents.squeeze(2)
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(1, 2)


def unpack_latents(latents: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
    """`(B, H*W, C)` token sequence -> `(B, C, H, W)` latents."""
    b, _, c = latents.shape
    return latents.transpose(1, 2).reshape(b, c, grid_height, grid_width)


def t2i_img_shapes(grid_height: int, grid_width: int, batch_size: int) -> list[list[tuple[int, int, int]]]:
    """`img_shapes` for text-to-image (no condition images): one `(1, h, w)` target per sample."""
    return [[(1, grid_height, grid_width)]] * batch_size


def build_t2i_img_mask(
    text_seq_len: int, grid_height: int, grid_width: int, batch_size: int, device=None
) -> torch.Tensor:
    """`img_mask` for text-to-image: `text_seq_len` text slots (`False`) followed by one slot
    per 2x2 group of target latents (`True`), i.e. `(B, text_seq_len + h*w/4)` bool."""
    if grid_height % 2 or grid_width % 2:
        raise ValueError(f"latent grid must be even on both axes, got {grid_height}x{grid_width}")
    slots = grid_height * grid_width // IMG_TOKENS_PER_SLOT
    mask = torch.zeros(batch_size, text_seq_len + slots, dtype=torch.bool, device=device)
    mask[:, text_seq_len:] = True
    return mask
