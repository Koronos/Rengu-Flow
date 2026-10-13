"""Nanosaur2 DiT (adaLN-single, 2D RoPE, SwiGLU, QK-norm, SPRINT sparse middle blocks,
x-prediction), ported to plain PyTorch from the official ComfyUI custom node
``nanosaur2_support/model.py`` (``well9472/Nanosaur2-670M``, MIT License).

Module and parameter names are the official ones (``blocks.N.attn.qkv_x``,
``text_refine_blocks.N.adaLN_modulation.0``, ``encoder_adaLN_offsets.N`` ...), so the released
``nanosaur2_diffusion_model.safetensors`` loads verbatim, full finetunes save in the same layout
and adapter keys address the same paths ComfyUI's LoRA loader does.

Differences from the reference, none of them numeric:

* ComfyUI ``operations`` / ``optimized_attention`` / ``comfy_kitchen`` fused kernels are replaced
  by ``torch.nn`` layers and ``F.scaled_dot_product_attention`` (the reference's own training path,
  ``comfy.model_management.in_training``, uses the same unfused math as here).
* ``forward`` is split into ``embed`` / ``run_block`` / ``sparse_path`` / ``final`` so the
  DeepSpeed pipeline layers (``layers.py``) run exactly the code the monolithic forward runs.
* The x-prediction to velocity conversion ``(x - x0) / t`` runs in float32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SPRINT_NUM_F = 2  # dense blocks before the sparse (skippable) middle path
SPRINT_NUM_H = 2  # dense blocks after it
TEXT_EMBED_DIM = 640  # Gemma3-270M hidden size
FREQUENCY_EMBEDDING_SIZE = 256


class RMSNorm(nn.Module):
    """Plain-scale RMSNorm (``torch.nn.RMSNorm`` semantics; the weight is cast to the input dtype,
    as ComfyUI's ``comfy.rmsnorm.rms_norm`` does)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight.to(x.dtype), self.eps)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding, cos first (``comfy.ldm.modules.diffusionmodules.util``)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = FREQUENCY_EMBEDDING_SIZE) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size).to(dtype))


def rope_2d(head_dim: int, height: int, width: int, device) -> torch.Tensor:
    """Split-half 2D RoPE over centered token coordinates (x frequencies, then y), as
    ``(1, N, 1, head_dim // 2, 2, 2)`` float32 rotations."""
    axis_dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim))
    y = torch.arange(height, dtype=torch.float32, device=device) - (height - 1) / 2
    x = torch.arange(width, dtype=torch.float32, device=device) - (width - 1) / 2
    y, x = torch.meshgrid(y, x, indexing="ij")
    angles = torch.cat([torch.outer(x.flatten(), inv_freq), torch.outer(y.flatten(), inv_freq)], dim=-1)
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.stack([cos, -sin, sin, cos], dim=-1).view(1, height * width, 1, axis_dim, 2, 2)


def rope_split_half(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Pair k is ``(x[k], x[k + head_dim // 2])``."""
    x_ = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(pe.dtype)
    return (pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]).movedim(-1, -2).reshape(x.shape).type_as(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.addcmul(shift, x, 1 + scale)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``(B, L, H, D)`` q/k/v -> ``(B, Lq, H * D)``; ``mask`` is additive ``(B, 1, 1, Lk)``."""
    if mask is not None:
        mask = mask.to(q.dtype)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class Embed(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, norm: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size, eps=1e-6) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Attention(nn.Module):
    """Self-attention over image tokens with 2D RoPE, optionally joint with (non-rotated) text
    keys/values (``kv_y``, even blocks only)."""

    def __init__(self, dim: int, num_heads: int, cross_attention: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=False)
        self.kv_y = nn.Linear(dim, dim * 2, bias=False) if cross_attention else None
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, txt, pe, txt_bias):
        b, n, _ = x.shape
        q, k, v = self.qkv_x(x).view(b, n, 3, self.num_heads, self.head_dim).unbind(2)
        q, k = rope_split_half(self.q_norm(q), pe), rope_split_half(self.k_norm(k), pe)
        mask = None
        if self.kv_y is not None:
            ky, vy = self.kv_y(txt).view(b, -1, 2, self.num_heads, self.head_dim).unbind(2)
            k = torch.cat([k, self.k_norm(ky)], dim=1)
            v = torch.cat([v, vy], dim=1)
            # Image keys are always visible; text keys carry log(emphasis weight), -inf for padding.
            mask = F.pad(txt_bias, (n, 0))[:, None, None, :]
        return self.proj(_sdpa(q, k, v, mask))


class DiTBlock(nn.Module):
    """Image block; its adaLN modulation is computed by the caller (adaLN-single)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_hidden: int, cross_attention: bool) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, cross_attention)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mlp_hidden)

    def forward(self, x, txt, pe, mod, txt_bias):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        x = torch.addcmul(x, gate_msa, self.attn(modulate(self.norm1(x), shift_msa, scale_msa), txt, pe, txt_bias))
        return torch.addcmul(x, gate_mlp, self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))


class TextRefineAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, mask):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).unbind(2)
        return self.proj(_sdpa(self.q_norm(q), self.k_norm(k), v, mask))


class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_hidden: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = TextRefineAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mlp_hidden)
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, mask, keep):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = torch.addcmul(x, gate_msa, self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask))
        x = torch.addcmul(x, gate_mlp, self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x * keep


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


def run_block(block, offset, s, txt, pe, mod, txt_bias):
    """One image block with its per-block adaLN offset added to the shared modulation."""
    return block(s, txt, pe, mod + offset.to(device=mod.device, dtype=mod.dtype), txt_bias)


def text_masks(token_weights: torch.Tensor, dtype: torch.dtype):
    """``(txt_bias, refine_mask, keep)`` from per-token emphasis weights ``(B, L)`` (0 = padding).

    ``txt_bias`` is log(weight) (-inf on padding) for the image blocks' joint attention; the
    text-refine attention is a hard mask, and a row with no text key keeps its first one (its
    tokens are zeroed afterwards by ``keep``)."""
    keep = token_weights > 0
    txt_bias = torch.log(token_weights.clamp(min=1e-4)).masked_fill(~keep, float("-inf"))
    refine_keep = keep.clone()
    refine_keep[:, 0] |= ~keep.any(dim=1)
    refine_mask = torch.zeros_like(txt_bias).masked_fill(~refine_keep, float("-inf"))[:, None, None, :]
    return txt_bias, refine_mask, keep.unsqueeze(-1).to(dtype)


class Nanosaur2Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1536,
        num_heads: int = 16,
        num_blocks: int = 18,
        num_text_blocks: int = 2,
        mlp_hidden: int = 4096,
        text_embed_dim: int = TEXT_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_blocks": num_blocks,
            "num_text_blocks": num_text_blocks,
            "mlp_hidden": mlp_hidden,
            "text_embed_dim": text_embed_dim,
        }
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.s_embedder = Embed(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = Embed(text_embed_dim, hidden_size, norm=True)
        self.shared_encoder_adaLN = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.encoder_adaLN_offsets = nn.ParameterList(
            [nn.Parameter(torch.zeros(6 * hidden_size)) for _ in range(num_blocks)]
        )
        self.y_pool_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sprint_out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_hidden, i % 2 == 0) for i in range(num_blocks)]
        )
        self.text_refine_blocks = nn.ModuleList(
            [TextRefineBlock(hidden_size, num_heads, mlp_hidden) for _ in range(num_text_blocks)]
        )
        self.final_layer = FinalLayer(hidden_size, in_channels)

    @property
    def sparse_range(self) -> range:
        """Indices of the SPRINT sparse (skippable) middle blocks."""
        return range(SPRINT_NUM_F, len(self.blocks) - SPRINT_NUM_H)

    # ---- pieces (shared by forward and the pipeline layers) --------------------------------

    def embed(self, x, timestep, context, token_weights):
        """Text refinement, conditioning and image embedding.

        Returns ``(s, txt, pe, mod, txt_bias, condition)``: image tokens ``(B, N, H)``, refined
        text ``(B, L, H)``, RoPE table, shared adaLN modulation ``(B, 1, 6H)``, text attention
        bias ``(B, L)`` and the final-layer condition ``(B, 1, H)``."""
        _, _, h, w = x.shape
        dtype = x.dtype
        txt_bias, refine_mask, keep = text_masks(token_weights, dtype)

        t = self.t_embedder(timestep * 1000.0, dtype).unsqueeze(1)
        txt = self.y_embedder(context)
        time_condition = F.silu(t)
        for block in self.text_refine_blocks:
            txt = block(txt, time_condition, refine_mask, keep)

        weights = token_weights.unsqueeze(-1).to(txt.dtype)
        pooled = (txt * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        condition = F.silu(t + self.y_pool_proj(pooled).unsqueeze(1))
        mod = self.shared_encoder_adaLN(condition)

        pe = rope_2d(self.head_dim, h, w, x.device)
        s = self.s_embedder(x.flatten(2).transpose(1, 2))
        return s, txt, pe, mod, txt_bias, condition

    def run_block(self, i, s, txt, pe, mod, txt_bias):
        return run_block(self.blocks[i], self.encoder_adaLN_offsets[i], s, txt, pe, mod, txt_bias)

    def sparse_path(self, s, txt, pe, mod, txt_bias):
        g = s
        for i in self.sparse_range:
            g = self.run_block(i, g, txt, pe, mod, txt_bias)
        return s + self.sprint_out_proj(g - s)

    def final(self, s, condition, x, timestep):
        """x-prediction head -> velocity ``(x - x0) / t`` (float32)."""
        b, _, h, w = x.shape
        x0 = self.final_layer(s, condition).transpose(1, 2).reshape(b, self.in_channels, h, w)
        return (x.float() - x0.float()) / timestep.float().view(-1, 1, 1, 1)

    def forward(self, x, timestep, context, token_weights, sparse_skip=None):
        """``sparse_skip``: optional per-row flags (sequence or bool tensor); flagged rows skip the
        sparse middle blocks (path-drop). Returns the flow velocity, float32."""
        s, txt, pe, mod, txt_bias, condition = self.embed(x, timestep, context, token_weights)
        for i in range(SPRINT_NUM_F):
            s = self.run_block(i, s, txt, pe, mod, txt_bias)

        skip = None if sparse_skip is None else torch.as_tensor(sparse_skip, dtype=torch.bool).tolist()
        if skip is None or not any(skip):
            s = self.sparse_path(s, txt, pe, mod, txt_bias)
        elif not all(skip):
            rows = [i for i, flag in enumerate(skip) if not flag]
            s = s.clone()
            s[rows] = self.sparse_path(s[rows], txt[rows], pe, mod[rows], txt_bias[rows])

        for i in range(len(self.blocks) - SPRINT_NUM_H, len(self.blocks)):
            s = self.run_block(i, s, txt, pe, mod, txt_bias)
        return self.final(s, condition, x, timestep)


def detect_config(state_dict: dict) -> dict:
    """Architecture kwargs from a checkpoint (the reference ``detect_unet_config``), so larger
    Nanosaur2 releases load without code changes."""

    def count(prefix: str) -> int:
        n = 0
        while any(k.startswith(f"{prefix}{n}.") for k in state_dict):
            n += 1
        return n

    hidden_size = state_dict["s_embedder.proj.weight"].shape[0]
    return {
        "in_channels": state_dict["final_layer.linear.weight"].shape[0],
        "hidden_size": hidden_size,
        "num_heads": hidden_size // state_dict["blocks.0.attn.q_norm.weight"].shape[0],
        "num_blocks": count("blocks."),
        "num_text_blocks": count("text_refine_blocks."),
        "mlp_hidden": state_dict["blocks.0.mlp.w12.weight"].shape[0] // 2,
        "text_embed_dim": state_dict["y_embedder.proj.weight"].shape[1],
    }
