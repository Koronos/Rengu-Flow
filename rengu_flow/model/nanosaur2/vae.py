"""Nanosaur2 semantic VAE: a DINOv2-B encoder (last 6 layers + a semantic path with its own patch
embedding -> 64 latent channels, downsample 16) and an LDM convolutional decoder with a tanh
output. Ported to plain PyTorch from the official ComfyUI custom node
``nanosaur2_support/vae.py`` (``well9472/Nanosaur2-670M``, MIT License).

The DINOv2 transformer (Meta, Apache-2.0) and the LDM decoder (CompVis latent-diffusion, MIT)
are small and written out here instead of borrowed from ``transformers`` / ``diffusers``: the
encoder needs its own embedding path (antialiased bicubic position-embedding resize, a second
patch embedding) and per-layer outputs, and the decoder puts attention in its lowest up level,
which diffusers' ``Decoder`` cannot express. Parameter names follow the checkpoint
(HF-style ``encoder.encoder.layer.N...``; LDM-style ``decoder.up.N.block.M...``).

Pixels in ``[-1, 1]`` -> normalized latents ``(z - latent_mean) / latent_std`` and back.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_HIDDEN = 768
DINO_LAYERS = 12
DINO_HEADS = 12
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
DINO_PATCH_SIZE = 14
DINO_GRID = 37  # position-embedding grid of the 518 px pretraining resolution
LATENT_DOWNSAMPLE_FACTOR = 16
ENCODER_LAYERS = 6


# ---- DINOv2 ---------------------------------------------------------------------------------


class DinoSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = (proj(x).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.query, self.key, self.value))
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)


class DinoAttentionOutput(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(dim, dim)


class DinoAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attention = DinoSelfAttention(dim, heads)
        self.output = DinoAttentionOutput(dim)

    def forward(self, x):
        return self.output.dense(self.attention(x))


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.lambda1.to(x.dtype)


class DinoMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DinoLayer(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attention = DinoAttention(dim, heads)
        self.layer_scale1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = DinoMLP(dim)
        self.layer_scale2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class DinoPatchEmbeddings(nn.Module):
    def __init__(self, dim: int, patch_size: int = DINO_PATCH_SIZE) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixels):
        return self.projection(pixels).flatten(2).transpose(1, 2)


class DinoEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, DINO_GRID * DINO_GRID + 1, dim))
        self.patch_embeddings = DinoPatchEmbeddings(dim)


class DinoEncoderLayers(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList([DinoLayer(dim, heads) for _ in range(layers)])


class Dinov2(nn.Module):
    def __init__(self, dim: int = DINO_HIDDEN, heads: int = DINO_HEADS, layers: int = DINO_LAYERS) -> None:
        super().__init__()
        self.embeddings = DinoEmbeddings(dim)
        self.encoder = DinoEncoderLayers(dim, heads, layers)
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def hidden_states(self, pixels: torch.Tensor, patch_embeddings: nn.Module) -> list[torch.Tensor]:
        """Per-layer outputs (CLS first) for ``pixels`` embedded by ``patch_embeddings``; the
        position embeddings are resized to the patch grid (bicubic, antialiased, in float32)."""
        x = patch_embeddings(pixels)
        pos = self.embeddings.position_embeddings.to(device=x.device, dtype=torch.float32)
        patch_pos = pos[:, 1:].reshape(1, DINO_GRID, DINO_GRID, -1).permute(0, 3, 1, 2)
        grid = (pixels.shape[-2] // DINO_PATCH_SIZE, pixels.shape[-1] // DINO_PATCH_SIZE)
        patch_pos = F.interpolate(patch_pos, size=grid, mode="bicubic", antialias=True).flatten(2).transpose(1, 2)
        pos = torch.cat([pos[:, :1], patch_pos], dim=1).to(x.dtype)
        cls = self.embeddings.cls_token.to(device=x.device, dtype=x.dtype)
        x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1) + pos
        out = []
        for layer in self.encoder.layer:
            x = layer(x)
            out.append(x)
        return out


# ---- LDM decoder ----------------------------------------------------------------------------


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = _norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = _norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0) if in_channels != out_channels else None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, hh, ww = h.shape
        q, k, v = (proj(h).view(b, 1, c, hh * ww).transpose(2, 3) for proj in (self.q, self.k, self.v))
        out = F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(b, c, hh, ww)
        return x + self.proj_out(out)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        resolution: int = 256,
        z_channels: int = 64,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res *= 2
            self.up.insert(0, up)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for i_level in reversed(range(self.num_resolutions)):
            up = self.up[i_level]
            for i_block in range(self.num_res_blocks + 1):
                h = up.block[i_block](h)
                if len(up.attn) > 0:
                    h = up.attn[i_block](h)
            if i_level != 0:
                h = up.upsample(h)
        return torch.tanh(self.conv_out(F.silu(self.norm_out(h))))


# ---- VAE ------------------------------------------------------------------------------------

DINO_HEAD_DIM = 64  # DINOv2 ViT-S/B/L/g all use 64-wide heads


def detect_vae_config(state_dict: dict) -> dict:
    """``Nanosaur2VAE`` kwargs from a checkpoint (widths and depths read from the weights)."""

    def count(prefix: str) -> int:
        n = 0
        while any(k.startswith(f"{prefix}{n}.") for k in state_dict):
            n += 1
        return n

    dino_dim = state_dict["encoder.embeddings.cls_token"].shape[-1]
    decoder_ch = state_dict["decoder.conv_out.weight"].shape[1]
    levels = count("decoder.up.")
    ch_mult = tuple(
        state_dict[f"decoder.up.{i}.block.0.conv1.weight"].shape[0] // decoder_ch for i in range(levels)
    )
    return {
        "latent_channels": state_dict["latent_mean"].shape[1],
        "dino_dim": dino_dim,
        "dino_heads": max(1, dino_dim // DINO_HEAD_DIM),
        "dino_layers": count("encoder.encoder.layer."),
        "decoder_ch": decoder_ch,
        "decoder_ch_mult": ch_mult,
    }



class Nanosaur2VAE(nn.Module):
    downscale = LATENT_DOWNSAMPLE_FACTOR

    def __init__(
        self,
        latent_channels: int = 64,
        dino_dim: int = DINO_HIDDEN,
        dino_heads: int = DINO_HEADS,
        dino_layers: int = DINO_LAYERS,
        decoder_ch: int = 128,
        decoder_ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = Dinov2(dino_dim, dino_heads, dino_layers)
        self.semantic_patch_embeddings = DinoPatchEmbeddings(dino_dim)
        # nn.LayerNorm default eps (1e-5), as in the reference.
        self.feature_norms = nn.ModuleList([nn.LayerNorm(dino_dim) for _ in range(ENCODER_LAYERS)])
        self.encoder_projection = nn.Conv2d(dino_dim * ENCODER_LAYERS, latent_channels // 2, 1)
        self.semantic_projection = nn.Conv2d(dino_dim, latent_channels // 2, 1)
        self.decoder = Decoder(ch=decoder_ch, ch_mult=tuple(decoder_ch_mult), z_channels=latent_channels)
        self.register_buffer("latent_mean", torch.zeros(1, latent_channels, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, latent_channels, 1, 1))

    @property
    def dtype(self) -> torch.dtype:
        return self.latent_mean.dtype

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` pixels in ``[-1, 1]`` -> normalized latents ``(B, C, ~H/16, ~W/16)``."""
        b, _, h, w = x.shape
        grid_h = max(1, (h + LATENT_DOWNSAMPLE_FACTOR // 2) // LATENT_DOWNSAMPLE_FACTOR)
        grid_w = max(1, (w + LATENT_DOWNSAMPLE_FACTOR // 2) // LATENT_DOWNSAMPLE_FACTOR)
        x = F.interpolate(
            x, size=(grid_h * DINO_PATCH_SIZE, grid_w * DINO_PATCH_SIZE), mode="bicubic", align_corners=False, antialias=True
        )
        mean = torch.tensor(DINO_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(DINO_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x.add(1.0).mul(0.5) - mean) / std

        features = self.encoder.hidden_states(x, self.encoder.embeddings.patch_embeddings)[-ENCODER_LAYERS:]
        features = [norm(f[:, 1:]) for f, norm in zip(features, self.feature_norms)]
        features = torch.cat(features, dim=-1).transpose(1, 2).reshape(b, -1, grid_h, grid_w)
        semantic = self.encoder.layernorm(self.encoder.hidden_states(x, self.semantic_patch_embeddings)[-1][:, 1:])
        semantic = semantic.transpose(1, 2).reshape(b, -1, grid_h, grid_w)
        z = torch.cat([self.encoder_projection(features), self.semantic_projection(semantic)], dim=1)
        return (z - self.latent_mean.to(z.dtype)) / self.latent_std.to(z.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Normalized latents -> pixels in ``[-1, 1]``."""
        z = z * self.latent_std.to(z.dtype) + self.latent_mean.to(z.dtype)
        return self.decoder(z)
