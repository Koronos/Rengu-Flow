"""Wan VAE wrapper for Cosmos Predict2 latent encoding."""

from __future__ import annotations

import torch

from rengu_flow.model.cosmos_predict2.wan_vae import WanVAE_
from rengu_flow.utils.common import load_state_dict


def _video_vae(pretrained_path=None, z_dim=None, device="cpu", **kwargs):
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)
    with torch.device("meta"):
        model = WanVAE_(**cfg)
    model.load_state_dict(load_state_dict(pretrained_path), assign=True)
    return model


class WanVAE:
    def __init__(self, vae_pth=None, z_dim=16, dtype=torch.float, device="cpu"):
        self.dtype = dtype
        self.device = device
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = _video_vae(pretrained_path=vae_pth, z_dim=z_dim).eval().requires_grad_(False).to(device)


def vae_encode(tensor, vae: WanVAE):
    return vae.model.encode(tensor, vae.scale)


def vae_decode(latent, vae: WanVAE):
    """Decode a single latent (C, T, H, W) tensor to pixels."""
    import torch

    with torch.autocast("cuda", dtype=vae.dtype, enabled=torch.cuda.is_available()):
        return vae.model.decode(latent.unsqueeze(0), vae.scale).float().clamp(-1, 1).squeeze(0)


def _blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Linearly blend the bottom rows of ``a`` into the top rows of ``b`` (C, T, H, W)."""
    n = min(a.shape[-2], b.shape[-2], blend_extent)
    if n <= 0:
        return b
    w = torch.arange(n, device=b.device, dtype=b.dtype).div(n).view(1, 1, n, 1)
    b[:, :, :n, :] = a[:, :, -n:, :] * (1 - w) + b[:, :, :n, :] * w
    return b


def _blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Linearly blend the right columns of ``a`` into the left columns of ``b`` (C, T, H, W)."""
    n = min(a.shape[-1], b.shape[-1], blend_extent)
    if n <= 0:
        return b
    w = torch.arange(n, device=b.device, dtype=b.dtype).div(n).view(1, 1, 1, n)
    b[:, :, :, :n] = a[:, :, :, -n:] * (1 - w) + b[:, :, :, :n] * w
    return b


def vae_decode_tiled(latent, vae: WanVAE, *, tile_latent: int = 64, overlap_latent: int = 16):
    """Decode a (C, T, H, W) latent in overlapping spatial tiles to bound the VRAM peak.

    A full-frame decode allocates activations proportional to the output pixel area
    (hundreds of MB of contiguous VRAM at 1024x1024 in float32), which OOMs on a tight
    GPU while the DiT and training state stay resident. The decoder has no attention
    blocks (``attn_scales=[]``), so tiles decode independently; the ``overlap_latent``
    margin is linearly blended to hide tile-edge padding. A latent that fits in a single
    tile takes the plain full-decode path unchanged.
    """
    _, _, h, w = latent.shape
    if h <= tile_latent and w <= tile_latent:
        return vae_decode(latent, vae)

    sf = 2 ** (len(vae.model.dim_mult) - 1)  # spatial upscale factor (8 for Wan)
    stride = tile_latent - overlap_latent
    blend_extent = overlap_latent * sf
    keep = tile_latent * sf - blend_extent

    rows: list[list[torch.Tensor]] = []
    for i in range(0, h, stride):
        row: list[torch.Tensor] = []
        for j in range(0, w, stride):
            row.append(vae_decode(latent[:, :, i : i + tile_latent, j : j + tile_latent], vae))
        rows.append(row)

    out_rows: list[torch.Tensor] = []
    for i, row in enumerate(rows):
        out_row: list[torch.Tensor] = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = _blend_h(row[j - 1], tile, blend_extent)
            out_row.append(tile[:, :, :keep, :keep])
        out_rows.append(torch.cat(out_row, dim=-1))
    return torch.cat(out_rows, dim=-2)[:, :, : h * sf, : w * sf]
