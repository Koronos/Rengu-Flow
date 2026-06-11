"""Tests for the tiled Wan VAE preview decode (CPU, tiny model)."""

from types import SimpleNamespace

import torch

from rengu_flow.model.cosmos_predict2.vae import vae_decode, vae_decode_tiled
from rengu_flow.model.cosmos_predict2.wan_vae import WanVAE_

Z_DIM = 4


def _tiny_vae() -> SimpleNamespace:
    torch.manual_seed(0)
    model = WanVAE_(
        dim=8,
        z_dim=Z_DIM,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    ).eval()
    mean = torch.zeros(Z_DIM)
    std = torch.ones(Z_DIM)
    return SimpleNamespace(model=model, scale=[mean, 1.0 / std], dtype=torch.float)


def test_tiled_decode_single_tile_is_exact_full_decode():
    vae = _tiny_vae()
    latent = torch.randn(Z_DIM, 1, 16, 16, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        full = vae_decode(latent, vae)
        tiled = vae_decode_tiled(latent, vae, tile_latent=16, overlap_latent=8)
    assert torch.equal(full, tiled)


def test_tiled_decode_shape_matches_full_decode():
    vae = _tiny_vae()
    latent = torch.randn(Z_DIM, 1, 24, 20, generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        full = vae_decode(latent, vae)
        tiled = vae_decode_tiled(latent, vae, tile_latent=16, overlap_latent=8)
    assert tiled.shape == full.shape == (3, 1, 24 * 8, 20 * 8)
    assert torch.isfinite(tiled).all()


def test_tiled_decode_close_to_full_decode():
    vae = _tiny_vae()
    latent = torch.randn(Z_DIM, 1, 24, 24, generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        full = vae_decode(latent, vae)
        tiled = vae_decode_tiled(latent, vae, tile_latent=16, overlap_latent=8)
    # Tiles see zero padding at their edges instead of real neighbors, so the outputs are
    # not bit-identical; the blended overlap must keep them visually equivalent.
    assert (tiled - full).abs().mean().item() < 0.05
