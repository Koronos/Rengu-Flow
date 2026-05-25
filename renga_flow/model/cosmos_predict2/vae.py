"""Wan VAE wrapper for Cosmos Predict2 latent encoding."""

from __future__ import annotations

import torch

from renga_flow.model.cosmos_predict2.wan_vae import WanVAE_
from renga_flow.utils.common import load_state_dict


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
