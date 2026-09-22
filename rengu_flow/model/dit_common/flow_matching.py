"""Rectified-flow training math shared by the DiT pipelines (krea2, cosmos_predict2, ...).

Convention: ``t = 1`` is pure noise, ``t = 0`` is data; the model predicts the velocity
``noise - latents``. Each pipeline keeps its own dynamic-shift constants and decides
*whether* a dynamic shift applies; these helpers only do the arithmetic.
"""

from __future__ import annotations

import math

import torch


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Linear interpolation of the resolution-aware shift ``mu`` (diffusers' ``calculate_shift``)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """Exponential time shift (diffusers' FlowMatchEuler ``time_shift``)."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def sample_timesteps(
    model_config: dict,
    batch_size: int,
    device: torch.device,
    timestep_quantile: float | None = None,
) -> torch.Tensor:
    """Draw unshifted training timesteps in (0, 1), shape ``(batch_size,)``.

    ``model.timestep_sample_method``: ``logit_normal`` (default; sigmoid of a normal scaled by
    ``model.sigmoid_scale``) or ``uniform``. ``timestep_quantile`` picks a fixed quantile of the
    distribution instead of sampling (validation probes).
    """
    method = model_config.get("timestep_sample_method", "logit_normal")
    if method == "logit_normal":
        dist = torch.distributions.normal.Normal(0, 1)
    elif method == "uniform":
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise NotImplementedError()

    if timestep_quantile is not None:
        t = dist.icdf(torch.full((batch_size,), timestep_quantile, device=device))
    else:
        t = dist.sample((batch_size,)).to(device)

    if method == "logit_normal":
        sigmoid_scale = model_config.get("sigmoid_scale", 1.0)
        t = torch.sigmoid(t * sigmoid_scale)
    return t


def shift_timesteps(t: torch.Tensor, shift: float | None, mu: float | None = None) -> torch.Tensor:
    """A truthy fixed ``shift`` (``model.shift``) wins; else apply the exponential shift at
    ``mu`` when given; else return ``t`` unchanged."""
    if shift:
        return (t * shift) / (1 + (shift - 1) * t)
    if mu is not None:
        return time_shift(mu, 1.0, t)
    return t


def add_flow_noise(
    latents: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate ``latents`` toward fresh Gaussian noise at ``t`` (shape ``(B,)``).

    Returns ``(noisy_latents, target, t.view(-1, 1))`` with ``target = noise - latents``.
    Works for any latent rank (image ``B,C,H,W`` or video ``B,C,T,H,W``).
    """
    noise = torch.randn_like(latents)
    t_expanded = t.view(-1, *([1] * (latents.ndim - 1)))
    noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
    target = noise - latents
    return noisy_latents, target, t.view(-1, 1)
