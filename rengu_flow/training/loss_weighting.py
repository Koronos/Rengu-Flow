"""Shared loss weighting (min-SNR, debiased estimation) for diffusion training."""

from __future__ import annotations

import torch


def apply_min_snr_weight(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    gamma: float,
    *,
    v_prediction: bool = False,
) -> torch.Tensor:
    """Scale per-sample loss by min-SNR gamma (SimpleTuner / Kohya-style)."""
    snr = torch.stack([noise_scheduler.all_snr[int(t)] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    return loss * snr_weight


def apply_debiased_estimation(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    *,
    v_prediction: bool = False,
) -> torch.Tensor:
    snr_t = torch.stack([noise_scheduler.all_snr[int(t)] for t in timesteps])
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)
    weight = 1 / (snr_t + 1) if v_prediction else 1 / torch.sqrt(snr_t)
    return loss * weight.to(loss.device)
