"""Shared loss weighting (min-SNR, debiased estimation) for diffusion training."""

from __future__ import annotations

import torch


def _snr_for_timesteps(noise_scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """Index all_snr by the timesteps tensor on the timesteps device.

    Per-element ``int(t)`` indexing forces one GPU->CPU sync per sample every step;
    keeping all_snr on the timesteps device makes the lookup a single gather.
    """
    all_snr = noise_scheduler.all_snr
    if all_snr.device != timesteps.device:
        all_snr = all_snr.to(timesteps.device)
        noise_scheduler.all_snr = all_snr
    return all_snr[timesteps.long()]


def apply_min_snr_weight(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    gamma: float,
    *,
    v_prediction: bool = False,
) -> torch.Tensor:
    """Scale per-sample loss by min-SNR gamma (SimpleTuner / Kohya-style)."""
    snr = _snr_for_timesteps(noise_scheduler, timesteps)
    min_snr_gamma = torch.clamp(snr, max=gamma)
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
    snr_t = _snr_for_timesteps(noise_scheduler, timesteps)
    snr_t = torch.clamp(snr_t, max=1000)
    weight = 1 / (snr_t + 1) if v_prediction else 1 / torch.sqrt(snr_t)
    return loss * weight.to(loss.device)
