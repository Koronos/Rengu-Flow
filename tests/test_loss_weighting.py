"""Vectorized SNR loss weighting matches the per-element reference implementation."""

from __future__ import annotations

import torch

from rengu_flow.training.loss_weighting import (
    apply_debiased_estimation,
    apply_min_snr_weight,
)


class FakeScheduler:
    def __init__(self, num_timesteps: int = 1000) -> None:
        alphas_cumprod = torch.linspace(0.9999, 0.0001, num_timesteps)
        alpha = torch.sqrt(alphas_cumprod)
        sigma = torch.sqrt(1.0 - alphas_cumprod)
        self.all_snr = (alpha / sigma) ** 2


def _reference_min_snr(loss, timesteps, scheduler, gamma, v_prediction):
    snr = torch.stack([scheduler.all_snr[int(t)] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float()
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float()
    return loss * snr_weight


def _reference_debiased(loss, timesteps, scheduler, v_prediction):
    snr_t = torch.stack([scheduler.all_snr[int(t)] for t in timesteps])
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)
    weight = 1 / (snr_t + 1) if v_prediction else 1 / torch.sqrt(snr_t)
    return loss * weight


def test_min_snr_matches_reference():
    scheduler = FakeScheduler()
    timesteps = torch.tensor([0, 1, 17, 500, 998, 999])
    loss = torch.rand(len(timesteps))
    for v_prediction in (False, True):
        got = apply_min_snr_weight(
            loss, timesteps, scheduler, 5.0, v_prediction=v_prediction
        )
        want = _reference_min_snr(loss, timesteps, scheduler, 5.0, v_prediction)
        torch.testing.assert_close(got, want)


def test_debiased_estimation_matches_reference():
    scheduler = FakeScheduler()
    timesteps = torch.tensor([0, 3, 250, 999])
    loss = torch.rand(len(timesteps))
    for v_prediction in (False, True):
        got = apply_debiased_estimation(
            loss, timesteps, scheduler, v_prediction=v_prediction
        )
        want = _reference_debiased(loss, timesteps, scheduler, v_prediction)
        torch.testing.assert_close(got, want)
