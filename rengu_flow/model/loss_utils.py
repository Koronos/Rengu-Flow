"""Per-element diffusion loss helpers (testable; aligned with diffusion-pipe)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_diffusion_loss_per_element(
    output: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    """Element-wise loss before masking and batch reductions."""
    if "huber_delta" in config:
        return F.huber_loss(output, target, reduction="none", delta=config["huber_delta"])
    if "smooth_l1_beta" in config:
        return F.smooth_l1_loss(
            output, target, reduction="none", beta=config["smooth_l1_beta"]
        )
    if "pseudo_huber_c" in config:
        c = config["pseudo_huber_c"]
        return torch.sqrt((output - target) ** 2 + c**2) - c
    return F.mse_loss(output, target, reduction="none")
