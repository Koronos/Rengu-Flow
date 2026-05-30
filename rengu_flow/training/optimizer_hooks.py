"""Optimizer step hooks shared across models (fused backward / gradient release)."""

from __future__ import annotations

from typing import Any

import torch


def validate_fused_optimizer_config(config: dict[str, Any]) -> None:
    """Raise if fused optimizer options conflict with gradient accumulation."""
    optim = config.get("optimizer") or {}
    fused = optim.get("fused_backward") or optim.get("fused_optimizer_groups")
    if not fused:
        return
    if int(config.get("gradient_accumulation_steps", 1)) > 1:
        raise ValueError(
            "optimizer.fused_backward and optimizer.fused_optimizer_groups "
            "require gradient_accumulation_steps = 1."
        )


def partition_parameters_for_fused_groups(
    parameters: list[torch.nn.Parameter],
    num_groups: int,
) -> list[list[torch.nn.Parameter]]:
    """Split parameters into N contiguous groups for per-group fused optimizers."""
    n = max(1, int(num_groups))
    if n == 1:
        return [parameters]
    chunk = max(1, (len(parameters) + n - 1) // n)
    return [parameters[i : i + chunk] for i in range(0, len(parameters), chunk)]
