"""Optimizer step hooks shared across models (fused backward / gradient release)."""

from __future__ import annotations

from typing import Any


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
