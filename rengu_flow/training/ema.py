"""Exponential moving average of trainable weights (optional, CPU-backed)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TrainingEMA:
    """Track EMA copies of parameters on CPU to reduce VRAM."""

    def __init__(self, parameters: list[nn.Parameter], decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[int, torch.Tensor] = {}
        for p in parameters:
            if p.requires_grad:
                self.shadow[id(p)] = p.detach().float().cpu().clone()

    @classmethod
    def from_config(cls, config: dict[str, Any], parameters: list[nn.Parameter]) -> TrainingEMA | None:
        decay = config.get("ema_decay")
        if decay is None:
            return None
        return cls(parameters, float(decay))

    def update(self, parameters: list[nn.Parameter]) -> None:
        d = self.decay
        for p in parameters:
            if not p.requires_grad:
                continue
            key = id(p)
            if key not in self.shadow:
                self.shadow[key] = p.detach().float().cpu().clone()
                continue
            self.shadow[key].mul_(d).add_(p.detach().float().cpu(), alpha=1.0 - d)
