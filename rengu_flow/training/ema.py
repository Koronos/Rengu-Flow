"""Exponential moving average of trainable weights (optional, CPU-backed)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TrainingEMA:
    """Track EMA copies of parameters on CPU to reduce VRAM."""

    def __init__(
        self, parameters: list[nn.Parameter], decay: float, update_interval: int = 1
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        if update_interval < 1:
            raise ValueError(f"ema_update_interval must be >= 1, got {update_interval}")
        # Updating every N steps with decay^N keeps the same smoothing horizon as a
        # per-step update (the decay products match; only intra-window drift differs,
        # negligible for N << 1/(1-decay)). This matters because each update round-trips
        # every trainable param over PCIe to the fp32 CPU shadow — measured ~0.4-0.7s per
        # step on a 348M-param full finetune, i.e. it can silently dominate the step time
        # (it runs outside the timed train_batch window).
        self.decay = decay ** update_interval
        self.update_interval = update_interval
        self.shadow: dict[int, torch.Tensor] = {}
        for p in parameters:
            if p.requires_grad:
                self.shadow[id(p)] = p.detach().float().cpu().clone()

    @classmethod
    def from_config(cls, config: dict[str, Any], parameters: list[nn.Parameter]) -> TrainingEMA | None:
        decay = config.get("ema_decay")
        if decay is None:
            return None
        return cls(
            parameters,
            float(decay),
            update_interval=int(config.get("ema_update_interval", 1)),
        )

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
