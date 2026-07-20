"""Exponential moving average of trainable weights (optional, CPU-backed).

Enabled by ``ema_decay`` in config. The shadow lives in fp32 on CPU (VRAM-free); it is
updated every optimizer step, persisted inside the resume checkpoint, and swapped into the
model at export so the exported weights are the smoothed average, not the last noisy step.

Not applied at preview time: ``run_previews`` toggles ``optimizer.eval()`` internally
(lookahead optimizers), which would overwrite weights swapped in from outside. The export
path does not toggle the optimizer, so ``average_parameters`` is correct there.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
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
        self._backup: list[tuple[nn.Parameter, torch.Tensor]] | None = None

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

    def copy_to(self, parameters: list[nn.Parameter]) -> None:
        """Write the EMA shadow into the live parameters (in place)."""
        for p in parameters:
            saved = self.shadow.get(id(p))
            if saved is not None:
                p.data.copy_(saved.to(dtype=p.dtype, device=p.device))

    @contextmanager
    def average_parameters(self, parameters: list[nn.Parameter]):
        """Temporarily swap EMA weights into the model, then restore the live weights.

        Backup is held on CPU to avoid a VRAM spike; restore always runs.
        """
        tracked = [p for p in parameters if id(p) in self.shadow]
        self._backup = [(p, p.detach().cpu().clone()) for p in tracked]
        self.copy_to(tracked)
        try:
            yield
        finally:
            for p, live in self._backup:
                p.data.copy_(live.to(dtype=p.dtype, device=p.device))
            self._backup = None

    def state_dict(self, parameters: list[nn.Parameter]) -> dict[str, Any]:
        """Serialize the shadow as an ordered tensor list (order = ``parameters`` order)."""
        return {
            "decay": self.decay,
            "shadow": [self.shadow[id(p)] for p in parameters if id(p) in self.shadow],
        }

    def load_state_dict(self, state: dict[str, Any], parameters: list[nn.Parameter]) -> None:
        tracked = [p for p in parameters if id(p) in self.shadow]
        shadow_list = state["shadow"]
        if len(shadow_list) != len(tracked):
            raise ValueError(
                f"EMA checkpoint has {len(shadow_list)} tensors but model has {len(tracked)} "
                "trainable params — parameter set changed since the checkpoint was written"
            )
        self.decay = float(state.get("decay", self.decay))
        for p, saved in zip(tracked, shadow_list):
            self.shadow[id(p)].copy_(saved.float().cpu())


def save_ema_checkpoint(save_root, ema: TrainingEMA | None, parameters: list[nn.Parameter]) -> None:
    """Write ``ema.pt`` next to the DeepSpeed/accelerate checkpoint just saved (rank-0 only)."""
    if ema is None:
        return
    from rengu_flow.utils.common import is_main_process

    if not is_main_process():
        return
    root = Path(save_root)
    latest = root / "latest"
    if not latest.is_file():
        return
    tag = latest.read_text().strip()
    torch.save(ema.state_dict(parameters), root / tag / "ema.pt")


def load_ema_checkpoint(load_path, ema: TrainingEMA | None, parameters: list[nn.Parameter]) -> None:
    """Restore the EMA shadow from a resumed checkpoint. No-op if EMA off or file absent.

    ``load_path`` is what the engine's ``load_checkpoint`` returns: a file for the accelerate
    engine (``.../torch_engine.pt``), a directory for DeepSpeed. Handle both.
    """
    if ema is None or not load_path:
        return
    p = Path(load_path)
    ema_path = (p if p.is_dir() else p.parent) / "ema.pt"
    if not ema_path.is_file():
        print("rengu_flow: no ema.pt in checkpoint — EMA shadow starts from current weights", flush=True)
        return
    ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=False), parameters)
    print(f"rengu_flow: restored EMA shadow from {ema_path}", flush=True)
