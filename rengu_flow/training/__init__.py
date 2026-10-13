"""Shared training techniques (VRAM, speed, quality) used across pipeline models.

Names are exported lazily (PEP 562) so torch-free submodules such as ``token_routing.parse_tread``
(used by config validation) can be imported without pulling torch in through this package.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "BlockSwapOffloader": "rengu_flow.training.block_swap",
    "HookBlockSwapOffloader": "rengu_flow.training.block_swap",
    "NoopOffloader": "rengu_flow.training.block_swap",
    "patch_deepspeed_for_block_swap": "rengu_flow.training.block_swap",
    "TrainingEMA": "rengu_flow.training.ema",
    "apply_debiased_estimation": "rengu_flow.training.loss_weighting",
    "apply_min_snr_weight": "rengu_flow.training.loss_weighting",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
