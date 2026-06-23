"""Engine backend selection."""
from __future__ import annotations

import os

from rengu_flow.engine.base import Engine, TrainingBackend
from rengu_flow.engine.single_device import SequentialPipe, SingleDeviceBackend, TorchEngine
from rengu_flow.engine.deepspeed_pipe import DeepSpeedPipeBackend

_BACKENDS = {b.name: b for b in (SingleDeviceBackend, DeepSpeedPipeBackend)}


def resolve_backend(config: dict | None = None) -> str:
    from rengu_flow.platform_compat import PLATFORM
    name = (os.environ.get("RENGU_ENGINE") or (config or {}).get("engine") or "").strip().lower()
    return name or PLATFORM.default_engine


def select_backend(config: dict | None = None) -> TrainingBackend:
    name = resolve_backend(config)
    try:
        return _BACKENDS[name](config)
    except KeyError:
        raise SystemExit(f"unknown engine {name!r} (accelerate|deepspeed)")


__all__ = ["Engine", "TrainingBackend", "SingleDeviceBackend", "DeepSpeedPipeBackend",
           "select_backend", "resolve_backend", "TorchEngine", "SequentialPipe"]
