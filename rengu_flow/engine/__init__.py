"""Engine backend selection + back-compat shims."""
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


# Back-compat free functions (delegate to the selected backend); removed in Task 6 once call sites migrate.
def build_pipe(backend: str, *, layers, num_stages, partition_method, manual_partition_split, loss_fn, extra_kw):
    return _BACKENDS[backend]({}).build_pipe(
        layers=layers, num_stages=num_stages, partition_method=partition_method,
        manual_partition_split=manual_partition_split, loss_fn=loss_fn, extra_kw=extra_kw,
    )


def build_engine(backend: str, *, pipeline_model, ds_config, args, get_optimizer, parameters_to_train):
    return _BACKENDS[backend]({}).build_engine(
        pipeline_model=pipeline_model, ds_config=ds_config, args=args,
        get_optimizer=get_optimizer, parameters_to_train=parameters_to_train,
    )


__all__ = ["Engine", "TrainingBackend", "SingleDeviceBackend", "DeepSpeedPipeBackend",
           "select_backend", "resolve_backend", "build_pipe", "build_engine",
           "TorchEngine", "SequentialPipe"]
