"""Optimizer registry: string type -> optimizer class. Resolution via registry, aliases, or qualified path."""

from __future__ import annotations

import importlib
from typing import Callable, Type, TypeVar

import torch

T = TypeVar("T", bound=torch.optim.Optimizer)

# name (lowercase) -> optimizer class
optimizer_registry: dict[str, Type[torch.optim.Optimizer]] = {}

# Lazy third-party aliases: name -> (module_path, class_name)
OPTIMIZER_ALIASES: dict[str, tuple[str, str]] = {
    "adamw8bit": ("bitsandbytes.optim", "AdamW8bit"),
    "adamw_optimi": ("optimi", "AdamW"),
    "stableadamw": ("optimi", "StableAdamW"),
    "offload": ("torchao.prototype.low_bit_optim", "CPUOffloadOptimizer"),
    "prodigy": ("pytorch_optimizer", "Prodigy"),
}


def register_optimizer(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register an optimizer class by name (case-insensitive)."""

    def decorator(cls: Type[T]) -> Type[T]:
        optimizer_registry[name.lower()] = cls  # type: ignore[assignment]
        return cls

    return decorator


def _import_class(module_path: str, class_name: str) -> Type[torch.optim.Optimizer]:
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Optimizer alias requires optional dependency (module '{module_path}'). "
            f"Install with: pip install -e '.[optim]'"
        ) from e
    return getattr(module, class_name)


# Lazy vendor imports (optional deps: bitsandbytes, optimum.quanto, deepspeed, transformers)
VENDOR_OPTIMIZER_ALIASES: dict[str, tuple[str, str]] = {
    "genericoptim": (
        "rengu_flow.vendor.diffusion_pipe_optimizers.generic_optim",
        "GenericOptim",
    ),
    "automagic": (
        "rengu_flow.vendor.diffusion_pipe_optimizers.automagic",
        "Automagic",
    ),
    "adamw8bitkahan": (
        "rengu_flow.vendor.diffusion_pipe_optimizers.adamw_8bit",
        "AdamW8bitKahan",
    ),
}


def _resolve_vendor_optimizer(name: str) -> Type[torch.optim.Optimizer]:
    module_path, class_name = VENDOR_OPTIMIZER_ALIASES[name]
    return _import_class(module_path, class_name)

# Built-in torch optimizers
register_optimizer("adamw")(torch.optim.AdamW)
register_optimizer("sgd")(torch.optim.SGD)
register_optimizer("adam")(torch.optim.Adam)


def get_optimizer_class(optim_type: str) -> Type[torch.optim.Optimizer]:
    """Resolve optimizer class from string.

    1. Look up in registry (case-insensitive), including vendored genericoptim/automagic.
    2. Look up in OPTIMIZER_ALIASES (lazy import; optional deps).
    3. If optim_type contains '.', treat as fully-qualified path.
    4. Fallback to pytorch_optimizer library by class name (e.g. Prodigy).
    """
    key = optim_type.lower()
    if key in optimizer_registry:
        return optimizer_registry[key]

    if key in VENDOR_OPTIMIZER_ALIASES:
        return _resolve_vendor_optimizer(key)

    if key in OPTIMIZER_ALIASES:
        module_path, class_name = OPTIMIZER_ALIASES[key]
        return _import_class(module_path, class_name)

    if "." in optim_type:
        module_path, class_name = optim_type.rsplit(".", 1)
        return _import_class(module_path, class_name)

    try:
        import pytorch_optimizer

        return getattr(pytorch_optimizer, optim_type)
    except (ImportError, AttributeError):
        pass

    raise ValueError(
        f"Unknown optimizer type '{optim_type}'. "
        f"Built-in names: {sorted(optimizer_registry)}. "
        f"Vendor names: {sorted(VENDOR_OPTIMIZER_ALIASES)}. "
        f"Optional aliases: {sorted(OPTIMIZER_ALIASES)}. "
        f"Or use a fully-qualified path (e.g. torch.optim.AdamW) or a pytorch_optimizer class name."
    )
