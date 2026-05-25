"""Registry for pluggable components (models, optimizers, schedulers)."""

from renga_flow.registry.optimizers import (
    get_optimizer_class,
    optimizer_registry,
    register_optimizer,
)

__all__ = [
    "get_model",
    "get_optimizer_class",
    "model_registry",
    "optimizer_registry",
    "register_model",
    "register_optimizer",
]


def __getattr__(name: str):
    """Lazy load models submodule so tests can use optimizer registry without diffusers."""
    if name in ("get_model", "model_registry", "register_model"):
        from renga_flow.registry import models
        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
