"""Registry for pluggable components (models, optimizers, schedulers).

Every name is resolved lazily (PEP 562): ``rengu_flow.registry.model_capabilities`` (and the
model specs it registers) must stay importable without torch, and the optimizer / model
registries pull torch (and diffusers) on import.
"""

__all__ = [
    "get_model",
    "get_optimizer_class",
    "model_registry",
    "optimizer_registry",
    "register_model",
    "register_optimizer",
]

_OPTIMIZER_NAMES = ("get_optimizer_class", "optimizer_registry", "register_optimizer")
_MODEL_NAMES = ("get_model", "model_registry", "register_model")


def __getattr__(name: str):
    if name in _OPTIMIZER_NAMES:
        from rengu_flow.registry import optimizers

        return getattr(optimizers, name)
    if name in _MODEL_NAMES:
        from rengu_flow.registry import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
