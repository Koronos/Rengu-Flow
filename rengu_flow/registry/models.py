"""Model registry: string type -> factory(config) -> model instance."""

from __future__ import annotations

from typing import Any, Callable

from rengu_flow.model.base import ModelPipelineProtocol
from rengu_flow.registry.model_capabilities import ensure_default_capability

# type -> callable(config: dict) -> ModelPipelineProtocol
model_registry: dict[str, Callable[[dict[str, Any]], ModelPipelineProtocol]] = {}
# Canonical types only (@register_model); aliases via register_model_alias are excluded.
canonical_model_types: set[str] = set()


def register_model(name: str) -> Callable[[type], type]:
    """Decorator to register a model class by name. The class must accept config in __init__."""

    def decorator(cls: type) -> type:
        def factory(config: dict[str, Any]) -> ModelPipelineProtocol:
            return cls(config)  # type: ignore[return-value]

        key = name.lower()
        model_registry[key] = factory
        canonical_model_types.add(key)
        ensure_default_capability(key)
        return cls

    return decorator


def get_model(config: dict[str, Any]) -> ModelPipelineProtocol:
    """Resolve and instantiate model from config['model']['type'] via registry.

    Raises:
        KeyError: If config['model'] or config['model']['type'] is missing.
        ValueError: If the model type is not registered.
    """
    model_config = config.get("model")
    if not model_config:
        raise KeyError("config must contain 'model'")
    model_type = model_config.get("type")
    if model_type is None:
        raise KeyError("config['model'] must contain 'type'")
    key = str(model_type).lower()
    if key not in model_registry:
        raise ValueError(
            f"Unknown model type '{model_type}'. Registered: {sorted(model_registry)}."
        )
    return model_registry[key](config)


def register_model_alias(alias: str, canonical: str) -> None:
    """Register an alias that uses the same factory as an existing model type."""
    key = canonical.lower()
    if key not in model_registry:
        raise KeyError(f"Canonical model '{canonical}' is not registered")
    model_registry[alias.lower()] = model_registry[key]


def _register_builtin_models() -> None:
    """Import pipeline modules so @register_model decorators run (avoids circular import)."""
    import rengu_flow.model.sdxl  # noqa: F401
    import rengu_flow.model.cosmos_predict2  # noqa: F401


_register_builtin_models()
