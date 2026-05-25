"""Model registry: string type -> factory(config) -> model instance."""

from __future__ import annotations

from typing import Any, Callable

from renga_flow.model.base import ModelPipelineProtocol

# type -> callable(config: dict) -> ModelPipelineProtocol
model_registry: dict[str, Callable[[dict[str, Any]], ModelPipelineProtocol]] = {}


def register_model(name: str) -> Callable[[type], type]:
    """Decorator to register a model class by name. The class must accept config in __init__."""

    def decorator(cls: type) -> type:
        def factory(config: dict[str, Any]) -> ModelPipelineProtocol:
            return cls(config)  # type: ignore[return-value]

        model_registry[name.lower()] = factory
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
