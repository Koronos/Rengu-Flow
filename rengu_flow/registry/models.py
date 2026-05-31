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


# type/alias -> pipeline module, imported lazily in get_model so an SDXL run doesn't pull in the
# Cosmos pipeline (and its cosmos-only deps). Model enumeration/validation come from
# model_capabilities, not these imports, so deferring is safe.
_BUILTIN_MODEL_MODULES: dict[str, str] = {
    "sdxl": "rengu_flow.model.sdxl",
    "cosmos_predict2": "rengu_flow.model.cosmos_predict2",
    "anima": "rengu_flow.model.cosmos_predict2",  # alias registered inside the cosmos module
}


def _ensure_model_imported(key: str) -> None:
    """Import the pipeline module for ``key`` so its @register_model side-effects run."""
    if key in model_registry:
        return
    module = _BUILTIN_MODEL_MODULES.get(key)
    if module is not None:
        import importlib

        importlib.import_module(module)


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
    _ensure_model_imported(key)
    if key not in model_registry:
        known = sorted(set(model_registry) | set(_BUILTIN_MODEL_MODULES))
        raise ValueError(f"Unknown model type '{model_type}'. Known: {known}.")
    return model_registry[key](config)


def register_model_alias(alias: str, canonical: str) -> None:
    """Register an alias that uses the same factory as an existing model type."""
    key = canonical.lower()
    if key not in model_registry:
        raise KeyError(f"Canonical model '{canonical}' is not registered")
    model_registry[alias.lower()] = model_registry[key]
