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


def _model_pipelines() -> dict[str, str]:
    """type/alias -> ``"<module>:<Class>"`` of its pipeline, derived at call time from the
    registered model specs (registration order; an alias maps to its canonical model's pipeline).

    Imported lazily in get_model so an SDXL run doesn't pull in the Cosmos pipeline (and its
    cosmos-only deps); model enumeration/validation come from model_capabilities, not these
    imports. Derived per call (not at import) so a spec registered later resolves too."""
    from rengu_flow.registry.model_capabilities import model_capability_registry

    out: dict[str, str] = {}
    for spec in model_capability_registry.values():
        if not spec.pipeline:
            continue
        for key in (spec.type_id, *spec.aliases):
            out[key.lower()] = spec.pipeline
    return out


def _model_modules() -> dict[str, str]:
    """type/alias -> pipeline module (the alias ``anima`` is registered inside the cosmos module)."""
    return {key: target.partition(":")[0] for key, target in _model_pipelines().items()}


def __getattr__(name: str) -> Any:
    # Back-compat: the former import-time maps stay readable (freshly derived).
    if name == "_BUILTIN_MODEL_PIPELINES":
        return _model_pipelines()
    if name == "_BUILTIN_MODEL_MODULES":
        return _model_modules()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _ensure_model_imported(key: str) -> None:
    """Import the pipeline class for ``key`` so its @register_model side-effects run (model
    packages are lazy: importing the package alone does not import the pipeline)."""
    if key in model_registry:
        return
    target = _model_pipelines().get(key)
    if target is not None:
        import importlib

        module, _, class_name = target.partition(":")
        getattr(importlib.import_module(module), class_name)


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
        known = sorted(set(model_registry) | set(_model_pipelines()))
        raise ValueError(f"Unknown model type '{model_type}'. Known: {known}.")
    return model_registry[key](config)


def register_model_alias(alias: str, canonical: str) -> None:
    """Register an alias that uses the same factory as an existing model type."""
    key = canonical.lower()
    if key not in model_registry:
        raise KeyError(f"Canonical model '{canonical}' is not registered")
    model_registry[alias.lower()] = model_registry[key]
