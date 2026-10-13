"""Config loading, defaults, and validation for Rengu Flow.

Names are resolved lazily (PEP 562): ``import rengu_flow`` reaches ``rengu_flow.config.local_config``
through ``rengu_flow.version``, and an eager import of ``defaults`` here would pull torch into every
torch-free consumer (model specs, the UI schema, ``rengu validate``).
"""

from __future__ import annotations

import importlib

_EXPORTS: dict[str, str] = {
    "load_config": "rengu_flow.config.loader",
    "load_dataset_config": "rengu_flow.config.loader",
    "load_eval_dataset_config": "rengu_flow.config.loader",
    "set_config_defaults": "rengu_flow.config.defaults",
    "validate_config": "rengu_flow.config.validation",
    "load_local_config": "rengu_flow.config.local_config",
    "ensure_local_config_loaded": "rengu_flow.config.local_config",
    "apply_local_config_to_environ": "rengu_flow.config.local_config",
    "load_repo_dotenv": "rengu_flow.config.local_env",
    "apply_model_paths_from_env": "rengu_flow.config.local_env",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
