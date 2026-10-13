"""The built-in model spec modules, in registration order.

Each module defines ``SPEC = ModelCapability(...)`` (torch-free). Tuple order = registry order =
schema field order. Adding a model = its package + ``spec.py`` + one line here (explicit
registration, no filesystem discovery).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rengu_flow.registry.model_spec import ModelCapability

BUILTIN_MODEL_SPECS: tuple[str, ...] = (
    "rengu_flow.model.sdxl_spec",
    "rengu_flow.model.cosmos_predict2.spec",
    "rengu_flow.model.krea2.spec",
    "rengu_flow.model.qwen_image21.spec",
    "rengu_flow.model.nanosaur2.spec",
)


def builtin_specs() -> tuple[ModelCapability, ...]:
    """The ``SPEC`` of every built-in model, in registration order (imports the spec modules
    only — never the pipelines)."""
    return tuple(importlib.import_module(module).SPEC for module in BUILTIN_MODEL_SPECS)
