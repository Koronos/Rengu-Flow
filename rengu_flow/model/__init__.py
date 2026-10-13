"""Model contract and base for Rengu Flow pipelines.

Everything is lazy (PEP 562): model spec modules live inside this package
(``rengu_flow/model/<m>/spec.py``, ``sdxl_spec.py``) and must import without torch, so neither the
base contract nor any pipeline is imported up front. Pipeline classes (``Krea2Pipeline``, ...) are
derived from the built-in specs' ``pipeline`` (``"<module>:<Class>"``).
"""

from __future__ import annotations

import importlib

_BASE_EXPORTS: dict[str, str] = {
    "BasePipeline": "rengu_flow.model.base",
    "ModelPipelineProtocol": "rengu_flow.model.base",
    "make_contiguous": "rengu_flow.model.base",
}


def _pipeline_exports() -> dict[str, str]:
    """Pipeline class name -> module, from the built-in model specs (registration order)."""
    from rengu_flow.registry.builtin_models import builtin_specs

    return {spec.pipeline_class_name: spec.pipeline_module for spec in builtin_specs() if spec.pipeline}


def __getattr__(name: str):
    if name == "__all__":
        return [*_BASE_EXPORTS, *_pipeline_exports()]
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _BASE_EXPORTS.get(name) or _pipeline_exports().get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
