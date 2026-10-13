"""Krea 2 (open-weights 12B DiT) training pipeline.

Lazy (PEP 562): importing the package (e.g. for its torch-free ``spec``) does not import the
pipeline; ``from <package> import <Name>`` still works."""

from __future__ import annotations

import importlib

_EXPORTS: dict[str, str] = {
    "Krea2Pipeline": "rengu_flow.model.krea2.pipeline",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
