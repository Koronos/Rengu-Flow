"""Qwen-Image 2.1 (single-stream block-causal DiT + RGBA 16x VAE) modeling code and training
pipeline.

Lazy (PEP 562): importing the package (e.g. for its torch-free ``spec``) does not import the
pipeline; ``from <package> import <Name>`` still works."""

from __future__ import annotations

import importlib

_EXPORTS: dict[str, str] = {
    "AutoencoderKLQwenImage21": "rengu_flow.model.qwen_image21.vae",
    "QwenImage21KVCache": "rengu_flow.model.qwen_image21.dit",
    "QwenImage21Pipeline": "rengu_flow.model.qwen_image21.pipeline",
    "QwenImage21Transformer2DModel": "rengu_flow.model.qwen_image21.dit",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
