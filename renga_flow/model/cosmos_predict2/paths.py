"""Package asset paths (tokenizer configs bundled with renga-flow)."""

from __future__ import annotations

from importlib import resources


def assets_dir() -> str:
    return str(resources.files("renga_flow.model.cosmos_predict2").joinpath("assets"))


def qwen3_config_dir() -> str:
    return str(resources.files("renga_flow.model.cosmos_predict2").joinpath("assets", "qwen3_06b"))


def t5_config_dir() -> str:
    return str(resources.files("renga_flow.model.cosmos_predict2").joinpath("assets", "t5_old"))
