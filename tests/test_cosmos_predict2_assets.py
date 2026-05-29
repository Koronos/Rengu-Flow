"""Bundled tokenizer assets are packaged with cosmos_predict2."""

from importlib import resources
from pathlib import Path


def test_qwen3_assets_exist():
    root = resources.files("rengu_flow.model.cosmos_predict2").joinpath("assets", "qwen3_06b")
    assert Path(str(root / "config.json")).is_file()
    assert Path(str(root / "tokenizer.json")).is_file()


def test_t5_assets_exist():
    root = resources.files("rengu_flow.model.cosmos_predict2").joinpath("assets", "t5_old")
    assert Path(str(root / "config.json")).is_file()
    assert Path(str(root / "spiece.model")).is_file()
