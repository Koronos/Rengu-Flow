"""Tests for UI config store and validation helpers."""

from pathlib import Path

import pytest

from rengu_flow_ui import configs_store


def test_config_crud(ui_data_tmp: Path) -> None:
    cid = configs_store.insert_config('dataset = "x.toml"\n[model]\ntype="sdxl"\n')
    assert cid in configs_store.list_config_ids()
    text = configs_store.read_config_text(cid)
    assert "sdxl" in text
    dup = configs_store.duplicate_config(cid)
    assert dup != cid
    configs_store.delete_config(cid)
    with pytest.raises(FileNotFoundError):
        configs_store.read_config_text(cid)


def test_validate_minimal_toml() -> None:
    bad = "not toml"
    r = configs_store.validate_toml_text(bad)
    assert r["ok"] is False
    minimal = """
dataset = "examples/minimal_dataset.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"
[optimizer]
type = "adamw"
"""
    r2 = configs_store.validate_toml_text(minimal)
    assert r2["ok"] is True
