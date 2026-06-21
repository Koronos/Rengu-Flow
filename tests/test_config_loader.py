"""Tests for config loader: load_config."""

from pathlib import Path

import pytest
import toml

from rengu_flow.config.loader import load_config


def test_load_config_valid_toml(tmp_path, valid_toml_content):
    path = tmp_path / "config.toml"
    path.write_text(valid_toml_content)
    config = load_config(path)
    assert isinstance(config, dict)
    assert config["dataset"] == "examples/minimal_dataset.toml"
    assert config["model"]["type"] == "sdxl"
    assert config["model"]["dtype"] == "bfloat16"
    assert config["optimizer"]["type"] == "adamw"


def test_load_config_is_pickleable(tmp_path, valid_toml_content):
    path = tmp_path / "config.toml"
    path.write_text(valid_toml_content)
    config = load_config(path)
    assert isinstance(config["model"]["dtype"], str)
    assert config["model"]["dtype"] == "bfloat16"


def test_load_config_missing_file():
    with pytest.raises((FileNotFoundError, OSError)):
        load_config(Path("/nonexistent/config.toml"))


def test_load_config_invalid_toml(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("invalid [toml content =")
    with pytest.raises(toml.TomlDecodeError):
        load_config(path)
