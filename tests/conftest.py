"""Pytest fixtures for renga-flow tests."""

import copy
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def examples_dir() -> Path:
    """Path to the examples/ directory at repo root."""
    return _repo_root() / "examples"


@pytest.fixture
def minimal_config() -> dict:
    """Minimal valid config dict (model.type, model.dtype, optimizer.type, dataset)."""
    return {
        "dataset": "examples/minimal_dataset.toml",
        "model": {"type": "sdxl", "dtype": "bfloat16", "checkpoint_path": "path/to/sdxl.safetensors"},
        "optimizer": {"type": "adamw", "lr": 1.0e-4},
    }


@pytest.fixture
def minimal_config_copy(minimal_config) -> dict:
    """Copy of minimal_config for mutating in tests (e.g. set_config_defaults)."""
    return copy.deepcopy(minimal_config)


@pytest.fixture
def valid_toml_content() -> str:
    """Valid TOML string for temporary config files."""
    return """
dataset = "examples/minimal_dataset.toml"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "path/to/sdxl.safetensors"

[optimizer]
type = "adamw"
lr = 1.0e-4
"""
