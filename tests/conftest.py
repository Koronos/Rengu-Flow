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


def _patch_ui_data_paths(monkeypatch: pytest.MonkeyPatch, base: Path) -> None:
    """Point all renga_flow_ui storage at a temporary directory."""
    monkeypatch.setenv("RENGA_FLOW_UI_DATA", str(base))
    monkeypatch.setattr("renga_flow_ui.settings.ui_data_dir", lambda: base)
    monkeypatch.setattr("renga_flow_ui.settings.staging_dir", lambda: base / "staging")
    monkeypatch.setattr("renga_flow_ui.settings.logs_dir", lambda: base / "logs")
    monkeypatch.setattr("renga_flow_ui.settings.db_path", lambda: base / "jobs.db")
    monkeypatch.setattr("renga_flow_ui.configs_store.staging_dir", lambda: base / "staging")


@pytest.fixture
def ui_data_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolated UI data dir with staging/, logs/, jobs.db (configs/datasets in SQLite)."""
    from renga_flow_ui import db

    base = tmp_path / "ui"
    for sub in ("staging", "logs"):
        (base / sub).mkdir(parents=True)
    _patch_ui_data_paths(monkeypatch, base)
    db.init_db()
    return base


@pytest.fixture
def ui_client(ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    """FastAPI TestClient with UI routes (no auth token)."""
    from starlette.testclient import TestClient

    from renga_flow_ui.app import create_app

    monkeypatch.delenv("RENGA_FLOW_UI_TOKEN", raising=False)
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def ui_client_auth(ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    from starlette.testclient import TestClient

    from renga_flow_ui.app import create_app

    monkeypatch.setenv("RENGA_FLOW_UI_TOKEN", "test-secret")
    headers = {"X-Renga-Flow-Token": "test-secret"}
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, headers
