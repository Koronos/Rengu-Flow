"""Pytest fixtures for rengu-flow tests."""

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
    """Point all rengu_flow_ui storage at a temporary directory."""
    db_file = base / "jobs.db"

    def _db_path() -> Path:
        return db_file

    monkeypatch.setenv("RENGU_FLOW_UI_DATA", str(base))
    monkeypatch.setattr("rengu_flow_ui.settings.ui_data_dir", lambda: base)
    monkeypatch.setattr("rengu_flow_ui.settings.staging_dir", lambda: base / "staging")
    monkeypatch.setattr("rengu_flow_ui.settings.logs_dir", lambda: base / "logs")
    monkeypatch.setattr("rengu_flow_ui.settings.db_path", _db_path)
    monkeypatch.setattr("rengu_flow_ui.db.db_path", _db_path)
    monkeypatch.setattr("rengu_flow_ui.library_db.db_path", _db_path)
    monkeypatch.setattr("rengu_flow_ui.run_staging.staging_dir", lambda: base / "staging")


def _init_ui_data_dir(base: Path) -> None:
    for sub in ("staging", "logs"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    from rengu_flow_ui import db

    db.init_db()


@pytest.fixture
def ui_data_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolated UI data dir with staging/, logs/, jobs.db (configs/datasets in SQLite)."""
    base = tmp_path / "ui"
    _patch_ui_data_paths(monkeypatch, base)
    _init_ui_data_dir(base)
    return base


@pytest.fixture(autouse=True)
def _isolated_ui_sqlite(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Per-test temp jobs.db unless a test already uses ui_data_tmp or ui_client.

    Uses a dedicated temp dir (not the test's ``tmp_path``) so it never pollutes tests that
    enumerate their own ``tmp_path`` (e.g. checkpoint/export retention).
    """
    if request.node.get_closest_marker("no_ui_db"):
        return
    if "ui_data_tmp" in request.fixturenames:
        return
    if (
        "ui_client" in request.fixturenames
        or "ui_client_auth" in request.fixturenames
    ):
        return
    base = tmp_path_factory.mktemp("ui_auto")
    _patch_ui_data_paths(monkeypatch, base)
    _init_ui_data_dir(base)


@pytest.fixture(autouse=True)
def _isolate_cache_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep dataset caches out of the real <repo>/cache during tests.

    Datasets resolve their cache under ``default_cache_root()`` (= <repo>/cache) when no
    cache_root is configured; redirect that to a temp dir so tests stay hermetic. The dir
    is still named ``cache`` so ``test_cache_paths`` default-name assertions hold.
    """
    base = tmp_path_factory.mktemp("cache_root") / "cache"
    monkeypatch.setattr("rengu_flow.data.cache_paths.default_cache_root", lambda: base)


@pytest.fixture
def ui_client(ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    """FastAPI TestClient with UI routes (no auth token)."""
    pytest.importorskip("starlette", reason="ui extra not installed (uv sync --extra ui)")
    from starlette.testclient import TestClient

    from rengu_flow_ui.app import create_app

    monkeypatch.delenv("RENGU_FLOW_UI_TOKEN", raising=False)
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def ui_client_auth(ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("starlette", reason="ui extra not installed (uv sync --extra ui)")
    from starlette.testclient import TestClient

    from rengu_flow_ui.app import create_app

    monkeypatch.setenv("RENGU_FLOW_UI_TOKEN", "test-secret")
    headers = {"X-Rengu-Flow-Token": "test-secret"}
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, headers
