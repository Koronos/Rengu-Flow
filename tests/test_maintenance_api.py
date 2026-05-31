"""Maintenance API: DB reset, submodule subprocess (mocked)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rengu_flow_ui import datasets_store, db, library_db


@pytest.fixture
def maintenance_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENGUFLOW_MAINTENANCE", "1")


@pytest.fixture
def maintenance_client(ui_data_tmp: Path, maintenance_on: None):
    """TestClient with maintenance API enabled (env set before create_app)."""
    from starlette.testclient import TestClient

    from rengu_flow_ui.app import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_maintenance_disabled_by_default(ui_client) -> None:
    r = ui_client.get("/api/v1/maintenance/enabled")
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    r2 = ui_client.get("/api/v1/maintenance/status")
    assert r2.status_code == 403


def test_maintenance_status_when_enabled(maintenance_client) -> None:
    r = maintenance_client.get("/api/v1/maintenance/status")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert "database" in body
    assert "jobs" in body["database"]["tables"]
    assert body["pyproject_exists"] is True


def test_database_reset_recreates_tables(maintenance_client, ui_data_tmp) -> None:
    did = datasets_store.insert_dataset("resolutions = [512]\nframe_buckets = [1]\n")
    job = db.create_job(
        config_path="",
        log_path="x",
        config_content='dataset = "x"\n[model]\ntype = "sdxl"\n',
    )
    assert library_db.dataset_exists(did)

    r = maintenance_client.post("/api/v1/maintenance/database/reset", json={"confirmation": "RESET"})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "datasets" in data["tables_after"]
    assert "jobs" in data["tables_after"]

    assert not library_db.dataset_exists(did)
    with pytest.raises(KeyError):
        db.get_job(job.id)


def test_database_reset_requires_confirm(ui_client, maintenance_on) -> None:
    r = ui_client.post("/api/v1/maintenance/database/reset")
    assert r.status_code == 400


def test_submodules_update_mocked(maintenance_client) -> None:
    fake = {
        "ok": True,
        "returncode": 0,
        "stdout": "Submodule path 'foo': checked out\n",
        "stderr": "",
        "command": ["git", "submodule", "update", "--init", "--recursive"],
        "message": "Submodule update finished.",
    }
    with patch("rengu_flow_ui.maintenance.submodule_update", return_value=fake):
        r = maintenance_client.post("/api/v1/maintenance/submodules/update")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "foo" in r.json()["stdout"]


def test_deps_install_dry_run(maintenance_client) -> None:
    r = maintenance_client.post(
        "/api/v1/maintenance/deps/install",
        json={"profile": "cosmos_predict2", "execute": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["executed"] is False
    assert "cosmos_predict2" in body["command"]
