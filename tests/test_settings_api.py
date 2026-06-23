"""Settings API: GET/PUT /api/v1/settings."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def cfg_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "rengu.local.toml"
    p.write_text(
        "[ui]\nhost = \"127.0.0.1\"\nport = 8765\npublic = false\ndata_dir = \"data\"\n\n"
        "[maintenance]\nenabled = false\nallow_pip = false\n\n"
        "[training]\nnum_gpus = 1\nmaster_port = 29500\nextra_args = \"\"\n\n"
        "[training.env]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("rengu_flow_ui.settings_store.config_path", lambda: p)
    return p


def test_get_settings_shape(ui_client, cfg_file: Path) -> None:
    r = ui_client.get("/api/v1/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["editable"]["training"]["num_gpus"] == 1
    assert body["readOnly"]["ui"]["port"] == 8765
    assert body["restartRequired"]["ui"]["public"] is False


def test_put_settings_writes_training(ui_client, cfg_file: Path) -> None:
    r = ui_client.put("/api/v1/settings", json={"training": {"num_gpus": 2, "extra_args": "--x"}})
    assert r.status_code == 200
    assert r.json()["editable"]["training"]["num_gpus"] == 2
    assert "num_gpus = 2" in cfg_file.read_text(encoding="utf-8")


def test_maintenance_is_force_disabled_and_not_editable(ui_client, cfg_file: Path) -> None:
    # Maintenance is no longer an editable settings section: the schema rejects it.
    r = ui_client.put("/api/v1/settings", json={"maintenance": {"enabled": True}})
    assert r.status_code == 422


def test_put_settings_invalid_returns_422(ui_client, cfg_file: Path) -> None:
    r = ui_client.put("/api/v1/settings", json={"training": {"num_gpus": 0}})
    assert r.status_code == 422


def test_put_settings_rejects_non_editable(ui_client, cfg_file: Path) -> None:
    r = ui_client.put("/api/v1/settings", json={"ui": {"host": "0.0.0.0"}})
    assert r.status_code == 422
