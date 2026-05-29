"""Tests for UI signal file helpers and API."""

from pathlib import Path

import pytest

from rengu_flow.utils.signal_files import (
    SIGNAL_CONTINUE,
    SIGNAL_EXPORT_MODEL,
    SIGNAL_EXPORT_MODEL_QUIT,
    SIGNAL_PREVIEW,
    SIGNAL_QUIT,
    SIGNAL_SAVE,
    SIGNAL_SAVE_QUIT,
)
from rengu_flow_ui import signals


@pytest.mark.parametrize(
    "signal_type,filename",
    [
        ("save", SIGNAL_SAVE),
        ("save_quit", SIGNAL_SAVE_QUIT),
        ("export_model", SIGNAL_EXPORT_MODEL),
        ("export_model_quit", SIGNAL_EXPORT_MODEL_QUIT),
        ("preview", SIGNAL_PREVIEW),
        ("continue", SIGNAL_CONTINUE),
        ("quit", SIGNAL_QUIT),
    ],
)
def test_send_signal_creates_file(tmp_path, signal_type, filename):
    path = signals.send_signal(tmp_path, signal_type)
    assert path == str(tmp_path / filename)
    assert (tmp_path / filename).is_file()


def test_send_signal_unknown_type_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown signal"):
        signals.send_signal(tmp_path, "pause")


def test_send_signal_missing_run_dir_raises(tmp_path):
    missing = tmp_path / "no_such_run"
    with pytest.raises(FileNotFoundError, match="Run directory not found"):
        signals.send_signal(missing, "save")


def test_job_signal_api(ui_client, ui_data_tmp: Path) -> None:
    from rengu_flow_ui import db

    run_dir = ui_data_tmp / "runs" / "test_run"
    run_dir.mkdir(parents=True)
    job = db.create_job(
        config_path="configs/x.toml",
        config_id=None,
        log_path=str(ui_data_tmp / "logs" / "job.log"),
        output_dir=str(ui_data_tmp / "output"),
    )
    db.update_job(job.id, state="running", run_dir=str(run_dir))
    r = ui_client.post(f"/api/v1/jobs/{job.id}/signals", json={"type": "save"})
    assert r.status_code == 200
    assert (run_dir / SIGNAL_SAVE).is_file()


def test_job_signal_rejects_stopped_job(ui_client, ui_data_tmp: Path) -> None:
    from rengu_flow_ui import db

    run_dir = ui_data_tmp / "runs" / "stopped_run"
    run_dir.mkdir(parents=True)
    job = db.create_job(
        config_path="configs/x.toml",
        config_id=None,
        log_path=str(ui_data_tmp / "logs" / "job.log"),
        output_dir=str(ui_data_tmp / "output"),
    )
    db.update_job(job.id, state="stopped", run_dir=str(run_dir))
    r = ui_client.post(f"/api/v1/jobs/{job.id}/signals", json={"type": "save"})
    assert r.status_code == 409
    assert not (run_dir / SIGNAL_SAVE).exists()


def test_list_signals_api(ui_client) -> None:
    r = ui_client.get("/api/v1/signals")
    assert r.status_code == 200
    data = r.json()
    ids = {item["id"] for item in data["signals"]}
    assert ids == set(signals.SIGNAL_MAP)
    assert data["active_job_states"] == ["running", "stopping"]


def test_run_dir_accepts_signals_with_active_job(ui_data_tmp: Path) -> None:
    from rengu_flow_ui import db

    run_dir = ui_data_tmp / "active_run"
    run_dir.mkdir()
    job = db.create_job(
        config_path="configs/x.toml",
        config_id=None,
        log_path=str(ui_data_tmp / "logs" / "job.log"),
        output_dir=str(ui_data_tmp / "output"),
    )
    db.update_job(job.id, state="running", run_dir=str(run_dir))
    assert signals.run_dir_accepts_signals(run_dir) is True


def test_run_dir_rejects_signals_without_active_job(tmp_path) -> None:
    run_dir = tmp_path / "finished_run"
    run_dir.mkdir()
    assert signals.run_dir_accepts_signals(run_dir) is False
