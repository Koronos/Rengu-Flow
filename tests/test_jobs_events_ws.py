"""Server-pushed job-list change events (replaces per-client GET /jobs polling)."""

from __future__ import annotations

from pathlib import Path

from rengu_flow_ui import db


def test_jobs_version_bumps_on_writes(ui_data_tmp: Path) -> None:
    v0 = db.jobs_version()
    job = db.create_job(config_path="", log_path="", kind="prep")
    v1 = db.jobs_version()
    assert v1 > v0  # create bumps

    db.update_job(job.id, state="running")
    v2 = db.jobs_version()
    assert v2 > v1  # update bumps

    db.delete_job(job.id)
    assert db.jobs_version() > v2  # delete bumps


def test_jobs_version_no_bump_on_empty_update(ui_data_tmp: Path) -> None:
    job = db.create_job(config_path="", log_path="", kind="prep")
    before = db.jobs_version()
    db.update_job(job.id)  # no allowed fields → no write
    assert db.jobs_version() == before


def test_jobs_events_ws_pushes_on_change(ui_client, ui_data_tmp: Path) -> None:
    with ui_client.websocket_connect("/api/v1/jobs/events/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "jobs-changed"

        # A new job bumps db.jobs_version() → the socket pushes again without any client poll.
        db.create_job(config_path="", log_path="", kind="prep")
        nxt = ws.receive_json()
        assert nxt["type"] == "jobs-changed"
        assert nxt["version"] > first["version"]
