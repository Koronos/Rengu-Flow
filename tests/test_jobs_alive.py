"""Reconcile stale running jobs against live PIDs."""

from __future__ import annotations

import pytest

from rengu_flow_ui import db, job_queue, jobs


def _pending_job(ui_data_tmp) -> db.JobRecord:
    return db.create_job(
        config_path="configs/x.toml",
        log_path=str(ui_data_tmp / "logs" / "job.log"),
        output_dir=str(ui_data_tmp / "output"),
    )


def test_poll_job_finishes_running_without_pid(ui_data_tmp) -> None:
    job = _pending_job(ui_data_tmp)
    db.update_job(job.id, state="running", pid=None)

    reconciled = jobs.poll_job(job.id)

    assert reconciled.state == "finished"
    assert reconciled.pid is None
    assert reconciled.finished_at is not None


def test_poll_job_stops_stopping_without_pid(ui_data_tmp) -> None:
    job = _pending_job(ui_data_tmp)
    db.update_job(job.id, state="stopping", pid=None)

    reconciled = jobs.poll_job(job.id)

    assert reconciled.state == "stopped"
    assert reconciled.pid is None


def test_poll_job_finished_advances_queue(ui_data_tmp, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: calls.append(1))
    job = _pending_job(ui_data_tmp)
    db.update_job(job.id, state="running", pid=None)

    reconciled = jobs.poll_job(job.id)

    assert reconciled.state == "finished"
    assert calls == [1]  # a natural end advances the queue


def test_poll_job_stopped_does_not_advance_queue(
    ui_data_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: calls.append(1))
    job = _pending_job(ui_data_tmp)
    db.update_job(job.id, state="stopping", pid=None)  # user stop/quit

    reconciled = jobs.poll_job(job.id)

    assert reconciled.state == "stopped"
    assert calls == []  # a user stop/quit halts the queue


def test_poll_job_finishes_running_with_dead_pid(ui_data_tmp) -> None:
    job = _pending_job(ui_data_tmp)
    db.update_job(job.id, state="running", pid=999_999_999)

    reconciled = jobs.poll_job(job.id)

    assert reconciled.state == "finished"
    assert reconciled.pid is None


def test_list_jobs_sorted_reconciles_stale_running(
    ui_data_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    job = _pending_job(ui_data_tmp)
    db.update_job(job.id, state="running", pid=None)

    rows = job_queue.list_jobs_sorted()
    by_id = {j.id: j for j in rows}

    assert by_id[job.id].state == "finished"


def test_jobs_api_list_reconciles_stale_running(ui_client, ui_data_tmp) -> None:
    job = _pending_job(ui_data_tmp)
    db.update_job(job.id, state="running", pid=None)

    r = ui_client.get("/api/v1/jobs")
    assert r.status_code == 200
    jobs_payload = {int(j["id"]): j for j in r.json()["jobs"]}

    assert jobs_payload[job.id]["state"] == "finished"
    assert r.json()["stats"]["running"] == 0


def test_train_runs_reconciles_stale_running(ui_client, ui_data_tmp) -> None:
    job = _pending_job(ui_data_tmp)
    db.update_job(job.id, state="running", pid=None)

    r = ui_client.get("/api/v1/train/runs?page=1&page_size=50")
    assert r.status_code == 200
    items = {row["job_id"]: row for row in r.json()["items"] if row.get("job_id") is not None}

    assert items[job.id]["state"] == "finished"
    assert r.json()["stats"]["running"] == 0
