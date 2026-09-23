"""Prep "Start now" takes the GPU lease like every other launcher (spec Risk 14).

``enqueue_prep_job(start_now=True)`` and ``requeue_prep_job(start_now=True)`` used to call
``jobs.start_job`` straight after a ``has_active_runner()`` check — no lease at all. A training run
is still ``pending`` for the minutes its own ``uv sync`` takes, and a workflow GPU node holds the
lease with no job row at all, so in both cases "Start now" walked onto a GPU somebody else held.
"""

import os
from pathlib import Path

import pytest

from rengu_flow_ui import db, gpu_lease, job_queue, jobs, prep_jobs, workflow_db

_TOML = "path = 'x'\n"


@pytest.fixture(autouse=True)
def _single_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_lease, "enumerate_devices", lambda: [0])


def _hold_for_workflow() -> str:
    """A live workflow node's lease. The row must exist, or ``reap_dead`` frees it on sight."""
    workflow = workflow_db.create_workflow("wf", "{}")
    holder = f"wf:{workflow.id}:n2"
    assert gpu_lease.acquire("workflow", holder, None)
    return holder


def _fake_start(pid: int):
    started: list[int] = []

    def start(job: db.JobRecord, **_kwargs: object) -> int:
        started.append(job.id)
        db.update_job(job.id, state="running", pid=pid)
        return pid

    return start, started


def test_start_now_takes_and_binds_the_lease(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)

    job = prep_jobs.enqueue_prep_job("tag", _TOML, start_now=True)

    assert started == [job.id]
    rows = gpu_lease.snapshot()
    assert [(r["holder_kind"], r["holder_id"]) for r in rows] == [("train", f"job:{job.id}")]
    assert rows[0]["pid"] == os.getpid()  # bound, so a reap can judge the holder


def test_start_now_refuses_a_gpu_a_workflow_node_holds(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case ``has_active_runner()`` cannot see at all: a workflow node has no job row."""
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)
    holder = _hold_for_workflow()

    with pytest.raises(gpu_lease.LeaseBusyError, match=holder):
        prep_jobs.enqueue_prep_job("tag", _TOML, start_now=True)

    assert started == []
    # Nothing is left behind: a refused "Start now" is not a silent enqueue that nothing drains.
    assert [j for j in db.list_jobs(limit=50) if j.kind == "prep"] == []
    assert [r["holder_id"] for r in gpu_lease.snapshot()] == [holder]


def test_start_now_refuses_while_a_training_run_is_mid_launch(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The training row reads ``pending`` for the whole ``uv sync``; only the lease says busy."""
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)
    train = db.create_job(config_path="/tmp/x.toml", log_path="/tmp/x.log", state="pending")
    assert gpu_lease.acquire("train", f"job:{train.id}", None)
    assert not job_queue.has_active_runner()  # the blind spot the old check had

    with pytest.raises(gpu_lease.LeaseBusyError):
        prep_jobs.enqueue_prep_job("tag", _TOML, start_now=True)

    assert started == []


def test_start_job_raising_system_exit_releases_the_prep_lease(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BaseException, not Exception: an immortal ``pid IS NULL`` lease kills both lanes."""

    def boom(job: db.JobRecord, **_kwargs: object) -> int:
        raise SystemExit("uv is not installed")

    monkeypatch.setattr(jobs, "start_job", boom)

    with pytest.raises(SystemExit):
        prep_jobs.enqueue_prep_job("tag", _TOML, start_now=True)

    assert gpu_lease.snapshot() == []


def test_requeue_start_now_with_the_gpu_held_restores_the_row(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)
    job = prep_jobs.enqueue_prep_job("tag", _TOML)
    db.update_job(job.id, state="failed", exit_code=1, finished_at="2026-01-01", queue_position=None)
    _hold_for_workflow()

    with pytest.raises(gpu_lease.LeaseBusyError):
        prep_jobs.requeue_prep_job(job.id, start_now=True)

    after = db.get_job(job.id)
    assert (after.state, after.exit_code, after.finished_at) == ("failed", 1, "2026-01-01")
    assert started == []


def test_requeue_start_now_takes_the_lease(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)
    job = prep_jobs.enqueue_prep_job("tag", _TOML)
    db.update_job(job.id, state="stopped", exit_code=1, finished_at="2026-01-01")

    prep_jobs.requeue_prep_job(job.id, start_now=True)

    assert started == [job.id]
    assert [r["holder_id"] for r in gpu_lease.snapshot()] == [f"job:{job.id}"]


def test_start_now_behind_a_running_job_still_queues(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged contract: with a run active the job waits its FIFO turn, no error."""
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)
    db.create_job(config_path="/tmp/x.toml", log_path="/tmp/x.log", state="running")

    job = prep_jobs.enqueue_prep_job("tag", _TOML, start_now=True)

    assert job.state == "pending"
    assert started == []


def test_the_prep_route_answers_409_when_the_gpu_is_held(
    ui_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)
    folder = tmp_path / "imgs"
    folder.mkdir()
    _hold_for_workflow()

    res = ui_client.post(
        "/api/v1/prep/jobs",
        json={"stage": "quality", "config": {"path": str(folder)}, "start_now": True},
    )

    assert res.status_code == 409, res.text
    assert "GPU" in res.json()["detail"]
    assert started == []
