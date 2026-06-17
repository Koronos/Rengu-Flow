"""The background queue poller must advance the queue without any UI/HTTP activity."""

from pathlib import Path

import pytest

from rengu_flow_ui import db, job_queue, queue_poller

_CFG = """
dataset = "x.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"
[optimizer]
type = "adamw"
"""


def _enqueue(num_gpus: int) -> db.JobRecord:
    return job_queue.enqueue_job(
        content=_CFG,
        num_gpus=num_gpus,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )


def test_tick_advances_queue_without_http(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished run's successor starts on a bare poller tick — no GET /jobs in the loop."""

    def fake_start(job: db.JobRecord) -> int:
        db.update_job(job.id, state="running", pid=99999)
        return 99999

    # Patch the real launcher only; poll_job/refresh_all_jobs run for real (that's the path
    # the poller exercises).
    monkeypatch.setattr("rengu_flow_ui.jobs.start_job", fake_start)

    j1 = _enqueue(1)
    j2 = _enqueue(2)

    started = job_queue.try_start_next()
    assert started is not None and started.id == j1.id and started.state == "running"
    assert db.get_job(j2.id).state == "pending"

    # Simulate j1's process exiting cleanly: pid gone + a success marker in its own log.
    Path(db.get_job(j1.id).log_path).write_text(
        "Process 1 exits successfully.\n", encoding="utf-8"
    )
    db.update_job(j1.id, pid=None)

    # One reconciliation pass, with NO HTTP request anywhere.
    queue_poller._tick()

    assert db.get_job(j1.id).state == "finished"
    assert db.get_job(j2.id).state == "running"


def test_start_poller_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two start calls share one thread; stop tears it down."""
    # A no-op tick keeps the thread idle; the long interval means it just waits on _stop.
    monkeypatch.setattr(queue_poller, "_tick", lambda: None)
    try:
        queue_poller.start_poller(interval=100.0)
        first = queue_poller._thread
        assert first is not None and first.is_alive()

        queue_poller.start_poller(interval=100.0)
        assert queue_poller._thread is first  # no second thread spawned
    finally:
        queue_poller.stop_poller()

    assert queue_poller._thread is None
    assert not first.is_alive()
