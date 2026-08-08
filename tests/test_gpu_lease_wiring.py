"""How the training lane is *wired* to the GPU lease — the part a green suite does not prove.

`test_gpu_lease.py` covers the lease primitive in isolation. These tests cover the call sites:
that `try_start_next` acquires, that `poll_job` releases, that a launch which blows up still
gives the GPU back, that a lease reaped mid-launch kills the process it stranded, and that the
background tick neither dies nor starts a queue the user never started. Every one of them fails
against a plausible wrong implementation while the rest of the suite stays green — which is the
only reason they exist.
"""

import os
import time
from pathlib import Path

import pytest

from rengu_flow_ui import db, gpu_lease, job_queue, jobs, queue_poller


@pytest.fixture(autouse=True)
def _single_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to a one-GPU host so tests never depend on the machine they run on."""
    monkeypatch.setattr(gpu_lease, "enumerate_devices", lambda: [0])


def _pending_job(tmp_path: Path, name: str = "a") -> db.JobRecord:
    """A queued job whose ``config_path`` really exists (``try_start_next`` checks)."""
    config = tmp_path / f"{name}.toml"
    config.write_text("dataset = 'x'\n", encoding="utf-8")
    return db.create_job(
        config_path=str(config),
        log_path=str(tmp_path / f"{name}.log"),
        state="pending",
        queue_position=job_queue.next_queue_position(),
    )


def _fake_start(pid: int):
    """A ``jobs.start_job`` stand-in that flips the row to running and reports ``pid``."""
    started: list[int] = []

    def start(job: db.JobRecord, **_kwargs: object) -> int:
        started.append(job.id)
        db.update_job(job.id, state="running", pid=pid)
        return pid

    return start, started


# ------------------------------------------------------------------------------ acquire / release


def test_try_start_next_acquires_the_lease_for_the_job_it_starts(
    ui_data_tmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _pending_job(tmp_path)
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)

    assert gpu_lease.snapshot() == []
    out = job_queue.try_start_next()

    assert out is not None and out.id == job.id and out.state == "running"
    assert started == [job.id]
    rows = gpu_lease.snapshot()
    assert [(r["holder_kind"], r["holder_id"]) for r in rows] == [("train", f"job:{job.id}")]
    assert rows[0]["pid"] == os.getpid()  # bind_pid ran, so a reap can judge this holder


def test_poll_job_releases_the_lease_on_the_terminal_transition(
    ui_data_tmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eager release must fire on its own — reap_dead is not allowed to cover for it here.

    The job is ``stopping``, so it lands in ``stopped`` and ``poll_job`` does NOT advance the
    queue: nothing else runs a reap, and the lease is freed only if the release really happened.
    """
    job = _pending_job(tmp_path)
    holder = f"job:{job.id}"
    db.update_job(job.id, state="stopping", pid=4321)
    assert gpu_lease.acquire("train", holder, None) is True
    gpu_lease.bind_pid(holder, 4321, pid_create_time=1.0)

    monkeypatch.setattr(jobs, "pid_alive", lambda pid: False)
    # Zero-padded, exactly as an id arrives off an HTTP path: it resolves to the same row, so
    # the release must derive its holder id from the row rather than from this string.
    out = jobs.poll_job(f"0{job.id}")

    assert out.state == "stopped"
    assert gpu_lease.snapshot() == []


def test_start_job_raising_system_exit_releases_the_lease_and_a_retry_starts(
    ui_data_tmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ensure_training_extras`` -> ``ensure_profiles`` raises ``SystemExit``, not ``Exception``.

    Caught only as ``Exception`` the release is skipped, the row stays ``pending`` and the lease
    keeps ``pid IS NULL`` — which by the no-timer rule is immortal, so every retry collides with
    the job's own lease and only a server restart revives the queue.
    """
    job = _pending_job(tmp_path)
    calls: list[int] = []

    def flaky_start(j: db.JobRecord, **_kwargs: object) -> int:
        calls.append(j.id)
        if len(calls) == 1:
            raise SystemExit("uv is not installed")
        db.update_job(j.id, state="running", pid=os.getpid())
        return os.getpid()

    monkeypatch.setattr(jobs, "start_job", flaky_start)

    with pytest.raises(SystemExit):
        job_queue.try_start_next()
    assert gpu_lease.snapshot() == []  # no immortal pid-NULL lease left behind
    assert db.get_job(job.id).state == "pending"

    # Fixing the environment revives the queue by itself — no server restart.
    out = job_queue.try_start_next()
    assert out is not None and out.state == "running"
    assert calls == [job.id, job.id]


def test_stop_job_on_a_pidless_row_unwedges_the_queue(
    ui_data_tmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stop_job`` with ``pid is None`` never touches the lease; the reap in try_start_next does."""
    stuck = _pending_job(tmp_path, "stuck")
    db.update_job(stuck.id, state="running")  # running with pid IS NULL
    assert gpu_lease.acquire("train", f"job:{stuck.id}", None) is True

    nxt = _pending_job(tmp_path, "next")
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)

    assert job_queue.try_start_next() is None  # wedged: runner busy and GPU taken
    assert started == []

    jobs.stop_job(stuck.id)
    assert db.get_job(stuck.id).state == "stopped"

    out = job_queue.try_start_next()
    assert out is not None and out.id == nxt.id and out.state == "running"
    assert [r["holder_id"] for r in gpu_lease.snapshot()] == [f"job:{nxt.id}"]


# ------------------------------------------------------------------------------ bind_pid fallout


def test_bind_pid_failure_kills_the_launched_process_and_fails_the_row(
    ui_data_tmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lease reaped mid-launch leaves training running with ZERO leases — kill it.

    Reachable today: the row sits ``pending`` for the minutes ``uv sync`` takes, the user
    dequeues or deletes it (neither takes ``_start_lock``), the next reap frees the lease
    legitimately, and then ``start_job`` returns. Ignoring ``bind_pid``'s answer means another
    holder can take the GPU on top of a live trainer.
    """
    job = _pending_job(tmp_path)
    holder = f"job:{job.id}"

    def start_then_lose_the_lease(j: db.JobRecord, **_kwargs: object) -> int:
        db.update_job(j.id, state="running", pid=999_001)
        gpu_lease.release(holder)  # what a reap does once the row is dequeued/deleted
        return 999_001

    killed: list[int] = []
    monkeypatch.setattr(jobs, "start_job", start_then_lose_the_lease)
    monkeypatch.setattr(job_queue, "terminate_process_tree", lambda pid: killed.append(pid))

    out = job_queue.try_start_next()

    assert killed == [999_001]
    assert out is not None and out.state == "failed"
    assert out.pid is None and out.exit_code == -1
    assert "lease" in Path(job.log_path).read_text(encoding="utf-8").lower()
    assert gpu_lease.snapshot() == []


def test_a_launcher_that_reports_no_pid_is_not_treated_as_a_lost_lease(
    ui_data_tmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``bind_pid(pid=None)`` means "nothing to bind", never "the lease evaporated".

    ``False`` is a kill-the-process signal. Conflating the two would kill on any launcher that
    does not report a pid — which is exactly what the prep-route test stubs do.
    """
    job = _pending_job(tmp_path)

    def start_without_reporting_a_pid(j: db.JobRecord, **_kwargs: object) -> None:
        db.update_job(j.id, state="running")
        return None

    killed: list[int] = []
    monkeypatch.setattr(jobs, "start_job", start_without_reporting_a_pid)
    monkeypatch.setattr(job_queue, "terminate_process_tree", lambda pid: killed.append(pid))

    out = job_queue.try_start_next()

    assert killed == []
    assert out is not None and out.state == "running"
    assert gpu_lease.bind_pid(f"job:{job.id}", None) is None
    assert [r["holder_id"] for r in gpu_lease.snapshot()] == [f"job:{job.id}"]


# ------------------------------------------------------------------------------ the poller tick


def _explode() -> None:
    raise SystemExit("uv is missing")


def test_tick_survives_a_system_exit_and_still_reconciles_active_runs(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-step guards: a failing reap must not abort the reconciliation that predates it."""
    refreshed: list[int] = []
    monkeypatch.setattr(gpu_lease, "reap_dead", _explode)
    monkeypatch.setattr(jobs, "refresh_all_jobs", lambda: refreshed.append(1))

    queue_poller._tick()  # must not raise

    assert refreshed == [1]


def test_the_poller_thread_survives_a_system_exit(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This thread owns reap_dead: if it dies, no lease is ever freed again."""
    ticks: list[int] = []
    monkeypatch.setattr(gpu_lease, "reap_dead", _explode)
    monkeypatch.setattr(jobs, "refresh_all_jobs", lambda: ticks.append(1))

    queue_poller.start_poller(interval=0.01)
    try:
        deadline = time.monotonic() + 10.0
        while len(ticks) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(ticks) >= 3  # the loop kept ticking across repeated SystemExit
        assert queue_poller._thread is not None and queue_poller._thread.is_alive()
    finally:
        queue_poller.stop_poller()


def test_tick_does_not_start_an_idle_queue(
    ui_data_tmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare tick must never start a run the user did not start.

    "Add to the pending queue only — do NOT start" (``job_queue.enqueue_job``) and "a bare
    refresh that finds nothing active never starts an idle queue" (``jobs.refresh_all_jobs``)
    are user-visible contracts: enqueue with start_now=false, walk away, and DeepSpeed must not
    be running when you come back. The unconditional ``try_start_next`` belongs to Phase 1.
    """
    job = _pending_job(tmp_path)
    start, started = _fake_start(os.getpid())
    monkeypatch.setattr(jobs, "start_job", start)

    queue_poller._tick()
    queue_poller._tick()

    assert started == []
    assert db.get_job(job.id).state == "pending"
    assert gpu_lease.snapshot() == []
