"""GPU lease arbitration: the device PRIMARY KEY is the mutex, reaping is by holder validity.

No threads here on purpose: the mutex IS SQLite's uniqueness constraint, so two sequential calls
exercise the real thing. Threads would only add flakiness without testing anything extra.
"""

import os
from pathlib import Path

import pytest

from rengu_flow_ui import db, gpu_lease


@pytest.fixture(autouse=True)
def _single_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to a one-GPU host so tests never depend on the machine they run on."""
    monkeypatch.setattr(gpu_lease, "enumerate_devices", lambda: [0])


def _pending_job() -> db.JobRecord:
    return db.create_job(config_path="/tmp/x.toml", log_path="/tmp/x.log", state="pending")


def _held_devices() -> list[int]:
    return [row["device"] for row in gpu_lease.snapshot()]


# ------------------------------------------------------------------------------ exclusivity


def test_acquire_is_exclusive_and_release_frees(ui_data_tmp: Path) -> None:
    assert gpu_lease.acquire("train", "job:1", None) is True
    assert _held_devices() == [0]

    # A second holder cannot take the same device.
    assert gpu_lease.acquire("workflow", "wf:1:n1", None) is False
    assert [row["holder_id"] for row in gpu_lease.snapshot()] == ["job:1"]

    gpu_lease.release("job:1")
    assert gpu_lease.snapshot() == []
    assert gpu_lease.acquire("workflow", "wf:1:n1", None) is True


def test_release_of_unknown_holder_is_a_noop(ui_data_tmp: Path) -> None:
    gpu_lease.release("job:404")  # must not raise
    assert gpu_lease.snapshot() == []

    gpu_lease.acquire("train", "job:1", None)
    gpu_lease.release("job:404")
    assert [row["holder_id"] for row in gpu_lease.snapshot()] == ["job:1"]


def test_acquire_twice_by_the_same_holder_fails(ui_data_tmp: Path) -> None:
    """The table is a mutex, not a refcount: re-taking a held device conflicts like any other."""
    assert gpu_lease.acquire("train", "job:1", None) is True
    assert gpu_lease.acquire("train", "job:1", None) is False
    assert _held_devices() == [0]


def test_multi_device_acquire_is_all_or_nothing(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The multi-row insert must be one transaction.

    Under autocommit the executemany writes device 0, then conflicts on device 1 — leaving GPU 0
    taken forever by an acquisition that reported failure. This test is the reason `with conn:`
    wraps the insert.
    """
    monkeypatch.setattr(gpu_lease, "enumerate_devices", lambda: [0, 1])
    assert gpu_lease.acquire("workflow", "wf:1:n1", [1]) is True

    # "auto" wants every device; device 1 is taken, so the whole acquisition must fail.
    assert gpu_lease.acquire("train", "job:1", None) is False
    assert _held_devices() == [1]  # device 0 must NOT have been left inserted


def test_acquire_of_an_unenumerated_device_falls_back_to_host_exclusive(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host whose enumeration failed caches [-1]; a node pinned to device 0 must not slip past.

    Without the validation, the "auto" holder takes -1, the pinned holder takes 0, they do not
    conflict, and both land on the same physical GPU.
    """
    monkeypatch.setattr(gpu_lease, "enumerate_devices", lambda: [gpu_lease.HOST_DEVICE])
    assert gpu_lease.acquire("train", "job:1", None) is True
    assert _held_devices() == [gpu_lease.HOST_DEVICE]

    assert gpu_lease.acquire("workflow", "wf:1:n1", [0]) is False


# ------------------------------------------------------------------------------ bind_pid


def test_bind_pid_records_the_pid_and_reports_a_vanished_lease(ui_data_tmp: Path) -> None:
    gpu_lease.acquire("train", "job:1", None)
    assert gpu_lease.snapshot()[0]["pid"] is None

    assert gpu_lease.bind_pid("job:1", os.getpid()) is True
    row = gpu_lease.snapshot()[0]
    assert row["pid"] == os.getpid()
    assert row["pid_create_time"] is not None  # captured internally, for reuse detection

    # A lease reaped underneath the launch updates zero rows.
    gpu_lease.release("job:1")
    assert gpu_lease.bind_pid("job:1", os.getpid()) is False


# ------------------------------------------------------------------------------ reap_dead


def test_reap_dead_frees_a_terminal_job_and_keeps_an_active_one(ui_data_tmp: Path) -> None:
    job = _pending_job()
    holder = f"job:{job.id}"
    assert gpu_lease.acquire("train", holder, None) is True

    # pending: the launch may still be inside ensure_training_extras -> uv sync. Keep it.
    assert gpu_lease.reap_dead() == []
    db.update_job(job.id, state="running", pid=os.getpid())
    gpu_lease.bind_pid(holder, os.getpid())
    assert gpu_lease.reap_dead() == []
    assert _held_devices() == [0]

    # finished: the holder is gone, whatever the pid says.
    db.update_job(job.id, state="finished")
    assert gpu_lease.reap_dead() == [holder]
    assert gpu_lease.snapshot() == []


def test_reap_dead_frees_a_deleted_or_dequeued_job(ui_data_tmp: Path) -> None:
    dequeued = _pending_job()
    deleted = _pending_job()
    gpu_lease.acquire("train", f"job:{dequeued.id}", None)

    db.update_job(dequeued.id, state="new")  # job_queue.dequeue_job
    assert gpu_lease.reap_dead() == [f"job:{dequeued.id}"]

    gpu_lease.acquire("train", f"job:{deleted.id}", None)
    db.delete_job(deleted.id)  # job_queue.delete_job_record
    assert gpu_lease.reap_dead() == [f"job:{deleted.id}"]
    assert gpu_lease.snapshot() == []


def test_reap_dead_frees_a_dead_pid_but_not_a_live_one(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _pending_job()
    holder = f"job:{job.id}"
    db.update_job(job.id, state="running")
    gpu_lease.acquire("train", holder, None)
    gpu_lease.bind_pid(holder, os.getpid())

    # This very process is alive, so its lease survives a reap.
    assert gpu_lease.reap_dead() == []
    assert _held_devices() == [0]

    # Same row, same active job state — only the liveness of the pid changes.
    monkeypatch.setattr(gpu_lease, "pid_alive", lambda pid: False)
    assert gpu_lease.reap_dead() == [holder]
    assert gpu_lease.snapshot() == []


def test_reap_dead_frees_a_reused_pid(ui_data_tmp: Path) -> None:
    """A pid whose create_time no longer matches is a different process (post-reboot reuse)."""
    job = _pending_job()
    holder = f"job:{job.id}"
    db.update_job(job.id, state="running")
    gpu_lease.acquire("train", holder, None)
    gpu_lease.bind_pid(holder, os.getpid(), pid_create_time=1.0)

    assert gpu_lease.reap_dead() == [holder]


def test_reap_dead_ignores_a_workflow_holder_with_a_live_pid(ui_data_tmp: Path) -> None:
    """Only `job:` holders are reconciled against the jobs table; others go by pid alone."""
    gpu_lease.acquire("workflow", "wf:1:n1", None)
    gpu_lease.bind_pid("wf:1:n1", os.getpid())
    assert gpu_lease.reap_dead() == []
    assert _held_devices() == [0]


# ------------------------------------------------------------------------------ startup sweep


def test_reconcile_on_start_sweeps_unbound_leases(ui_data_tmp: Path) -> None:
    """An unbound lease is legitimate mid-launch but cannot survive the process that made it."""
    job = _pending_job()
    holder = f"job:{job.id}"
    gpu_lease.acquire("train", holder, None)

    # reap_dead leaves it alone (the row still says pending) ...
    assert gpu_lease.reap_dead() == []
    # ... but a restart means its launch never happened.
    assert gpu_lease.reconcile_on_start() == [holder]
    assert gpu_lease.snapshot() == []


def test_reconcile_on_start_keeps_a_bound_live_lease(ui_data_tmp: Path) -> None:
    job = _pending_job()
    holder = f"job:{job.id}"
    db.update_job(job.id, state="running")
    gpu_lease.acquire("train", holder, None)
    gpu_lease.bind_pid(holder, os.getpid())

    assert gpu_lease.reconcile_on_start() == []
    assert _held_devices() == [0]


# ------------------------------------------------------------------------------ wait_reason


def test_wait_reason_names_the_device_and_the_holder(ui_data_tmp: Path) -> None:
    assert gpu_lease.wait_reason(None) == ""
    gpu_lease.acquire("train", "job:42", None)
    assert gpu_lease.wait_reason(None) == "Waiting for GPU 0 — held by training job 42."


# ------------------------------------------------------------------------------ training lane


def test_devices_for_job_pins_only_a_single_gpu_run(ui_data_tmp: Path) -> None:
    """num_gpus >= 2 is always host-exclusive: DeepSpeed enumerates every device."""
    from rengu_flow_ui.job_queue import _devices_for_job

    job = _pending_job()
    assert _devices_for_job(job) is None  # gpu_index is NULL by default

    db.update_job(job.id, gpu_index=1)
    assert _devices_for_job(db.get_job(job.id)) == [1]

    db.update_job(job.id, num_gpus=2)
    assert _devices_for_job(db.get_job(job.id)) is None
