"""Tests for UI job queue helpers."""

import pytest

from rengu_flow_ui import configs_store, db, job_queue

_CFG = """
dataset = "x.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"
[optimizer]
type = "adamw"
"""


def test_enqueue_two_pending_sorted(ui_data_tmp, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_start(job: db.JobRecord) -> int:
        db.update_job(job.id, state="running", pid=99999)
        return 99999

    monkeypatch.setattr("rengu_flow_ui.jobs.start_job", fake_start)
    monkeypatch.setattr("rengu_flow_ui.jobs.poll_job", lambda job_id: db.get_job(job_id))

    cid = configs_store.insert_config(_CFG)
    j1 = job_queue.enqueue_job(
        config_id=cid,
        content=None,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    j2 = job_queue.enqueue_job(
        config_id=cid,
        content=None,
        num_gpus=2,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    assert j1.state == "running"
    assert j2.state == "pending"
    pending = [j for j in job_queue.list_jobs_sorted() if j.state == "pending"]
    assert len(pending) == 1
    assert pending[0].id == j2.id


def test_update_pending_job(ui_data_tmp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    cid = configs_store.insert_config(_CFG)
    job = job_queue.enqueue_job(
        config_id=cid,
        content=None,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    updated = job_queue.update_pending_job(
        job.id,
        num_gpus=4,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=True,
        reset_optimizer=False,
    )
    assert updated.num_gpus == 4
