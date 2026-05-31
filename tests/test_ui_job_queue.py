"""Extended job queue behaviour tests."""

from pathlib import Path

import pytest

from rengu_flow_ui import db, job_queue

JOB_TOML = """
dataset = "examples/minimal_dataset.toml"
output_dir = "output"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"

[optimizer]
type = "adamw"
lr = 1.0e-4

epochs = 1
micro_batch_size_per_gpu = 1
"""


@pytest.fixture
def job_content(ui_data_tmp: Path) -> str:
    return JOB_TOML


def test_move_queue_up_down(job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)

    j1 = job_queue.enqueue_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    j2 = job_queue.enqueue_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    assert j1.state == "pending"
    assert j2.state == "pending"

    moved = job_queue.move_queue(j2.id, "up")
    assert moved.queue_position <= j1.queue_position

    job_queue.move_queue(j2.id, "down")


def test_delete_pending_job(job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    job = job_queue.enqueue_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    job_queue.delete_job_record(job.id)
    with pytest.raises(KeyError):
        db.get_job(job.id)


def test_try_start_next_after_finish(job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []

    def fake_start(job: db.JobRecord) -> int:
        started.append(job.id)
        db.update_job(job.id, state="running", pid=1)
        return 1

    monkeypatch.setattr("rengu_flow_ui.jobs.start_job", fake_start)
    monkeypatch.setattr("rengu_flow_ui.jobs.poll_job", lambda job_id: db.get_job(job_id))

    j1 = job_queue.enqueue_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    assert j1.state == "running"
    assert len(started) == 1

    j2 = job_queue.enqueue_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    assert j2.state == "pending"

    db.update_job(j1.id, state="finished", pid=None)
    nxt = job_queue.try_start_next()
    assert nxt is not None
    assert nxt.id == j2.id
    assert nxt.state == "running"


def test_prepare_job_reset_flags(job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    job = job_queue.prepare_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=True,
        reset_optimizer=True,
    )
    assert "--reset_dataloader" in job.extra_args
    assert "--reset_optimizer" in job.extra_args
    assert Path(job.config_path).is_file()


def test_prepare_job_cache_flags(job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    cache_job = job_queue.prepare_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
        cache_only=True,
    )
    assert "--cache_only" in cache_job.extra_args
    assert "--trust_cache" not in cache_job.extra_args

    train_job = job_queue.prepare_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
        trust_cache=True,
    )
    assert "--trust_cache" in train_job.extra_args
    assert "--cache_only" not in train_job.extra_args


def test_merge_job_cli_args_rejects_cache_only_with_trust() -> None:
    with pytest.raises(ValueError, match="cache_only and trust_cache"):
        job_queue.merge_job_cli_args("", cache_only=True, trust_cache=True)


def test_prepare_job_snapshots_config_content(
    job_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    job = job_queue.prepare_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    # The run carries its own config snapshot (library content), independent of the library.
    assert 'type = "sdxl"' in job.config_content


def test_clone_run_creates_fresh_run_from_snapshot(
    job_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    src = job_queue.prepare_job(
        content=job_content,
        num_gpus=2,
        resume_from="/tmp/prev/checkpoint",
        output_dir=None,
        extra_args="--foo",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    clone = job_queue.clone_run(src.id)

    assert clone.id != src.id
    assert clone.state == "pending"
    assert clone.config_content == src.config_content  # same config
    assert clone.num_gpus == src.num_gpus  # inherited runtime knobs
    assert clone.extra_args == src.extra_args
    assert clone.resume_from is None  # fresh: no data from the previous run
    assert clone.run_dir is None


def test_save_draft_creates_new_without_staging(
    job_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    draft = job_queue.save_draft(
        content=job_content,
        num_gpus=3,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
        trust_cache=True,
    )
    assert draft.state == "new"
    assert draft.queue_position is None
    assert draft.config_path == ""  # not materialized yet
    assert draft.config_content == job_content
    assert draft.trust_cache is True
    # A draft never competes for the runner.
    assert job_queue.try_start_next() is None


def test_enqueue_existing_promotes_draft(
    job_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    draft = job_queue.save_draft(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    promoted = job_queue.enqueue_existing(draft.id)
    assert promoted.id == draft.id
    assert promoted.state == "pending"
    assert promoted.queue_position is not None
    assert Path(promoted.config_path).is_file()  # staged on promotion
    # Only drafts can be enqueued.
    with pytest.raises(ValueError):
        job_queue.enqueue_existing(promoted.id)


def test_reorder_queue_sets_positions(
    job_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    kwargs = dict(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    a = job_queue.enqueue_job(**kwargs)
    b = job_queue.enqueue_job(**kwargs)
    c = job_queue.enqueue_job(**kwargs)

    pending = job_queue.reorder_queue([c.id, a.id, b.id])
    order = [j.id for j in pending]
    assert order == [c.id, a.id, b.id]
    assert [j.queue_position for j in pending] == [0, 1, 2]


def test_next_run_name_avoids_collisions(
    job_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    from rengu_flow_ui import run_staging

    named = job_content + '\nrun_name = "my_run"\n'
    job_queue.save_draft(
        content=named,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    # "my_run" is taken by the draft above → first free suffix is _2.
    assert run_staging.next_run_name("my_run") == "my_run_2"
    # Cloning a clone keeps a single counter rather than stacking suffixes.
    assert run_staging.next_run_name("my_run_2") == "my_run_2"
