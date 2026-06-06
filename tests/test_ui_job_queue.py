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


def test_dequeue_job_keeps_as_draft(job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert job.state == "pending"
    out = job_queue.dequeue_job(job.id)
    # Removed from the queue but kept: now a saved (new) draft with no queue slot.
    assert out.state == "new"
    assert out.queue_position is None
    assert db.get_job(job.id).config_content == job.config_content
    # Re-queueing it works (round-trip).
    assert job_queue.enqueue_existing(job.id).state == "pending"


def test_dequeue_rejects_non_pending(job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert draft.state == "new"
    with pytest.raises(ValueError, match="pending"):
        job_queue.dequeue_job(draft.id)


def test_dequeue_endpoint(ui_client, job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
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
    r = ui_client.post(f"/api/v1/jobs/{job.id}/dequeue")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "new"


def test_edit_pending_writes_config_to_run_folder(
    job_content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing a run that owns an output folder updates that folder's on-disk TOML."""
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    folder = tmp_path / "run1"
    folder.mkdir()
    (folder / "train.toml").write_text("run_name = 'stale'\n", encoding="utf-8")

    job = job_queue.prepare_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
        source_run_dir=str(folder),
    )
    assert job.run_dir  # the run owns the folder

    job_queue.update_pending_job(job.id, content=job_content)

    written = (folder / "train.toml").read_text(encoding="utf-8")
    staged = Path(db.get_job(job.id).config_path).read_text(encoding="utf-8")
    assert written == staged  # folder TOML now mirrors the materialized config
    assert "stale" not in written  # the old folder config was replaced


def test_continue_existing_reuses_record(
    job_content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuing a run edits the same record and re-queues it — no new row is created."""
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    folder = tmp_path / "run1"
    folder.mkdir()
    (folder / "train.toml").write_text("run_name = 'stale'\n", encoding="utf-8")

    job = job_queue.prepare_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
        source_run_dir=str(folder),
    )
    db.update_job(job.id, state="finished")
    before = len(db.list_jobs())

    cont = job_queue.continue_existing(
        job.id, content=job_content, from_scratch=True, num_gpus=2
    )
    assert cont.id == job.id  # same record
    assert cont.state == "pending"  # re-queued
    assert cont.num_gpus == 2
    assert len(db.list_jobs()) == before  # no duplicate row
    assert "stale" not in (folder / "train.toml").read_text(encoding="utf-8")

    # "Save for later" on a continue keeps the same record as a draft.
    draft = job_queue.continue_existing(
        job.id, content=job_content, from_scratch=True, enqueue=False
    )
    assert draft.id == job.id
    assert draft.state == "new"


def test_continue_existing_specific_checkpoint_pins_folder(
    job_content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming a specific checkpoint passes the tag AND pins the run folder via --run_dir,
    so the trainer resolves output/<run>/<tag> instead of the bogus output/<tag>."""
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    folder = tmp_path / "run_ckpt"
    folder.mkdir()
    (folder / "train.toml").write_text("run_name = 'r'\n", encoding="utf-8")
    (folder / "global_step40").mkdir()  # the checkpoint to resume

    job = job_queue.prepare_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
        source_run_dir=str(folder),
    )
    db.update_job(job.id, state="finished")
    cont = job_queue.continue_existing(job.id, content=job_content, resume_from="global_step40")
    assert cont.resume_from == "global_step40"  # the tag (not a folder)
    assert "--run_dir" in (cont.extra_args or "")
    assert str(folder.resolve()) in (cont.extra_args or "")


def test_continue_existing_without_folder_reuses_record(
    job_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that never made a folder (failed at setup) is retried in place — no new row."""
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
    db.update_job(job.id, state="failed", run_dir=None, source_run_dir=None)
    before = len(db.list_jobs())

    cont = job_queue.continue_existing(job.id, content=job_content)
    assert cont.id == job.id  # same record, no duplicate
    assert cont.state == "pending"
    assert cont.resume_from is None  # nothing to resume — from scratch
    assert len(db.list_jobs()) == before


def test_enqueue_does_not_autostart(job_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding to the queue only enqueues (pending) — it must never launch a runner."""
    calls: list[int] = []
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: calls.append(1))
    job = job_queue.enqueue_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    assert job.state == "pending"
    assert calls == []


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
    # Enqueue no longer auto-starts; the run waits as pending until explicitly started.
    assert j1.state == "pending"
    assert started == []

    # Explicitly begin processing the queue (the "Start" button), which starts the first pending.
    first = job_queue.try_start_next()
    assert first is not None and first.id == j1.id and first.state == "running"
    assert started == [j1.id]

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

    # Once running, the queue drains: finishing j1 starts the next pending.
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


def test_clone_run_strips_resume_from_checkpoint(
    job_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import toml

    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    # A source config that carries a resume pointer must not pass it to a fresh clone,
    # or the clone would resume into (and show the stats of) the source run's folder.
    src = job_queue.prepare_job(
        # Top-level key (before any [table]) so it is a real top-level resume pointer.
        content="resume_from_checkpoint = true\n" + job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )
    assert toml.loads(src.config_content).get("resume_from_checkpoint") is True
    clone = job_queue.clone_run(src.id)
    assert "resume_from_checkpoint" not in toml.loads(clone.config_content)


def test_merge_job_cli_args_run_dir_managed() -> None:
    """--run_dir is managed like the cache flags: stripped, then re-added only when given."""
    from rengu_flow_ui.job_queue import merge_job_cli_args

    # Added when requested.
    assert merge_job_cli_args("", run_dir="output/run_a") == "--run_dir output/run_a"
    # A stale pin is dropped when none is requested (so a clone gets a fresh folder).
    assert merge_job_cli_args("--run_dir output/old", run_dir=None) == ""
    # A stale pin is replaced, other flags preserved.
    out = merge_job_cli_args(
        "--reset_optimizer --run_dir output/old", run_dir="output/new"
    )
    assert "--reset_optimizer" in out
    assert "--run_dir output/new" in out
    assert "output/old" not in out


def test_merge_job_cli_args_run_dir_spaces() -> None:
    """A pin with spaces (custom output path) survives the extra_args round-trip."""
    import shlex

    from rengu_flow_ui.job_queue import merge_job_cli_args

    out = merge_job_cli_args("", run_dir="/data/Link to train/run_x")
    assert shlex.split(out) == ["--run_dir", "/data/Link to train/run_x"]


def test_continue_from_scratch_pins_run_folder(
    job_content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continue-from-scratch reuses the run's folder (pins --run_dir) instead of a new one."""
    import shlex

    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)
    folder = tmp_path / "run_fs"
    folder.mkdir()
    job = job_queue.prepare_job(
        content=job_content,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
        source_run_dir=str(folder),
    )
    db.update_job(job.id, state="finished")

    cont = job_queue.continue_existing(job.id, content=job_content, from_scratch=True)
    assert cont.resume_from is None  # from scratch: no checkpoint resume
    toks = shlex.split(cont.extra_args)
    assert "--run_dir" in toks
    assert toks[toks.index("--run_dir") + 1] == str(folder.resolve())

    # Cloning that job for a brand-new run must NOT inherit the folder pin.
    clone = job_queue.clone_run(cont.id)
    assert "--run_dir" not in shlex.split(clone.extra_args)


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
