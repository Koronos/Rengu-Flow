"""Job queue: enqueue, reorder, update pending jobs, start next when idle."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any
import toml

from renga_flow_ui import configs_store, db, jobs, library_db
from renga_flow_ui.settings import logs_dir


def _job_sort_key(job: db.JobRecord) -> tuple:
    state_order = {
        "running": 0,
        "stopping": 1,
        "pending": 2,
        "finished": 3,
        "stopped": 4,
        "failed": 5,
    }
    pos = job.queue_position if job.queue_position is not None else 999999
    return (state_order.get(job.state, 9), pos, job.started_at or "")


def list_jobs_sorted(limit: int = 200) -> list[db.JobRecord]:
    jobs.refresh_all_jobs()
    rows = db.list_jobs(limit=limit * 2)
    rows.sort(key=_job_sort_key)
    return rows[:limit]


def has_active_runner() -> bool:
    for job in db.list_jobs(limit=500):
        if job.state in ("running", "stopping"):
            return True
    return False


def next_queue_position() -> int:
    pending = [j for j in db.list_jobs(limit=500) if j.state == "pending"]
    if not pending:
        return 0
    positions = [j.queue_position for j in pending if j.queue_position is not None]
    return (max(positions) + 1) if positions else len(pending)


def try_start_next() -> db.JobRecord | None:
    """Start the first pending job if nothing is running."""
    if has_active_runner():
        return None
    pending = sorted(
        [j for j in db.list_jobs(limit=500) if j.state == "pending"],
        key=lambda j: (
            j.queue_position if j.queue_position is not None else 999999,
            j.started_at or "",
        ),
    )
    if not pending:
        return None
    job = pending[0]
    if not job.config_path or not Path(job.config_path).is_file():
        db.update_job(job.id, state="failed", finished_at=_now(), exit_code=-1)
        return db.get_job(job.id)
    jobs.start_job(job)
    return db.get_job(job.id)


def prepare_job(
    *,
    config_id: str | None,
    content: str | None,
    num_gpus: int,
    resume_from: str | None,
    output_dir: str | None,
    extra_args: str,
    reset_dataloader: bool,
    reset_optimizer: bool,
    queue_position: int | None = None,
    source_run_dir: str | None = None,
) -> db.JobRecord:
    if config_id:
        content = configs_store.read_config_text(config_id)
    elif content:
        pass
    elif source_run_dir:
        from renga_flow_ui.run_config import read_run_config_text

        content = read_run_config_text(source_run_dir)
    else:
        raise ValueError("Provide config_id, content, or source_run_dir")

    job_stub = db.create_job(
        config_path="",
        config_id=config_id,
        log_path=str(logs_dir() / "pending.log"),
        num_gpus=num_gpus,
        resume_from=resume_from,
        output_dir=output_dir,
        extra_args=extra_args,
        queue_position=queue_position if queue_position is not None else next_queue_position(),
        source_run_dir=source_run_dir,
    )
    staging = configs_store.materialize_staging(content, job_stub.id)
    extra: list[str] = []
    if reset_dataloader:
        extra.append("--reset_dataloader")
    if reset_optimizer:
        extra.append("--reset_optimizer")
    extra_s = " ".join(extra) or extra_args
    cfg = toml.loads(staging.read_text(encoding="utf-8"))
    out_dir = output_dir or cfg.get("output_dir", "output")
    run_dir_for_job: str | None = None
    if source_run_dir:
        run_dir_for_job = str(Path(source_run_dir).resolve())
    db.update_job(
        job_stub.id,
        config_path=str(staging),
        log_path=str(logs_dir() / f"{job_stub.id}.log"),
        output_dir=out_dir,
        extra_args=extra_s,
        run_dir=run_dir_for_job,
    )
    return db.get_job(job_stub.id)


def enqueue_continue_run(
    run_path: str,
    content: str,
    *,
    config_id: str | None = None,
    save_to_library: bool = False,
    num_gpus: int = 1,
    extra_args: str = "",
    reset_dataloader: bool = False,
    reset_optimizer: bool = False,
    enqueue: bool = True,
    start_immediately: bool = False,
) -> db.JobRecord:
    """Queue training that resumes ``run_path`` using ``content`` (typically edited run TOML)."""
    from renga_flow_ui.job_import import resolve_run_path
    from renga_flow_ui.run_config import resume_checkpoint_arg

    run_dir = resolve_run_path(run_path)
    cfg = toml.loads(content)
    resume_arg = resume_checkpoint_arg(run_dir, cfg)
    cid = config_id
    if save_to_library:
        cid = library_db._safe_id(config_id or f"{run_dir.name}_continued")
        configs_store.write_config_text(cid, content)

    kwargs = dict(
        config_id=cid if save_to_library else None,
        content=content,
        num_gpus=num_gpus,
        resume_from=resume_arg,
        output_dir=str(cfg.get("output_dir", "output")),
        extra_args=extra_args,
        reset_dataloader=reset_dataloader,
        reset_optimizer=reset_optimizer,
        source_run_dir=str(run_dir),
    )
    if start_immediately or not enqueue:
        return start_job_immediately(**kwargs)
    return enqueue_job(**kwargs)


def enqueue_job(**kwargs: Any) -> db.JobRecord:
    job = prepare_job(**kwargs)
    try_start_next()
    return db.get_job(job.id)


def start_job_immediately(**kwargs: Any) -> db.JobRecord:
    """Create pending job at front of queue and try to start."""
    job = prepare_job(**kwargs, queue_position=0)
    _normalize_queue_positions()
    bump_pending_after(job.id)
    try_start_next()
    return db.get_job(job.id)


def update_pending_job(
    job_id: str,
    *,
    config_id: str | None = None,
    num_gpus: int | None = None,
    resume_from: str | None = None,
    output_dir: str | None = None,
    extra_args: str | None = None,
    reset_dataloader: bool | None = None,
    reset_optimizer: bool | None = None,
) -> db.JobRecord:
    job = db.get_job(job_id)
    if job.state != "pending":
        raise ValueError("Only pending jobs can be edited")

    fields: dict[str, Any] = {}
    if num_gpus is not None:
        fields["num_gpus"] = num_gpus
    if resume_from is not None:
        fields["resume_from"] = resume_from or None
    if output_dir is not None:
        fields["output_dir"] = output_dir or None
    if extra_args is not None:
        fields["extra_args"] = extra_args

    if config_id is not None:
        content = configs_store.read_config_text(config_id)
        staging = configs_store.materialize_staging(content, job_id)
        fields["config_id"] = config_id
        fields["config_path"] = str(staging)
        cfg = toml.loads(staging.read_text(encoding="utf-8"))
        if output_dir is None and "output_dir" in cfg:
            fields.setdefault("output_dir", cfg.get("output_dir", "output"))

    if reset_dataloader is not None or reset_optimizer is not None:
        extra = shlex.split(job.extra_args) if job.extra_args else []
        flags = {"--reset_dataloader", "--reset_optimizer"}
        extra = [a for a in extra if a not in flags]
        if reset_dataloader:
            extra.append("--reset_dataloader")
        if reset_optimizer:
            extra.append("--reset_optimizer")
        fields["extra_args"] = " ".join(extra)

    if fields:
        db.update_job(job_id, **fields)
    return db.get_job(job_id)


def delete_pending_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if job.state != "pending":
        raise ValueError("Only pending jobs can be removed from the queue")
    db.delete_job(job_id)
    _normalize_queue_positions()


def move_queue(job_id: str, direction: str) -> db.JobRecord:
    job = db.get_job(job_id)
    if job.state != "pending":
        raise ValueError("Only pending jobs can be reordered")
    pending = sorted(
        [j for j in db.list_jobs(limit=500) if j.state == "pending"],
        key=lambda j: (
            j.queue_position if j.queue_position is not None else 999999,
            j.started_at or "",
        ),
    )
    idx = next((i for i, j in enumerate(pending) if j.id == job_id), None)
    if idx is None:
        return job
    if direction == "up" and idx > 0:
        other = pending[idx - 1]
        a, b = job.queue_position, other.queue_position
        db.update_job(job.id, queue_position=b)
        db.update_job(other.id, queue_position=a)
    elif direction == "down" and idx < len(pending) - 1:
        other = pending[idx + 1]
        a, b = job.queue_position, other.queue_position
        db.update_job(job.id, queue_position=b)
        db.update_job(other.id, queue_position=a)
    return db.get_job(job_id)


def bump_pending_after(job_id: str) -> None:
    """Ensure job_id is first in pending queue."""
    job = db.get_job(job_id)
    if job.state != "pending":
        return
    pending = [j for j in db.list_jobs(limit=500) if j.state == "pending" and j.id != job_id]
    for i, j in enumerate(sorted(pending, key=lambda x: x.queue_position or 0)):
        db.update_job(j.id, queue_position=i + 1)
    db.update_job(job_id, queue_position=0)


def _normalize_queue_positions() -> None:
    pending = sorted(
        [j for j in db.list_jobs(limit=500) if j.state == "pending"],
        key=lambda j: (
            j.queue_position if j.queue_position is not None else 999999,
            j.started_at or "",
        ),
    )
    for i, job in enumerate(pending):
        if job.queue_position != i:
            db.update_job(job.id, queue_position=i)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
