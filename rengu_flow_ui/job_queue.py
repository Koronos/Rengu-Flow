"""Job queue: enqueue, reorder, update pending jobs, start next when idle."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any
import toml

from rengu_flow_ui import db, jobs, run_staging
from rengu_flow_ui.settings import logs_dir


def _resolve_run_content(content: str | None, source_run_dir: str | None) -> str:
    """Return the run's TOML, reading it from ``source_run_dir`` when not given inline."""
    if content:
        return content
    if source_run_dir:
        from rengu_flow_ui.run_config import read_run_config_text

        return read_run_config_text(source_run_dir)
    raise ValueError("Provide content or source_run_dir")


def _job_sort_key(job: db.JobRecord) -> tuple:
    state_order = {
        "running": 0,
        "stopping": 1,
        "pending": 2,
        "new": 3,
        "finished": 4,
        "stopped": 5,
        "failed": 6,
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


def merge_job_cli_args(
    extra_args: str,
    *,
    cache_only: bool = False,
    trust_cache: bool = False,
    regenerate_cache: bool = False,
    reset_dataloader: bool = False,
    reset_optimizer: bool = False,
) -> str:
    """Merge job toggle flags into a CLI ``extra_args`` string (idempotent, deduped).

    ``cache_only`` (cache latents/text-embeds then exit) and ``trust_cache`` (skip the
    cache freshness check) are mutually exclusive — passing both raises ``ValueError``.
    Existing cache flags in ``extra_args`` are stripped first so this is the single
    source of truth for the cache toggles.
    """
    if cache_only and trust_cache:
        raise ValueError(
            "cache_only and trust_cache are mutually exclusive — pick one."
        )
    managed = {"--cache_only", "--trust_cache", "--regenerate_cache"}
    tokens = [t for t in (extra_args or "").split() if t not in managed]

    def _add(flag: str) -> None:
        if flag not in tokens:
            tokens.append(flag)

    if reset_dataloader:
        _add("--reset_dataloader")
    if reset_optimizer:
        _add("--reset_optimizer")
    if cache_only:
        _add("--cache_only")
    if trust_cache:
        _add("--trust_cache")
    if regenerate_cache:
        _add("--regenerate_cache")
    return " ".join(tokens)


def prepare_job(
    *,
    content: str | None,
    num_gpus: int,
    resume_from: str | None,
    output_dir: str | None,
    extra_args: str,
    reset_dataloader: bool,
    reset_optimizer: bool,
    cache_only: bool = False,
    trust_cache: bool = False,
    regenerate_cache: bool = False,
    queue_position: int | None = None,
    source_run_dir: str | None = None,
) -> db.JobRecord:
    content = _resolve_run_content(content, source_run_dir)

    job_stub = db.create_job(
        config_path="",
        log_path=str(logs_dir() / "pending.log"),
        num_gpus=num_gpus,
        resume_from=resume_from,
        output_dir=output_dir,
        extra_args=extra_args,
        queue_position=queue_position if queue_position is not None else next_queue_position(),
        source_run_dir=source_run_dir,
        # Snapshot the run's own config (library refs intact) so the run is self-contained
        # and can seed a clone even if a library config is later edited or deleted.
        config_content=content,
        cache_only=cache_only,
        trust_cache=trust_cache,
        regenerate_cache=regenerate_cache,
    )
    staging = run_staging.materialize_staging(content, job_stub.id)
    extra_s = merge_job_cli_args(
        extra_args,
        cache_only=cache_only,
        trust_cache=trust_cache,
        regenerate_cache=regenerate_cache,
        reset_dataloader=reset_dataloader,
        reset_optimizer=reset_optimizer,
    )
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
    num_gpus: int = 1,
    extra_args: str = "",
    reset_dataloader: bool = False,
    reset_optimizer: bool = False,
    resume_from: str | None = None,
    from_scratch: bool = False,
    enqueue: bool = True,
    start_immediately: bool = False,
) -> db.JobRecord:
    """Queue training that resumes ``run_path`` using ``content`` (typically edited run TOML).

    ``resume_from`` overrides the checkpoint to resume from (a ``global_stepN`` folder
    name); when omitted the run's ``latest`` pointer is used. ``from_scratch=True``
    ignores all checkpoints and trains from step 0 in the same folder.
    """
    from rengu_flow_ui.job_import import resolve_run_path
    from rengu_flow_ui.run_config import resume_checkpoint_arg

    run_dir = resolve_run_path(run_path)
    cfg = toml.loads(content)
    if from_scratch:
        resume_arg: str | None = None
    elif resume_from:
        resume_arg = resume_from
    else:
        resume_arg = resume_checkpoint_arg(run_dir, cfg)

    kwargs = dict(
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
    # Add to the pending queue only — do NOT start. Processing begins when the user explicitly
    # starts the queue; once running, it drains automatically (try_start_next on each finish).
    job = prepare_job(**kwargs)
    return db.get_job(job.id)


def start_job_immediately(**kwargs: Any) -> db.JobRecord:
    """Create pending job at front of queue and try to start."""
    job = prepare_job(**kwargs, queue_position=0)
    _normalize_queue_positions()
    bump_pending_after(job.id)
    try_start_next()
    return db.get_job(job.id)


def clone_run(
    source_job_id: str | int,
    *,
    num_gpus: int | None = None,
    output_dir: str | None = None,
    extra_args: str | None = None,
    start_immediately: bool = False,
) -> db.JobRecord:
    """Create a NEW run seeded with an existing run's config, without its runtime data.

    Implements "edit = create new": the clone reuses the source run's config snapshot
    (library refs intact) but starts fresh — new run_dir/output, no resume checkpoint,
    no inherited logs. The source run stays immutable as history.
    """
    src = db.get_job(source_job_id)
    content = src.config_content or ""
    if not content.strip():
        raise ValueError(f"Run {source_job_id} has no config content to clone")

    kwargs: dict[str, Any] = dict(
        content=content,
        num_gpus=num_gpus if num_gpus is not None else src.num_gpus,
        resume_from=None,  # fresh run: no data from the previous run
        output_dir=output_dir,
        extra_args=extra_args if extra_args is not None else src.extra_args,
        reset_dataloader=False,
        reset_optimizer=False,
        cache_only=src.cache_only,
        trust_cache=src.trust_cache,
        regenerate_cache=src.regenerate_cache,
    )
    if start_immediately:
        return start_job_immediately(**kwargs)
    return enqueue_job(**kwargs)


def save_draft(
    *,
    content: str | None,
    num_gpus: int,
    resume_from: str | None,
    output_dir: str | None,
    extra_args: str,
    reset_dataloader: bool,
    reset_optimizer: bool,
    cache_only: bool = False,
    trust_cache: bool = False,
    regenerate_cache: bool = False,
    source_run_dir: str | None = None,
) -> db.JobRecord:
    """Save a run as a ``new`` draft: stored config + params, no staging, no queue slot.

    A draft does not compete for the runner (``try_start_next`` only looks at ``pending``);
    it is materialized into staging only when promoted via :func:`enqueue_existing`.
    """
    content = _resolve_run_content(content, source_run_dir)
    extra_s = merge_job_cli_args(
        extra_args,
        cache_only=cache_only,
        trust_cache=trust_cache,
        regenerate_cache=regenerate_cache,
        reset_dataloader=reset_dataloader,
        reset_optimizer=reset_optimizer,
    )
    out_dir = output_dir or toml.loads(content).get("output_dir", "output")
    job = db.create_job(
        config_path="",
        log_path=str(logs_dir() / "draft.log"),
        state="new",
        num_gpus=num_gpus,
        resume_from=resume_from,
        output_dir=out_dir,
        extra_args=extra_s,
        queue_position=None,
        source_run_dir=str(Path(source_run_dir).resolve()) if source_run_dir else None,
        config_content=content,
        cache_only=cache_only,
        trust_cache=trust_cache,
        regenerate_cache=regenerate_cache,
    )
    return db.get_job(job.id)


def enqueue_existing(job_id: str | int) -> db.JobRecord:
    """Promote a saved ``new`` draft into the pending queue (does not start it).

    Like :func:`enqueue_job`, this only adds to the queue; the run starts when the user
    explicitly starts the queue (then it drains via ``try_start_next`` on each finish).
    """
    job = db.get_job(job_id)
    if job.state != "new":
        raise ValueError("Only saved (new) runs can be enqueued")
    content = job.config_content or ""
    if not content.strip():
        raise ValueError(f"Run {job_id} has no config content to enqueue")
    staging = run_staging.materialize_staging(content, job_id)
    cfg = toml.loads(staging.read_text(encoding="utf-8"))
    out_dir = job.output_dir or cfg.get("output_dir", "output")
    db.update_job(
        job_id,
        state="pending",
        config_path=str(staging),
        log_path=str(logs_dir() / f"{job_id}.log"),
        output_dir=out_dir,
        queue_position=next_queue_position(),
    )
    return db.get_job(job_id)


def update_pending_job(
    job_id: str,
    *,
    content: str | None = None,
    num_gpus: int | None = None,
    resume_from: str | None = None,
    output_dir: str | None = None,
    extra_args: str | None = None,
    reset_dataloader: bool | None = None,
    reset_optimizer: bool | None = None,
    cache_only: bool | None = None,
    trust_cache: bool | None = None,
    regenerate_cache: bool | None = None,
) -> db.JobRecord:
    """Edit a saved (``new``) or pending run: config TOML, launch params, cache flags."""
    job = db.get_job(job_id)
    if job.state not in ("pending", "new"):
        raise ValueError("Only saved (new) or pending jobs can be edited")

    fields: dict[str, Any] = {}
    if num_gpus is not None:
        fields["num_gpus"] = num_gpus
    if resume_from is not None:
        fields["resume_from"] = resume_from or None
    if output_dir is not None:
        fields["output_dir"] = output_dir or None

    # Resolved cache state (current column value unless overridden).
    co = job.cache_only if cache_only is None else cache_only
    tc = job.trust_cache if trust_cache is None else trust_cache
    rc = job.regenerate_cache if regenerate_cache is None else regenerate_cache
    cache_changed = any(v is not None for v in (cache_only, trust_cache, regenerate_cache))
    if cache_changed:
        fields["cache_only"] = int(co)
        fields["trust_cache"] = int(tc)
        fields["regenerate_cache"] = int(rc)

    if content is not None:
        fields["config_content"] = content
        # Re-stage immediately only for pending runs (drafts stage on enqueue).
        if job.state == "pending":
            staging = run_staging.materialize_staging(content, job_id)
            fields["config_path"] = str(staging)
            cfg = toml.loads(staging.read_text(encoding="utf-8"))
            if output_dir is None and "output_dir" in cfg:
                fields.setdefault("output_dir", cfg.get("output_dir", "output"))

    # Recompute extra_args when the caller changed it, the reset toggles, or cache flags.
    if extra_args is not None or reset_dataloader is not None or reset_optimizer is not None or cache_changed:
        base = extra_args if extra_args is not None else job.extra_args
        tokens = shlex.split(base) if base else []
        if reset_dataloader is not None or reset_optimizer is not None:
            tokens = [t for t in tokens if t not in ("--reset_dataloader", "--reset_optimizer")]
            if reset_dataloader:
                tokens.append("--reset_dataloader")
            if reset_optimizer:
                tokens.append("--reset_optimizer")
        fields["extra_args"] = merge_job_cli_args(
            " ".join(tokens), cache_only=co, trust_cache=tc, regenerate_cache=rc
        )

    if fields:
        db.update_job(job_id, **fields)
    return db.get_job(job_id)


def delete_job_record(job_id: str) -> None:
    """Delete a run from the DB only (never touches files on disk).

    Allowed for saved (``new``), pending, and terminal runs. A run that is still
    running/stopping must be stopped first.
    """
    job = db.get_job(job_id)
    if job.state in ("running", "stopping"):
        raise ValueError("Stop the run before deleting it")
    db.delete_job(job_id)
    _normalize_queue_positions()


def reorder_queue(ordered_ids: list[int]) -> list[db.JobRecord]:
    """Set ``queue_position`` for pending jobs to match ``ordered_ids`` order.

    Non-pending or unknown ids are ignored; pending jobs missing from the list are
    appended after, preserving their previous relative order.
    """
    listed = [int(x) for x in ordered_ids]
    listed_set = set(listed)
    pos = 0
    for jid in listed:
        try:
            j = db.get_job(jid)
        except KeyError:
            continue
        if j.state != "pending":
            continue
        db.update_job(jid, queue_position=pos)
        pos += 1
    remaining = sorted(
        [
            j
            for j in db.list_jobs(limit=500)
            if j.state == "pending" and j.id not in listed_set
        ],
        key=lambda j: (
            j.queue_position if j.queue_position is not None else 999999,
            j.started_at or "",
        ),
    )
    for j in remaining:
        db.update_job(j.id, queue_position=pos)
        pos += 1
    return [j for j in list_jobs_sorted() if j.state == "pending"]


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
