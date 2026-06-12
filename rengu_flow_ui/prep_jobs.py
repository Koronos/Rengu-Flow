"""Create and enqueue dataset-prep jobs (kind='prep') on the shared job queue.

Prep jobs reuse the whole training-job lifecycle: same table, same single-runner
queue (so a prep job never shares the GPU with a training run), same log streaming,
graceful stop via signal files in the job's run_dir, and exit-code reconciliation
from the log. ``extra_args`` stores the stage name; ``config_content`` keeps the
staged prep TOML so the job is self-contained.
"""

from __future__ import annotations

from pathlib import Path

from rengu_flow.prep.config import STAGES
from rengu_flow_ui import db, job_queue
from rengu_flow_ui.settings import ensure_data_dirs, ui_data_dir


def prep_jobs_dir() -> Path:
    return ui_data_dir() / "prep"


def enqueue_prep_job(stage: str, config_toml: str, *, start_now: bool = False) -> db.JobRecord:
    if stage not in STAGES:
        raise ValueError(f"Unknown prep stage {stage!r}; expected one of {STAGES}")
    ensure_data_dirs()

    job = db.create_job(
        config_path="",  # filled below once the id exists
        log_path="",
        state="pending",
        extra_args=stage,
        queue_position=job_queue.next_queue_position(),
        config_content=config_toml,
        kind="prep",
    )
    job_dir = prep_jobs_dir() / str(job.id)
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "prep.toml"
    config_path.write_text(config_toml, encoding="utf-8")
    job = db.update_job(
        job.id,
        config_path=str(config_path),
        log_path=str(job_dir / "job.log"),
        run_dir=str(job_dir),
    )

    if start_now and not job_queue.has_active_runner():
        from rengu_flow_ui import jobs

        jobs.start_job(job)
        job = db.get_job(job.id)
    return job


def requeue_prep_job(job_id: str | int, *, start_now: bool = False) -> db.JobRecord:
    """Put a terminal prep job back on the queue (same record, same config).

    Re-running is also how a stopped job RESUMES: tag/caption skip images that
    already have their line written (unless the job was configured to overwrite),
    so only the remaining work runs.
    """
    job = db.get_job(job_id)
    if job.kind != "prep":
        raise ValueError("Only prep jobs can be re-queued here")
    if job.state not in ("stopped", "failed", "finished"):
        raise ValueError(f"Job is {job.state}; only stopped/failed/finished jobs can be re-queued")

    # Clear any leftover stop signals so the rerun doesn't exit immediately.
    if job.run_dir:
        from rengu_flow.utils.signal_files import SIGNAL_QUIT, SIGNAL_SAVE_QUIT

        for name in (SIGNAL_SAVE_QUIT, SIGNAL_QUIT):
            sig = Path(job.run_dir) / name
            try:
                sig.unlink(missing_ok=True)
            except OSError:
                pass

    job = db.update_job(
        job.id,
        state="pending",
        finished_at=None,
        exit_code=None,
        pid=None,
        queue_position=job_queue.next_queue_position(),
    )
    if start_now and not job_queue.has_active_runner():
        from rengu_flow_ui import jobs

        jobs.start_job(job)
        job = db.get_job(job.id)
    return job
