"""Launch and manage DeepSpeed training subprocesses."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from rengu_flow.cli.train_launcher import (
    _pick_master_port,
    base_train_command,
    training_subprocess_env,
)
from rengu_flow.cli.training_extras import ensure_training_extras
from rengu_flow.control.progress_stream import strip_progress_markers
from rengu_flow.platform_compat import pid_alive, terminate_process_tree
from rengu_flow_ui import db
from rengu_flow_ui.subprocess_util import popen_repo_subprocess


def build_train_command(
    config_path: Path,
    *,
    num_gpus: int = 1,
    resume_from: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    # Free --master_port so successive jobs don't collide on the default.
    extra = extra_args or []
    cmd = base_train_command(config_path, num_gpus=num_gpus, master_port=_pick_master_port(0))
    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])
    cmd.extend(extra)
    return cmd


def start_job(
    job: db.JobRecord,
    *,
    env: dict[str, str] | None = None,
) -> int:
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    from rengu_flow_ui import settings

    ensure_training_extras(Path(job.config_path), root=settings.repo_root())
    extra = shlex.split(job.extra_args) if job.extra_args else []
    cmd = build_train_command(
        Path(job.config_path),
        num_gpus=job.num_gpus,
        resume_from=job.resume_from,
        extra_args=extra,
    )
    # Apply [training.env] from rengu.local.toml (PYTORCH_CUDA_ALLOC_CONF, NCCL_*, TF32, ...) the
    # same way the `rengu train` CLI does — otherwise UI-launched jobs silently ignore it and only
    # inherit the UI process environment. respect_existing keeps any var already exported to the UI.
    run_env = training_subprocess_env()
    if env:
        run_env.update(env)
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    header = (
        f"\n--- rengu-flow-ui job {job.id} ---\n"
        f"CWD: {settings.repo_root()}\n"
        f"CMD: {shlex.join(cmd)}\n\n"
    ).encode()
    proc, _log_f = popen_repo_subprocess(
        cmd,
        log_path,
        log_header=header,
        env=run_env,
    )
    db.update_job(job.id, state="running", pid=proc.pid)
    return proc.pid


def stop_job(job_id: str, *, graceful_signal: bool = True) -> None:
    job = db.get_job(job_id)
    if job.pid is None:
        db.update_job(job_id, state="stopped", finished_at=_now())
        return
    # Graceful first: drop the save_quit signal file so training saves and exits cleanly
    # (on Windows there is no real SIGTERM, so this is the primary stop mechanism).
    if graceful_signal and job.run_dir:
        from rengu_flow.utils.signal_files import SIGNAL_SAVE_QUIT

        sig = Path(job.run_dir) / SIGNAL_SAVE_QUIT
        try:
            sig.touch()
        except OSError:
            pass
    # Hard stop fallback: terminate the process tree (children-first, SIGTERM->kill), which
    # replaces the POSIX-only killpg/getpgid path and works on Windows via psutil.
    terminate_process_tree(job.pid)
    db.update_job(job_id, state="stopping")


def poll_job(job_id: str) -> db.JobRecord:
    job = db.get_job(job_id)
    if job.state not in ("running", "stopping"):
        return job
    # Trust the trainer's own `Run dir:` (from THIS run's log segment) as authoritative: it fixes a
    # stale run_dir (e.g. a from-scratch continue whose record still pointed at the source folder),
    # which otherwise sent signals to the wrong folder and showed the wrong progress/previews.
    rd = _parse_run_dir_from_log(job)
    if rd and rd != job.run_dir:
        db.update_job(job_id, run_dir=rd)
        job = db.get_job(job_id)
    if job.pid is not None and pid_alive(job.pid):
        return job
    # PID gone: reconcile. stopping -> stopped; else nonzero exit -> failed, 0/unknown -> finished.
    exit_code = _read_exit_code(job)
    if job.state == "stopping":
        final_state = "stopped"
    elif exit_code not in (0, None):
        final_state = "failed"
    else:
        final_state = "finished"
    db.update_job(
        job_id,
        state=final_state,
        finished_at=_now(),
        exit_code=exit_code,
        pid=None,
    )
    return db.get_job(job_id)


def _read_log_text(job: db.JobRecord) -> str:
    path = Path(job.log_path)
    if not path.is_file():
        return ""
    return path.read_bytes().decode("utf-8", errors="replace")


def _current_run_log(job: db.JobRecord) -> str:
    """Log text for this job's MOST RECENT run only.

    The log file is appended across runs (same job id -> same log_path), each run prefixed with a
    ``--- rengu-flow-ui job <id> ---`` header. Scoping to the last segment keeps exit-code and
    run-dir parsing from picking up a PREVIOUS run's error/Run-dir (which marked a clean run failed
    and pinned a stale run folder).
    """
    text = _read_log_text(job)
    marker = f"--- rengu-flow-ui job {job.id} ---"
    idx = text.rfind(marker)
    return text[idx:] if idx != -1 else text


_RUN_DIR_RE = re.compile(r"^Run dir:\s*(.+?)\s*$", re.MULTILINE)


def _parse_run_dir_from_log(job: db.JobRecord) -> str | None:
    """The trainer prints `Run dir: <path>` (relative to the repo root) on rank 0."""
    m = _RUN_DIR_RE.search(_current_run_log(job))
    if not m:
        return None
    from rengu_flow_ui import settings

    p = Path(m.group(1))
    if not p.is_absolute():
        p = settings.repo_root() / p
    return str(p.resolve()) if p.is_dir() else None


def _read_exit_code(job: db.JobRecord) -> int | None:
    """Best-effort exit code parsed from the job log (the process is detached, no wait())."""
    text = _current_run_log(job)
    if not text:
        return None
    codes = re.findall(r"exits with return code\s*=\s*(-?\d+)", text)
    if codes:
        return int(codes[-1])
    if "exits successfully" in text:
        return 0
    if "Traceback (most recent call last)" in text or re.search(r"^\S*: error:", text, re.MULTILINE):
        return 1
    return None


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def refresh_all_jobs() -> None:
    # Poll active runs; note whether any just exited so we can advance the queue. We must NOT
    # start an idle queue from a bare refresh — the first run is started explicitly by the user
    # (Start / Run now). The queue only auto-drains as a consequence of a run finishing.
    finished_any = False
    for job in db.list_jobs():
        if job.state in ("running", "stopping"):
            updated = poll_job(job.id)
            if updated.state not in ("running", "stopping"):
                finished_any = True
    if finished_any:
        from rengu_flow_ui.job_queue import try_start_next

        try_start_next()


def read_raw_log(job_id: str) -> str:
    """Full job log text WITHOUT marker stripping (for progress-marker parsing)."""
    job = db.get_job(job_id)
    path = Path(job.log_path)
    if not path.is_file():
        return ""
    return path.read_bytes().decode("utf-8", errors="replace")


def tail_log(job_id: str, offset: int = 0) -> tuple[str, int]:
    job = db.get_job(job_id)
    path = Path(job.log_path)
    if not path.is_file():
        return "", 0
    data = path.read_bytes()
    if offset > len(data):
        offset = len(data)
    text = data[offset:].decode("utf-8", errors="replace")
    # Filter throttled progress markers out of the displayed log; the UI parses them
    # separately for its live bar (see live_stream / progress_stream).
    return strip_progress_markers(text), len(data)
