"""Launch and manage DeepSpeed training subprocesses."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from renga_flow_ui import db
from renga_flow_ui.settings import logs_dir, repo_root


def _which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def build_train_command(
    config_path: Path,
    *,
    num_gpus: int = 1,
    resume_from: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    extra = extra_args or []
    deepspeed = _which("deepspeed")
    if deepspeed:
        cmd = [deepspeed, f"--num_gpus={num_gpus}", "-m", "renga_flow.main", "--config", str(config_path)]
    else:
        cmd = [sys.executable, "-m", "renga_flow.main", "--config", str(config_path)]
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
    extra = shlex.split(job.extra_args) if job.extra_args else []
    cmd = build_train_command(
        Path(job.config_path),
        num_gpus=job.num_gpus,
        resume_from=job.resume_from,
        extra_args=extra,
    )
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    with open(log_path, "ab") as log_f:
        log_f.write(f"\n--- renga-flow-ui job {job.id} ---\n".encode())
        log_f.write(f"CWD: {repo_root()}\n".encode())
        log_f.write(f"CMD: {shlex.join(cmd)}\n\n".encode())
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root()),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=run_env,
            start_new_session=True,
        )
    db.update_job(job.id, state="running", pid=proc.pid)
    return proc.pid


def stop_job(job_id: str, *, graceful_signal: bool = True) -> None:
    job = db.get_job(job_id)
    if job.pid is None:
        db.update_job(job_id, state="stopped", finished_at=_now())
        return
    try:
        os.killpg(os.getpgid(job.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(job.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if graceful_signal and job.run_dir:
        from renga_flow.utils.signal_files import SIGNAL_SAVE_QUIT

        sig = Path(job.run_dir) / SIGNAL_SAVE_QUIT
        try:
            sig.touch()
        except OSError:
            pass
    db.update_job(job_id, state="stopping")


def poll_job(job_id: str) -> db.JobRecord:
    job = db.get_job(job_id)
    if job.state not in ("running", "stopping", "pending") or job.pid is None:
        return job
    try:
        os.kill(job.pid, 0)
    except ProcessLookupError:
        exit_code = _read_exit_code(job)
        db.update_job(
            job_id,
            state="finished",
            finished_at=_now(),
            exit_code=exit_code,
            pid=None,
        )
        return db.get_job(job_id)
    return job


def _read_exit_code(job: db.JobRecord) -> int | None:
    return None


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def refresh_all_jobs() -> None:
    for job in db.list_jobs():
        if job.state in ("running", "stopping"):
            poll_job(job.id)
    from renga_flow_ui.job_queue import try_start_next

    try_start_next()


def tail_log(job_id: str, offset: int = 0) -> tuple[str, int]:
    job = db.get_job(job_id)
    path = Path(job.log_path)
    if not path.is_file():
        return "", 0
    data = path.read_bytes()
    if offset > len(data):
        offset = len(data)
    return data[offset:].decode("utf-8", errors="replace"), len(data)
