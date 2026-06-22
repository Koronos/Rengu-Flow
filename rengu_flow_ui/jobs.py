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


def build_prep_command(config_path: Path, *, stage: str, job_dir: Path) -> list[str]:
    """Argv for a dataset-prep job: the same `rengu prep <stage>` the CLI runs.

    The prep CLI installs its own extras on demand (uv sync --extra prep), so no
    ensure_training_extras here. ``--job-dir`` points signals + report.json at the
    job's own folder.
    """
    import sys

    return [
        sys.executable,
        "-m",
        "rengu_flow.cli",
        "prep",
        stage,
        "--config",
        str(config_path),
        "--job-dir",
        str(job_dir),
    ]


def start_job(
    job: db.JobRecord,
    *,
    env: dict[str, str] | None = None,
) -> int:
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    from rengu_flow_ui import settings

    if job.kind == "prep":
        # extra_args holds the stage name; run_dir was pre-set to the job folder.
        cmd = build_prep_command(
            Path(job.config_path),
            stage=(job.extra_args or "tag").strip(),
            job_dir=Path(job.run_dir or log_path.parent),
        )
    else:
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
    rd = _parse_run_dir_from_log(job) if job.kind != "prep" else None
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
    # Auto-advance the queue on a natural end (finished or failed), regardless of which UI
    # page polled this job. A user stop/quit lands in "stopped" (force-stop, or a *_quit
    # signal that flipped the job to "stopping") and intentionally does NOT advance — the
    # queue only halts on an explicit user quit.
    if final_state in ("finished", "failed"):
        from rengu_flow_ui.job_queue import try_start_next

        try_start_next()
    return db.get_job(job_id)


# Cap how much of a (possibly tens-of-MB) log we read for run-dir / exit-code parsing. The
# trainer prints "Run dir:" once near a run's start and the exit markers near its end, so the
# most recent 256 KB reliably contains whichever we're after on a ~1 s poll while bounding the
# per-tick cost — reading the whole growing file every poll is what pinned the GIL and starved
# the dashboard. Mirrors read_raw_log_tail's tail-read approach.
_PARSE_TAIL_BYTES = 262_144


def _decode_log(data: bytes) -> str:
    """Decode log bytes as UTF-8 and normalize CRLF -> LF: on Windows the training subprocess
    writes ``\\r\\n`` line endings, but progress-marker parsing and the UI expect ``\\n``."""
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def _read_log_text(job: db.JobRecord, *, tail_bytes: int | None = None) -> str:
    path = Path(job.log_path)
    if not path.is_file():
        return ""
    if tail_bytes is not None and path.stat().st_size > tail_bytes:
        with path.open("rb") as f:
            f.seek(-tail_bytes, 2)
            data = f.read()
        return _decode_log(data)
    return _decode_log(path.read_bytes())


def _current_run_log(job: db.JobRecord, *, tail_bytes: int | None = None) -> str:
    """Log text for this job's MOST RECENT run only.

    The log file is appended across runs (same job id -> same log_path), each run prefixed with a
    ``--- rengu-flow-ui job <id> ---`` header. Scoping to the last segment keeps exit-code and
    run-dir parsing from picking up a PREVIOUS run's error/Run-dir (which marked a clean run failed
    and pinned a stale run folder). With ``tail_bytes`` only the most recent slice is read: when
    the marker isn't in it the run has already produced more than that, so the whole slice is the
    current run anyway.
    """
    text = _read_log_text(job, tail_bytes=tail_bytes)
    marker = f"--- rengu-flow-ui job {job.id} ---"
    idx = text.rfind(marker)
    return text[idx:] if idx != -1 else text


_RUN_DIR_RE = re.compile(r"^Run dir:\s*(.+?)\s*$", re.MULTILINE)


def _parse_run_dir_from_log(job: db.JobRecord) -> str | None:
    """The trainer prints `Run dir: <path>` (relative to the repo root) on rank 0."""
    m = _RUN_DIR_RE.search(_current_run_log(job, tail_bytes=_PARSE_TAIL_BYTES))
    if not m:
        return None
    from rengu_flow_ui import settings

    p = Path(m.group(1))
    if not p.is_absolute():
        p = settings.repo_root() / p
    return str(p.resolve()) if p.is_dir() else None


def _read_exit_code(job: db.JobRecord) -> int | None:
    """Best-effort exit code parsed from the job log (the process is detached, no wait())."""
    text = _current_run_log(job, tail_bytes=_PARSE_TAIL_BYTES)
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
    from rengu_flow_ui._time import now_utc_iso

    return now_utc_iso()


def refresh_all_jobs() -> None:
    # Poll active runs. poll_job advances the queue itself when a run ends naturally
    # (finished/failed), so a bare refresh that finds nothing active never starts an idle
    # queue — the first run is always started explicitly by the user (Start / Run now).
    for job in db.list_jobs():
        if job.state in ("running", "stopping"):
            poll_job(job.id)


def read_raw_log(job_id: str) -> str:
    """Full job log text WITHOUT marker stripping (for progress-marker parsing)."""
    job = db.get_job(job_id)
    path = Path(job.log_path)
    if not path.is_file():
        return ""
    return _decode_log(path.read_bytes())


def read_raw_log_tail(job_id: str, tail_bytes: int = 65536) -> str:
    """Read only the last ``tail_bytes`` of the raw log (for progress-marker parsing).

    Much cheaper than ``read_raw_log`` for long-running jobs whose logs grow to tens of
    MB. The progress marker is emitted at most ~1/s and is <200 bytes, so 64 KB always
    contains the most recent one.
    """
    job = db.get_job(job_id)
    path = Path(job.log_path)
    if not path.is_file():
        return ""
    size = path.stat().st_size
    if size <= tail_bytes:
        return _decode_log(path.read_bytes())
    with path.open("rb") as f:
        f.seek(-tail_bytes, 2)
        data = f.read()
    return _decode_log(data)


def tail_log(job_id: str, offset: int = 0) -> tuple[str, int]:
    job = db.get_job(job_id)
    path = Path(job.log_path)
    if not path.is_file():
        return "", 0
    # Read only the bytes appended since `offset` instead of slurping the whole (growing,
    # multi-MB) file on every poll/WS tick — the previous read_bytes() made each tail cost
    # O(filesize), which pinned the GIL under the log-poll flood and stalled other endpoints.
    if offset > path.stat().st_size:
        # File shrank below the saved offset (rotated/truncated/reset) — restart from the top.
        offset = 0
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
        new_offset = f.tell()
    text = _decode_log(data)
    # Filter throttled progress markers out of the displayed log; the UI parses them
    # separately for its live bar (see live_stream / progress_stream).
    return strip_progress_markers(text), new_offset


# WebSocket log streaming bounds. A multi-MB log dumped in one WS frame trips the 1 MB default
# frame limit (close code 1009 "message too big"), which silently drops the client back to HTTP
# polling. So: seek the initial catch-up to the recent tail (the UI only retains ~512 KB anyway),
# and split every send into frames that stay under the limit.
LOG_WS_TAIL_BYTES = 512 * 1024
LOG_WS_FRAME_BYTES = 256 * 1024


def log_tail_start_offset(job_id: str, tail_bytes: int = LOG_WS_TAIL_BYTES) -> int:
    """Byte offset ``tail_bytes`` before EOF (clamped to 0) for seeking a WS stream to the tail.

    Returns 0 for an unknown/missing job so the caller's normal ``tail_log`` path surfaces the
    "job not found" case instead of raising here.
    """
    try:
        job = db.get_job(job_id)
    except KeyError:
        return 0
    path = Path(job.log_path)
    if not path.is_file():
        return 0
    return max(0, path.stat().st_size - tail_bytes)


def iter_log_frames(text: str, limit_bytes: int = LOG_WS_FRAME_BYTES) -> list[str]:
    """Split ``text`` into pieces whose UTF-8 size stays under ``limit_bytes`` (the WS frame cap).

    The char budget is ``limit_bytes // 4`` so even all-4-byte-UTF-8 text never exceeds the limit;
    plain ASCII logs simply yield a few more, smaller frames. Empty text yields no frames.
    """
    if not text:
        return []
    step = max(1, limit_bytes // 4)
    return [text[i : i + step] for i in range(0, len(text), step)]
