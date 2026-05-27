"""Start/stop TensorBoard via venv binary or ``uv run --no-project --with tensorboard``."""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path
from threading import Lock
from typing import Any

from renga_flow_ui import settings
from renga_flow_ui.paths import resolve_repo_path
from renga_flow_ui.settings import logs_dir
from renga_flow_ui.subprocess_util import popen_repo_subprocess

_lock = Lock()
_proc: subprocess.Popen[bytes] | None = None
_log_path: Path | None = None
_meta: dict[str, Any] = {}

_TB_WITH = "tensorboard>=2.14"


def resolve_output_dir(output_dir: str) -> Path:
    """TensorBoard log root (parent of run folders)."""
    return resolve_repo_path(output_dir)


def build_tensorboard_cmd(logdir: Path, host: str, port: int) -> list[str]:
    """Match ``scripts/tensorboard.sh``: prefer ``.venv/bin/tensorboard``, else ``uv run``."""
    venv_tb = settings.repo_root() / ".venv" / "bin" / "tensorboard"
    args = [f"--logdir={logdir}", f"--host={host}", f"--port={port}"]
    if venv_tb.is_file():
        return [str(venv_tb), *args]
    uv = shutil.which("uv")
    if not uv:
        raise FileNotFoundError(
            "uv is not on PATH and .venv/bin/tensorboard is missing. "
            "Install uv (https://docs.astral.sh/uv/) or run: pip install -e '.[ui]'"
        )
    return [uv, "run", "--no-project", "--with", _TB_WITH, "tensorboard", *args]


def pick_free_port(host: str = "127.0.0.1", start: int = 6006, attempts: int = 100) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port in range {start}-{start + attempts - 1}")


def _is_running() -> bool:
    global _proc
    if _proc is None:
        return False
    if _proc.poll() is not None:
        _proc = None
        return False
    return True


def _status_unlocked() -> dict[str, Any]:
    running = _is_running()
    return {
        "running": running,
        "url": _meta.get("url"),
        "port": _meta.get("port"),
        "logdir": _meta.get("logdir"),
        "pid": _proc.pid if running and _proc else None,
        "log_path": str(_log_path) if _log_path else None,
    }


def tensorboard_status() -> dict[str, Any]:
    with _lock:
        return _status_unlocked()


def _stop_unlocked() -> dict[str, Any]:
    global _proc, _meta
    if not _is_running():
        _proc = None
        _meta = {}
        return {"running": False, "stopped": False}
    assert _proc is not None
    _proc.terminate()
    try:
        _proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _proc.kill()
        _proc.wait(timeout=3)
    _proc = None
    _meta = {}
    return {"running": False, "stopped": True}


def stop_tensorboard() -> dict[str, Any]:
    with _lock:
        return _stop_unlocked()


def _wait_for_http(
    proc: subprocess.Popen[bytes],
    host: str,
    port: int,
    log_path: Path,
    log_handle,
) -> None:
    for _ in range(50):
        time.sleep(0.1)
        if proc.poll() is not None:
            log_handle.close()
            err = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"TensorBoard exited immediately:\n{err}")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            continue
    log_handle.close()
    raise RuntimeError("TensorBoard did not open its HTTP port in time; see tensorboard.log")


def start_tensorboard(
    output_dir: str = "output",
    *,
    port: int | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    """Launch TensorBoard on ``output_dir`` (parent folder — run names appear in the sidebar)."""
    global _proc, _log_path, _meta

    logdir = resolve_output_dir(output_dir)
    if not logdir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {logdir}")

    bind_host = host or "127.0.0.1"
    with _lock:
        if (
            _is_running()
            and _meta.get("logdir") == str(logdir)
            and (host is None or _meta.get("host") == bind_host)
            and (port is None or _meta.get("port") == port)
        ):
            return {**_status_unlocked(), "reused": True}
        _stop_unlocked()
        bind_port = port or pick_free_port(bind_host)
        cmd = build_tensorboard_cmd(logdir, bind_host, bind_port)
        logs_dir().mkdir(parents=True, exist_ok=True)
        _log_path = logs_dir() / "tensorboard.log"
        _proc, log_handle = popen_repo_subprocess(cmd, _log_path, log_mode="w")
        _meta = {
            "url": f"http://{bind_host}:{bind_port}/",
            "port": bind_port,
            "logdir": str(logdir),
            "host": bind_host,
        }
        proc = _proc
        log_path = _log_path

    try:
        _wait_for_http(proc, bind_host, bind_port, log_path, log_handle)
    except RuntimeError:
        with _lock:
            _stop_unlocked()
        raise

    with _lock:
        return {**_status_unlocked(), "reused": False}
