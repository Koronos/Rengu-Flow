"""Shared subprocess helpers for the UI control plane."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import IO, Any

from rengu_flow.platform_compat import popen_kwargs_new_group
from rengu_flow_ui import settings


def popen_repo_subprocess(
    cmd: list[str],
    log_path: Path,
    *,
    log_mode: str = "ab",
    log_header: bytes | None = None,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[Any], IO[Any]]:
    """Run ``cmd`` under ``repo_root()`` with stdout/stderr appended to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    binary = "b" in log_mode
    log_f = log_path.open(log_mode, encoding=None if binary else "utf-8")
    if log_header:
        if binary:
            log_f.write(log_header)
        else:
            log_f.write(log_header.decode("utf-8", errors="replace"))
        log_f.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(settings.repo_root()),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        **popen_kwargs_new_group(),
    )
    return proc, log_f
