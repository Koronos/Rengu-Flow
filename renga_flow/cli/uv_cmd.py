"""Run ``uv sync`` for init/update."""

from __future__ import annotations

import subprocess
from pathlib import Path

from renga_flow.config.local_config import repo_root
from renga_flow.install_profiles import normalize_profiles, uv_sync_argv


def run_uv_venv(root: Path | None = None) -> int:
    """Create the project virtual environment with ``uv venv``."""
    root = root or repo_root()
    venv = root / ".venv"
    if (venv / "bin" / "python").is_file():
        return 0
    cmd = ["uv", "venv", str(venv)]
    print(f"==> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(root))
    return proc.returncode


def run_uv_venv_or_exit(root: Path | None = None) -> None:
    code = run_uv_venv(root)
    if code != 0:
        raise SystemExit(code)


def run_uv_sync(profiles: list[str]) -> int:
    cmd = uv_sync_argv(profiles)
    print(f"==> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(repo_root()))
    return proc.returncode


def run_uv_sync_or_exit(profiles: list[str]) -> None:
    code = run_uv_sync(profiles)
    if code != 0:
        raise SystemExit(code)
