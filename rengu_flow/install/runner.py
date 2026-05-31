"""Thin wrappers around ``uv`` subprocess invocations (venv, sync, pip install)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rengu_flow.config.local_config import repo_root
from rengu_flow.install.profiles import uv_sync_argv


def require_uv() -> None:
    """Re-exported from the platform guard so the installer is the one stop for uv access."""
    from rengu_flow.cli.platform import require_uv as _require_uv

    _require_uv()


def run_uv_venv(root: Path | None = None) -> int:
    """Create the project virtual environment with ``uv venv``."""
    from rengu_flow.platform_compat import venv_exe

    root = root or repo_root()
    venv = root / ".venv"
    if venv_exe(venv, "python").is_file():
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


def run_uv_pip_install(specs: list[str], *, root: Path | None = None) -> int:
    """Additively install pip/git requirement specs into the project venv.

    Used for packages uv cannot manage via pyproject extras (e.g. ``git+https://…``). This never
    removes anything; it only adds the requested specs.
    """
    if not specs:
        return 0
    root = root or repo_root()
    cmd = ["uv", "pip", "install", *specs]
    print(f"==> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(root))
    return proc.returncode


def run_uv_pip_install_or_exit(specs: list[str], *, root: Path | None = None) -> None:
    code = run_uv_pip_install(specs, root=root)
    if code != 0:
        raise SystemExit(code)
