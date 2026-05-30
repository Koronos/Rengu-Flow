"""Project ``.venv`` via uv (no manual ``source .venv`` or system ``python3``)."""

from __future__ import annotations

import sys
from pathlib import Path

from rengu_flow.config.local_config import repo_root
from rengu_flow.cli.platform import require_uv
from rengu_flow.cli.uv_cmd import run_uv_sync_or_exit, run_uv_venv
from rengu_flow.install_profiles import normalize_profiles
from rengu_flow.platform_compat import reexec, venv_exe


def venv_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".venv"


def venv_python(root: Path | None = None) -> Path:
    return venv_exe(venv_dir(root), "python")


def venv_rengu(root: Path | None = None) -> Path:
    return venv_exe(venv_dir(root), "rengu")


def ensure_project_venv(root: Path | None = None) -> Path:
    """Create ``.venv`` with ``uv venv`` when missing; return the venv python interpreter."""
    root = root or repo_root()
    py = venv_python(root)
    if py.is_file():
        return py
    require_uv()
    run_uv_venv(root)
    if not py.is_file():
        raise SystemExit(f"rengu: uv venv finished but {py} is missing")
    return py


def sync_dependencies(profiles: list[str], *, root: Path | None = None) -> None:
    """``uv venv`` (if needed) + ``uv sync`` for the given install profiles."""
    require_uv()
    root = root or repo_root()
    normalized = normalize_profiles(profiles)
    if not venv_python(root).is_file():
        run_uv_venv(root)
    run_uv_sync_or_exit(normalized)


def ensure_ui_dependencies(*, root: Path | None = None) -> Path:
    """Guarantee ``.venv`` with ``[ui]`` extra installed."""
    root = root or repo_root()
    sync_dependencies(["ui"], root=root)
    return ensure_project_venv(root)


def reexec_cli(argv: list[str] | None = None) -> None:
    """Hand off to the project venv's ``rengu`` (or ``python -m``) when it exists.

    POSIX replaces the process (``execv``); Windows runs it as a child and exits with its
    code (see ``platform_compat.reexec``).
    """
    argv = list(argv if argv is not None else sys.argv[1:])
    py = venv_python()
    rengu = venv_rengu()
    if not py.is_file():
        return
    if Path(sys.executable).resolve() == py.resolve():
        return
    if rengu.is_file():
        reexec(rengu, argv)
    reexec(py, ["-m", "rengu_flow.cli", *argv])
