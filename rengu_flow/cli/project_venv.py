"""Project ``.venv`` via uv (no manual ``source .venv`` or system ``python3``)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rengu_flow.config.local_config import repo_root
from rengu_flow.cli.platform import require_uv
from rengu_flow.cli.uv_cmd import run_uv_sync_or_exit, run_uv_venv
from rengu_flow.install_profiles import normalize_profiles


def venv_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".venv"


def venv_python(root: Path | None = None) -> Path:
    return venv_dir(root) / "bin" / "python"


def venv_rengu(root: Path | None = None) -> Path:
    return venv_dir(root) / "bin" / "rengu"


def ensure_project_venv(root: Path | None = None) -> Path:
    """Create ``.venv`` with ``uv venv`` when missing; return ``.venv/bin/python``."""
    root = root or repo_root()
    py = venv_python(root)
    if py.is_file():
        return py
    require_uv()
    run_uv_venv(root)
    if not py.is_file():
        raise SystemExit("rengu: uv venv finished but .venv/bin/python is missing")
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
    """Run the same CLI from ``.venv/bin/rengu`` when the project venv exists."""
    argv = list(argv if argv is not None else sys.argv[1:])
    py = venv_python()
    rengu = venv_rengu()
    if not py.is_file():
        return
    if Path(sys.executable).resolve() == py.resolve():
        return
    if rengu.is_file():
        os.execv(str(rengu), [str(rengu), *argv])
    os.execv(str(py), [str(py), "-m", "rengu_flow.cli", *argv])
