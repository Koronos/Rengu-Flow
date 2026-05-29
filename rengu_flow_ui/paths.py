"""Path helpers for the UI control plane (repo-relative resolution)."""

from __future__ import annotations

from pathlib import Path

from rengu_flow_ui import settings


def resolve_repo_path(path: str | Path) -> Path:
    """Expand ``~``, join relative paths under ``repo_root()``, and normalize."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = settings.repo_root() / p
    return p.resolve()
