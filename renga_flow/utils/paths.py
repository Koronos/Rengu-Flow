"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path


def path_is_under(path: str | Path, root: str | Path) -> bool:
    """True when ``path`` is ``root`` or a file/dir under ``root`` (resolved)."""
    try:
        p = Path(path).resolve()
        r = Path(root).resolve()
        return p == r or p.is_relative_to(r)
    except (ValueError, OSError):
        return False
