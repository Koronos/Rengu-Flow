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


class PathError(ValueError):
    """Raised when a user-supplied path is rejected as unsafe."""


def resolve_example_path(path: str | Path) -> Path:
    """Resolve *path* to an existing file under ``<repo>/examples/`` only.

    Sandboxes the import-example endpoints so a crafted ``path`` (``..`` or an
    absolute path) cannot read arbitrary files. Accepts paths with or without
    the leading ``examples/`` segment. Raises ``PathError`` for traversal or
    escape attempts and ``FileNotFoundError`` when the resolved file is missing.
    """
    normalized = str(path).replace("\\", "/").strip().lstrip("/")
    parts = [p for p in normalized.split("/") if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        raise PathError("Invalid example path")
    if parts[0] != "examples":
        parts = ["examples", *parts]
    examples_root = (settings.repo_root() / "examples").resolve()
    target = (settings.repo_root() / Path(*parts)).resolve()
    if not target.is_relative_to(examples_root):
        raise PathError("Path escapes examples directory")
    if not target.is_file():
        raise FileNotFoundError(normalized)
    return target
