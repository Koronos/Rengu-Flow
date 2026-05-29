"""Filesystem stat for UI path validation (exists, file vs directory)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from rengu_flow_ui.settings import repo_root

ExpectKind = Literal["file", "dir"]


def _normalize_raw(path: str) -> str:
    return (path or "").strip()


def resolve_validated_path(path: str) -> Path:
    """Resolve a user path; block ``..`` and relative traversal outside repo root."""
    raw = _normalize_raw(path)
    if not raw:
        raise ValueError("Path is required")
    if "\0" in raw:
        raise ValueError("Invalid path")

    normalized = raw.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise ValueError("Path must not contain ..")

    p = Path(raw).expanduser()
    repo = repo_root().resolve()

    if p.is_absolute():
        return p.resolve()

    resolved = (repo / p).resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise ValueError("Path escapes project root") from exc
    return resolved


def stat_path(path: str, *, expect: ExpectKind | None = None) -> dict[str, Any]:
    """Return existence and type info for a filesystem path."""
    try:
        resolved = resolve_validated_path(path)
    except ValueError as exc:
        return {
            "exists": False,
            "is_file": False,
            "is_dir": False,
            "error": str(exc),
        }

    exists = resolved.exists()
    is_file = resolved.is_file() if exists else False
    is_dir = resolved.is_dir() if exists else False

    out: dict[str, Any] = {
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "resolved_path": str(resolved),
    }

    if not exists:
        out["error"] = "Path does not exist"
        return out

    if expect == "file" and not is_file:
        out["error"] = "Expected a file"
    elif expect == "dir" and not is_dir:
        out["error"] = "Expected a directory"

    return out
