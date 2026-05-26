"""Serve repository markdown docs to the web UI (path-validated)."""

from __future__ import annotations

from pathlib import Path

from renga_flow_ui.settings import repo_root


class DocNotFoundError(FileNotFoundError):
    pass


class DocPathError(ValueError):
    pass


def resolve_doc_path(rel_path: str) -> Path:
    """Resolve a doc path under ``<repo>/docs/`` only."""
    normalized = rel_path.replace("\\", "/").strip().lstrip("/")
    if not normalized.endswith(".md"):
        raise DocPathError("Only .md files under docs/ are allowed")
    parts = [p for p in normalized.split("/") if p]
    if not parts or any(p == ".." for p in parts):
        raise DocPathError("Invalid documentation path")

    if parts[0] != "docs":
        if parts[0] in ("user", "developer"):
            parts = ["docs", *parts]
        else:
            raise DocPathError("Path must be under docs/")

    repo = repo_root().resolve()
    docs_root = (repo / "docs").resolve()
    target = (repo / Path(*parts)).resolve()
    if not str(target).startswith(str(docs_root)):
        raise DocPathError("Path escapes docs directory")
    if not target.is_file():
        raise DocNotFoundError(normalized)
    return target


def read_doc(rel_path: str) -> dict[str, str]:
    path = resolve_doc_path(rel_path)
    return {
        "path": str(path.relative_to(repo_root().resolve())).replace("\\", "/"),
        "title": path.stem.replace("-", " ").title(),
        "content": path.read_text(encoding="utf-8"),
    }
