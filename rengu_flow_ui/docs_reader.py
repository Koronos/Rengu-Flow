"""Serve repository markdown docs to the web UI (path-validated)."""

from __future__ import annotations

import re
from pathlib import Path

from rengu_flow_ui.settings import repo_root

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class DocNotFoundError(FileNotFoundError):
    pass


class DocPathError(ValueError):
    pass


def resolve_doc_path(rel_path: str, repo: Path | None = None) -> Path:
    """Resolve a doc path under ``<repo>/docs/`` only.

    Security: anything that looks like an escape attempt — null bytes, absolute paths
    (POSIX or Windows), ``..`` traversal, paths outside ``docs/``, or symlinks resolving
    outside ``docs/`` — is rejected as ``DocNotFoundError`` (do not reveal why). Only a
    wrong (non-``.md``) extension raises ``DocPathError``.
    """
    repo = (repo or repo_root()).resolve()
    docs_root = (repo / "docs").resolve()

    if "\x00" in rel_path:
        raise DocNotFoundError(rel_path)
    normalized = rel_path.replace("\\", "/").strip()
    if normalized.startswith("/") or _WINDOWS_DRIVE_RE.match(normalized):
        raise DocNotFoundError(normalized)  # absolute path
    parts = [p for p in normalized.split("/") if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        raise DocNotFoundError(normalized)  # traversal
    if not normalized.endswith(".md"):
        raise DocPathError("Only .md files under docs/ are allowed")
    if parts[0] != "docs":
        if parts[0] in ("user", "developer"):
            parts = ["docs", *parts]
        else:
            raise DocNotFoundError(normalized)  # outside docs/

    target = (repo / Path(*parts)).resolve()
    try:
        target.relative_to(docs_root)
    except ValueError:
        raise DocNotFoundError(normalized) from None  # escapes docs/ (incl. via symlink)
    if not target.is_file():
        raise DocNotFoundError(normalized)
    return target


def _title_from_markdown(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback.replace("-", " ").title()


def list_docs_index() -> list[dict[str, str]]:
    """Return user-facing documentation index entries."""
    repo = repo_root().resolve()
    items: list[dict[str, str]] = []
    user_dir = repo / "docs" / "user"
    if user_dir.is_dir():
        for path in sorted(user_dir.glob("*.md")):
            rel = str(path.relative_to(repo)).replace("\\", "/")
            content = path.read_text(encoding="utf-8")
            items.append({"path": rel, "title": _title_from_markdown(content, path.stem)})
    readme = repo / "docs" / "README.md"
    if readme.is_file():
        rel = str(readme.relative_to(repo)).replace("\\", "/")
        content = readme.read_text(encoding="utf-8")
        items.insert(0, {"path": rel, "title": _title_from_markdown(content, "Documentation")})
    return items


def read_doc(rel_path: str) -> dict[str, str]:
    path = resolve_doc_path(rel_path)
    return {
        "path": str(path.relative_to(repo_root().resolve())).replace("\\", "/"),
        "title": path.stem.replace("-", " ").title(),
        "content": path.read_text(encoding="utf-8"),
    }
