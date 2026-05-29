"""Serve repository markdown docs to the web UI (path-validated)."""

from __future__ import annotations

from pathlib import Path

from rengu_flow_ui.settings import repo_root


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
