"""Single source of truth for the version shown to users (CLI, API, UI).

The distribution version lives in pyproject (``[project].version``); we read it back via
``importlib.metadata`` so it is never duplicated in code. renga is updated by ``git pull`` (not
PyPI releases), so we also expose the short git SHA of the checkout to disambiguate builds that
share the same ``[project].version``.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
from shutil import which

DIST_NAME = "rengu-flow"


@lru_cache(maxsize=1)
def package_version() -> str:
    """The renga version, with pyproject ``[project].version`` as the single source of truth.

    Prefers installed distribution metadata (which is generated from pyproject); falls back to
    reading pyproject directly so a raw source checkout — not ``pip install``-ed — still reports a
    real version instead of a sentinel.
    """
    try:
        return _dist_version(DIST_NAME)
    except PackageNotFoundError:
        return _version_from_pyproject() or "0.0.0+unknown"


def _version_from_pyproject() -> str | None:
    """Read ``[project].version`` from pyproject (fallback for non-installed source checkouts)."""
    try:
        import toml

        from rengu_flow.config.local_config import repo_root

        data = toml.load(repo_root() / "pyproject.toml")
        value = data.get("project", {}).get("version")
        return str(value) if value else None
    except Exception:
        return None


def installed_version(dist: str) -> str | None:
    """Version of another installed distribution (e.g. ``kaon``), or None when absent."""
    try:
        return _dist_version(dist)
    except PackageNotFoundError:
        return None


def git_revision(root: Path | None = None) -> str | None:
    """Short git SHA of the renga checkout, or None when git/``.git`` is unavailable.

    Never raises: anything unexpected (no git, detached worktree, permission error) yields None
    so version reporting degrades gracefully instead of breaking the CLI/API.
    """
    try:
        from rengu_flow.config.local_config import repo_root

        base = Path(root) if root is not None else repo_root()
        if not which("git") or not (base / ".git").exists():
            return None
        proc = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        sha = proc.stdout.strip()
        return sha or None
    except Exception:
        return None


def version_string(root: Path | None = None) -> str:
    """One-liner, e.g. ``0.1.0 (a1b2c3d)`` — or just ``0.1.0`` outside a git checkout."""
    base = package_version()
    sha = git_revision(root)
    return f"{base} ({sha})" if sha else base


def version_info(root: Path | None = None) -> dict[str, str | None]:
    """Structured payload for the API/UI: renga version + git commit + installed kaon."""
    return {
        "version": package_version(),
        "commit": git_revision(root),
        "kaon": installed_version("kaon"),
    }
