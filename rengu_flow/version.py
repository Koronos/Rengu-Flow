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

# The beta channel is a git branch, not a version string: being on this branch is what marks a
# checkout as beta, so merging develop -> main flips the channel without editing [project].version.
BETA_BRANCH = "develop"


@lru_cache(maxsize=1)
def package_version() -> str:
    """The renga version, read from pyproject ``[project].version`` (the single source of truth).

    renga is deployed as a git checkout updated in place by ``rengu update`` (``git pull``), so the
    just-pulled ``pyproject.toml`` is authoritative. We read it **before** the installed distribution
    metadata because an editable install's recorded version lags behind a pull until the package is
    reinstalled — and ``uv sync`` skips that reinstall whenever it sees no resolution change (e.g. a
    version-only bump), leaving the UI showing a stale version. A plain wheel install has no
    pyproject beside the package, so there we fall back to the installed distribution metadata.
    """
    from_pyproject = _version_from_pyproject()
    if from_pyproject:
        return from_pyproject
    try:
        return _dist_version(DIST_NAME)
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _version_from_pyproject() -> str | None:
    """Read ``[project].version`` from the source checkout's pyproject (None for a wheel install)."""
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


def git_branch(root: Path | None = None) -> str | None:
    """Current git branch of the renga checkout, or None when unavailable/detached.

    Never raises: anything unexpected (no git, no ``.git``, detached HEAD, permission error)
    yields None so version/channel reporting degrades gracefully instead of breaking.
    """
    try:
        from rengu_flow.config.local_config import repo_root

        base = Path(root) if root is not None else repo_root()
        if not which("git") or not (base / ".git").exists():
            return None
        proc = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
        )
        branch = proc.stdout.strip()
        if proc.returncode != 0 or not branch or branch == "HEAD":
            return None
        return branch
    except Exception:
        return None


def is_beta(root: Path | None = None) -> bool:
    """True when the checkout is on the beta channel — the ``develop`` branch.

    Derived from the branch, not ``[project].version``, so the channel flips on merge to ``main``
    without a version edit.
    """
    return git_branch(root) == BETA_BRANCH


def version_string(root: Path | None = None) -> str:
    """One-liner, e.g. ``0.1.0 (a1b2c3d)`` — ``-beta`` suffix on the develop channel."""
    base = package_version()
    if is_beta(root):
        base = f"{base}-beta"
    sha = git_revision(root)
    return f"{base} ({sha})" if sha else base


def version_info(root: Path | None = None) -> dict[str, str | bool | None]:
    """Structured payload for the API/UI: version + commit + branch/beta channel + kaon."""
    branch = git_branch(root)
    return {
        "version": package_version(),
        "commit": git_revision(root),
        "branch": branch,
        "beta": branch == BETA_BRANCH,
        "kaon": installed_version("kaon"),
    }
