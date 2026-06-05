"""``rengu update`` — pull latest code from the project repo, ``uv sync``, recompile the UI."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from shutil import which

from rengu_flow.config.local_config import repo_root
from rengu_flow.cli.project_venv import reexec_cli, sync_dependencies
from rengu_flow.install.profiles import PROFILE_EXTRAS, normalize_profiles
from rengu_flow.install.state import read_installed_profiles, record_installed_profiles

# Canonical upstream the CLI updates from (fast-forward only — never rewrites local history).
REPO_URL = "https://github.com/Koronos/Rengu-Flow"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "update",
        help="Pull latest code from the project repo, then re-sync dependencies with uv sync",
    )
    p.add_argument(
        "profiles",
        nargs="*",
        default=["base"],
        help="Profiles: base, ui, cosmos, optim, lycoris, dev, koptim, all",
    )
    p.add_argument(
        "--all-extras",
        action="store_true",
        help="Sync with all documented optional extras",
    )
    p.add_argument(
        "--no-pull",
        action="store_true",
        help=f"Skip the git fast-forward pull from {REPO_URL}; only re-sync dependencies",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Discard local *tracked* code changes and hard-reset to the latest upstream when a "
            "plain fast-forward is blocked (e.g. line-ending noise). Never deletes untracked or "
            "ignored files, so the UI data dir and jobs.db are left untouched."
        ),
    )


def _current_branch(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    branch = proc.stdout.strip()
    if proc.returncode != 0 or not branch or branch == "HEAD":
        return None
    return branch


def git_pull(root: Path | None = None, *, force: bool = False) -> bool:
    """Update the current branch from ``REPO_URL``.

    Returns True when the working tree is up to date afterwards, False when the update was
    skipped or could not complete.

    Default (``force=False``) is non-destructive: it fast-forwards only, so a divergent branch
    or a dirty tree leaves the checkout untouched and prints how to resolve it.

    With ``force=True`` it stashes local *tracked* changes and hard-resets the branch to the
    fetched upstream tip, recovering from line-ending noise or stray edits that block a plain
    fast-forward. It runs ``git reset --hard`` only and never ``git clean``, so untracked and
    ignored files — the UI data dir and ``jobs.db`` — are never deleted.
    """
    root = root or repo_root()
    if not which("git"):
        print("==> git not found on PATH; skipping code update (re-syncing dependencies only).")
        return False
    if not (root / ".git").exists():
        print(f"==> {root} is not a git checkout; skipping code update (dependencies only).")
        return False
    branch = _current_branch(root)
    if branch is None:
        print("==> Detached HEAD or unknown branch; skipping code update (dependencies only).")
        return False

    if force:
        return _git_force_pull(root, branch)

    cmd = ["git", "-C", str(root), "pull", "--ff-only", REPO_URL, branch]
    print(f"==> {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(
            "==> Could not fast-forward "
            f"'{branch}' from {REPO_URL}. Your branch has local commits or uncommitted "
            "changes that would be overwritten. If you did not edit the code yourself (this is "
            "usually line-ending noise), re-run `rengu update --force` to discard the local "
            "tracked changes and reset to upstream. --force never touches untracked or ignored "
            "files, so your data dir and jobs.db are safe. Continuing with dependency sync."
        )
        return False
    return True


def _git_force_pull(root: Path, branch: str) -> bool:
    """Hard-reset ``branch`` to the fetched upstream tip, preserving untracked files.

    Local tracked changes are first stashed (recoverable via ``git stash list``) so nothing is
    silently lost, then the branch is reset to ``FETCH_HEAD``. ``git clean`` is deliberately
    never run: untracked/ignored paths such as the UI ``data/`` dir and ``jobs.db`` survive.
    """
    fetch = ["git", "-C", str(root), "fetch", REPO_URL, branch]
    print(f"==> {' '.join(fetch)}")
    if subprocess.run(fetch).returncode != 0:
        print(f"==> git fetch from {REPO_URL} failed. Continuing with dependency sync.")
        return False

    # Stash only tracked changes (no -u/-a) so untracked data/ and jobs.db are never touched.
    if _has_tracked_changes(root):
        stash = [
            "git", "-C", str(root), "stash", "push",
            "--message", "rengu update --force (auto-stash of local tracked changes)",
        ]
        print(f"==> {' '.join(stash)}")
        if subprocess.run(stash).returncode == 0:
            print(
                "==> Local tracked changes stashed; recover them later with `git stash pop` "
                "if they were intentional."
            )
        else:
            print("==> Could not stash local changes; aborting force update to avoid data loss.")
            return False

    reset = ["git", "-C", str(root), "reset", "--hard", "FETCH_HEAD"]
    print(f"==> {' '.join(reset)}")
    if subprocess.run(reset).returncode != 0:
        print("==> git reset --hard failed. Continuing with dependency sync.")
        return False
    print(f"==> Reset '{branch}' to the latest upstream. Untracked files (data/, jobs.db) kept.")
    return True


def _has_tracked_changes(root: Path) -> bool:
    """True when the working tree or index has uncommitted changes to *tracked* files."""
    proc = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())


def rebuild_web(profiles: list[str], *, root: Path | None = None) -> None:
    """Recompile the web UI so a pulled frontend change is actually served.

    Only rebuilds when the UI matters here: the ``dist`` was built before (the user runs the
    UI) or the ``ui`` profile was requested. A missing Node toolchain is a warning, not a hard
    failure — updating code and Python deps must still succeed on training-only machines.
    """
    root = root or repo_root()
    from rengu_flow.cli.ui_cmd import _build_web, _web_dir

    dist = _web_dir(root) / "dist" / "index.html"
    wants_ui = "ui" in profiles
    if not dist.is_file() and not wants_ui:
        return
    print("==> Recompiling web UI...")
    try:
        _build_web(root, force=True)
        print(f"==> Web UI rebuilt: {_web_dir(root) / 'dist'}")
    except SystemExit as e:
        print(f"==> Skipped UI rebuild: {e}. Run `rengu ui build` once Node.js is available.")
    except subprocess.CalledProcessError as e:
        print(f"==> UI rebuild failed ({e}). Run `rengu ui build` to retry.")


def run(args: argparse.Namespace) -> None:
    root = repo_root()
    if not getattr(args, "no_pull", False):
        git_pull(root, force=getattr(args, "force", False))

    if args.all_extras:
        requested = normalize_profiles(["all"])
    else:
        requested = normalize_profiles(list(args.profiles))

    # Also refresh the optional profiles the user already set up, so a plain `rengu update` keeps
    # them current without re-listing them — and, for git-pinned extras like koptim, applies a
    # bumped commit pin from the freshly pulled pyproject. Profiles that were never installed are
    # left out entirely (uv --inexact never touches them), so an update never pulls in, say,
    # K-Optimizers for someone who never enabled it. ``read_installed_profiles`` is filtered to
    # known names so a stale record can't break normalize_profiles.
    previously_installed = [p for p in read_installed_profiles(root) if p in PROFILE_EXTRAS]
    sync_set = normalize_profiles([*requested, *previously_installed])

    sync_dependencies(sync_set, root=root)
    record_installed_profiles([p for p in sync_set if p != "base"], root=root)
    rebuild_web(sync_set, root=root)
    reexec_cli()
