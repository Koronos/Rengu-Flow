"""``rengu update`` — pull latest code from the project repo, ``uv sync``, recompile the UI."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from shutil import which

from rengu_flow.config.local_config import repo_root
from rengu_flow.cli.project_venv import reexec_cli, sync_dependencies
from rengu_flow.install.profiles import normalize_profiles
from rengu_flow.install.state import record_installed_profiles

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
        help="Profiles: base, ui, cosmos, optim, lycoris, dev, all",
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


def git_pull(root: Path | None = None) -> bool:
    """Fast-forward the current branch from ``REPO_URL`` (non-destructive).

    Returns True when the working tree is up to date afterwards, False when the pull was
    skipped or could not fast-forward. Never rewrites local history: a divergent branch or a
    dirty tree leaves the checkout untouched and prints how to resolve it.
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

    cmd = ["git", "-C", str(root), "pull", "--ff-only", REPO_URL, branch]
    print(f"==> {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(
            "==> Could not fast-forward "
            f"'{branch}' from {REPO_URL}. Your branch has local commits or uncommitted "
            "changes that would be overwritten. Resolve them (commit/stash, then "
            "`git pull --ff-only`) and re-run `rengu update`. Continuing with dependency sync."
        )
        return False
    return True


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
    if not getattr(args, "no_pull", False):
        git_pull()

    if args.all_extras:
        profiles = ["all"]
    else:
        profiles = normalize_profiles(list(args.profiles))
    normalized = normalize_profiles(profiles)
    sync_dependencies(profiles)
    record_installed_profiles([p for p in normalized if p != "base"])
    rebuild_web(normalized)
    reexec_cli()
