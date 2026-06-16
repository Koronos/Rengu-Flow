"""Persisted record of which install profiles have been enabled.

Stored as JSON at ``<repo>/data/installed-profiles.json`` (git-ignored) — in the **visible**
``data/`` folder so users can see and delete it, and deliberately outside both the venv (so it
survives a venv wipe/recreate) and ``rengu.local.toml`` (user-edited, no writer). Used to self-heal
the environment after an external exact sync. Earlier versions kept it in a hidden ``.rengu-flow/``
folder; ``_migrate_legacy_install_state`` moves that into ``data/`` on first access.

It lives in ``data/`` (info worth keeping so a venv recreate restores your extras), not ``cache/``
(which is meant to be wiped freely): losing this record silently drops your optional extras on the
next self-heal until you re-run ``rengu init <profiles>``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rengu_flow.config.local_config import repo_root

STATE_DIRNAME = "data"
INSTALLED_PROFILES_FILE = "installed-profiles.json"
# Retired hidden folder that held this record before it moved into the visible data/ dir.
LEGACY_STATE_DIRNAME = ".rengu-flow"


def state_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / STATE_DIRNAME


def installed_profiles_path(root: Path | None = None) -> Path:
    return state_dir(root) / INSTALLED_PROFILES_FILE


def _migrate_legacy_install_state(root: Path | None = None) -> None:
    """Move ``installed-profiles.json`` out of the retired hidden ``.rengu-flow/`` into ``data/``.

    Best-effort and adopt-only: it never overwrites a record already in ``data/`` and never raises,
    so a migration hiccup can't break install/update. The emptied legacy folder is removed.
    """
    r = root or repo_root()
    new = installed_profiles_path(r)
    if new.exists():
        return
    old = r / LEGACY_STATE_DIRNAME / INSTALLED_PROFILES_FILE
    if not old.is_file():
        return
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        legacy_dir = r / LEGACY_STATE_DIRNAME
        if legacy_dir.is_dir() and not any(legacy_dir.iterdir()):
            legacy_dir.rmdir()
    except OSError:
        pass


def read_installed_profiles(root: Path | None = None) -> list[str]:
    """Return the recorded profile names (empty list when nothing recorded / unreadable).

    The state file outlives the set of known profiles, so we drop any recorded name that is no
    longer a valid profile. Profiles are independent libraries — a removed or renamed one (e.g. a
    dropped ``koptim`` package) is simply forgotten, never collided with or migrated. This keeps
    ``self_heal`` from raising on a stale record without making ``normalize_profiles`` (user CLI
    input) tolerate typos. To keep a profile after a library swap, re-enable it (``rengu init <p>``).
    """
    from rengu_flow.install.profiles import PROFILE_EXTRAS

    _migrate_legacy_install_state(root)
    path = installed_profiles_path(root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    profiles = data.get("profiles") if isinstance(data, dict) else data
    if not isinstance(profiles, list):
        return []
    out: list[str] = []
    for p in profiles:
        if isinstance(p, str) and p.strip() in PROFILE_EXTRAS and p.strip() not in out:
            out.append(p.strip())
    return out


def record_installed_profiles(profiles: list[str], *, root: Path | None = None) -> list[str]:
    """Merge ``profiles`` into the recorded set (additive, order-preserving). Returns the new set."""
    merged = read_installed_profiles(root)
    changed = False
    for p in profiles:
        key = p.strip()
        if key and key not in merged:
            merged.append(key)
            changed = True
    if changed:
        path = installed_profiles_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"profiles": merged}, indent=2) + "\n", encoding="utf-8"
        )
    return merged
