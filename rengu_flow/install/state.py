"""Persisted record of which install profiles have been enabled.

Stored as JSON at ``<repo>/.rengu-flow/installed-profiles.json`` (gitignored) — deliberately
outside both the venv (so it survives a venv wipe/recreate) and ``rengu.local.toml`` (which is
user-edited and has no writer). Used to self-heal the environment after an external exact sync.
"""

from __future__ import annotations

import json
from pathlib import Path

from rengu_flow.config.local_config import repo_root

STATE_DIRNAME = ".rengu-flow"
INSTALLED_PROFILES_FILE = "installed-profiles.json"


def state_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / STATE_DIRNAME


def installed_profiles_path(root: Path | None = None) -> Path:
    return state_dir(root) / INSTALLED_PROFILES_FILE


def read_installed_profiles(root: Path | None = None) -> list[str]:
    """Return the recorded profile names (empty list when nothing recorded / unreadable)."""
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
    return [str(p) for p in profiles if isinstance(p, str) and p.strip()]


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
