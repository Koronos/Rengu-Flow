"""Additive, on-demand dependency manager: probe imports, install only what's missing, self-heal."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import toml

from rengu_flow.config.local_config import repo_root
from rengu_flow.install.profiles import (
    PROFILE_GIT_REQUIREMENTS,
    PROFILE_IMPORT_CHECKS,
    normalize_profiles,
)
from rengu_flow.install.runner import require_uv, run_uv_pip_install_or_exit
from rengu_flow.install.state import read_installed_profiles, record_installed_profiles

# Optimizer types that need the ``[optim]`` extra (see docs/user/optimizer-and-scheduler.md).
_OPTIMIZER_EXTRA_TYPES = frozenset(
    {
        "genericoptim",
        "automagic",
        "adamw8bit",
        "adamw8bitkahan",
        "offload",
        "adamw_optimi",
        "stableadamw",
        "prodigy",
    }
)


def profile_installed(profile: str) -> bool:
    """True when all of the profile's required modules import (no checks => always True)."""
    modules = PROFILE_IMPORT_CHECKS.get(profile)
    if not modules:
        return True
    return all(importlib.util.find_spec(name) is not None for name in modules)


def missing_profiles(profiles: list[str]) -> list[str]:
    return [p for p in profiles if not profile_installed(p)]


def _record_satisfied(needed: list[str], root: Path) -> None:
    """Remember the requested profiles that are installed, so self_heal can restore them later."""
    record_installed_profiles(
        [p for p in needed if p != "base" and profile_installed(p)], root=root
    )


def ensure_profiles(
    profiles: list[str],
    *,
    root: Path | None = None,
    reason: str = "training",
) -> list[str]:
    """Install (additively) any requested profiles that are not importable yet.

    - Probes imports; if everything is present, installs nothing.
    - Otherwise runs an additive ``uv sync --inexact --extra …`` (never removes other packages),
      then ``uv pip install`` for any git/VCS requirements uv can't handle via extras.
    - Records the satisfied profiles for self-healing.

    Returns the list of profiles that were missing (and thus (re)installed). Raises ``SystemExit``
    when uv is missing or a profile is still unimportable after installing.
    """
    needed = normalize_profiles(profiles)
    root = root or repo_root()
    missing = missing_profiles(needed)
    if missing:
        require_uv()
        labels = ", ".join(missing)
        print(f"==> {reason} needs optional extras ({labels}); installing (additive)...")
        # Lazy import avoids an import cycle (project_venv imports from this package's runner).
        from rengu_flow.cli.project_venv import sync_dependencies

        sync_dependencies(missing, root=root)

        for profile in missing:
            specs = PROFILE_GIT_REQUIREMENTS.get(profile)
            if specs and not profile_installed(profile):
                run_uv_pip_install_or_exit(specs, root=root)

        still_missing = missing_profiles(needed)
        if still_missing:
            raise SystemExit(
                "rengu: required optional dependencies are still missing after install: "
                + ", ".join(still_missing)
                + f". Try manually: rengu init {' '.join(still_missing)}"
            )
        print(f"==> Optional extras ready: {labels}")

    _record_satisfied(needed, root)
    return missing


def self_heal(*, root: Path | None = None, reason: str = "Restoring recorded extras") -> list[str]:
    """Re-ensure every previously-installed profile (additive). Recovers from an external exact sync."""
    profiles = read_installed_profiles(root)
    if not profiles:
        return []
    return ensure_profiles(profiles, root=root, reason=reason)


def ensure_ui_dependencies(*, root: Path | None = None) -> Path:
    """Guarantee the ``[ui]`` extra (additive) and record it for self-healing."""
    from rengu_flow.cli.project_venv import ensure_ui_dependencies as _ensure_ui

    py = _ensure_ui(root=root)
    record_installed_profiles(["ui"], root=root)
    return py


def profiles_for_config_dict(data: dict[str, Any]) -> list[str]:
    """Return install profile names needed for this training config."""
    profiles: list[str] = []
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    mtype = str(model.get("type", "")).strip().lower()
    if mtype == "cosmos_predict2":
        profiles.append("cosmos")

    adapter = data.get("adapter") if isinstance(data.get("adapter"), dict) else {}
    if str(adapter.get("type", "")).strip().lower() == "lokr":
        profiles.append("lycoris")

    optim = data.get("optimizer") if isinstance(data.get("optimizer"), dict) else {}
    otype = str(optim.get("type", "")).strip().lower()
    if otype in _OPTIMIZER_EXTRA_TYPES:
        profiles.append("optim")

    out: list[str] = []
    for p in profiles:
        if p not in out:
            out.append(p)
    return out


def profiles_for_config_path(config_path: Path) -> list[str]:
    data = toml.load(config_path)
    if not isinstance(data, dict):
        return []
    return profiles_for_config_dict(data)


def ensure_training_extras(
    config_path: Path | str,
    *,
    root: Path | None = None,
) -> list[str]:
    """Install (additively) any optional extras required by ``config_path`` that are missing.

    Returns the list of profiles that were installed (empty if nothing was missing).
    Raises ``SystemExit`` when ``uv`` is missing or install still fails.
    """
    path = Path(config_path)
    needed = profiles_for_config_path(path)
    if not needed:
        return []
    return ensure_profiles(needed, root=root, reason="Training config")
