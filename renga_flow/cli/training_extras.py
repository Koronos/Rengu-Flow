"""Detect and install optional deps required by a training config (via uv sync)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import toml

from renga_flow.cli.platform import require_uv
from renga_flow.cli.project_venv import sync_dependencies
from renga_flow.config.local_config import repo_root
from renga_flow.install_profiles import normalize_profiles

# Profile name (``renga init``) -> modules that must import after sync.
_PROFILE_IMPORT_CHECKS: dict[str, tuple[str, ...]] = {
    "cosmos": ("transformers", "einops"),
    "lycoris": ("lycoris",),
    "optim": ("bitsandbytes",),
}

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
    }
)


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


def profile_installed(profile: str) -> bool:
    modules = _PROFILE_IMPORT_CHECKS.get(profile)
    if not modules:
        return True
    return all(importlib.util.find_spec(name) is not None for name in modules)


def missing_profiles(profiles: list[str]) -> list[str]:
    return [p for p in profiles if not profile_installed(p)]


def ensure_profiles(
    profiles: list[str],
    *,
    root: Path | None = None,
    reason: str = "training",
) -> list[str]:
    """``uv sync`` any listed profiles that are not importable yet."""
    needed = normalize_profiles(profiles)
    missing = missing_profiles(needed)
    if not missing:
        return []

    require_uv()
    root = root or repo_root()
    labels = ", ".join(missing)
    print(f"==> {reason} needs optional extras ({labels}); running uv sync...")
    sync_dependencies(missing, root=root)

    still_missing = missing_profiles(needed)
    if still_missing:
        raise SystemExit(
            "renga: required optional dependencies are still missing after uv sync: "
            + ", ".join(still_missing)
            + f". Try manually: renga init {' '.join(still_missing)}"
        )
    print(f"==> Optional extras ready: {labels}")
    return missing


def ensure_training_extras(
    config_path: Path | str,
    *,
    root: Path | None = None,
) -> list[str]:
    """
    ``uv sync`` any optional extras required by ``config_path`` that are not importable yet.

    Returns the list of profiles that were installed (empty if nothing was missing).
    Raises ``SystemExit`` when ``uv`` is missing or sync/install still fails.
    """
    path = Path(config_path)
    needed = profiles_for_config_path(path)
    if not needed:
        return []
    return ensure_profiles(needed, root=root, reason="Training config")