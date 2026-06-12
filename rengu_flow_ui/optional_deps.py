"""Maintenance UI helpers for optional extras (prefer auto-install via training_extras)."""

from __future__ import annotations

from typing import Any

from rengu_flow.install import ensure_profiles, profile_installed
from rengu_flow.install.profiles import (
    PROFILE_DESCRIPTIONS,
    PROFILE_LABELS,
    rengu_init_command,
    uv_sync_argv,
)
from rengu_flow.install.profiles import normalize_profiles as normalize_install_profiles

OPTIONAL_PROFILE_IDS: tuple[str, ...] = ("cosmos_predict2", "lycoris", "optim", "prep")

_PROFILE_ID_TO_SYNC_KEY: dict[str, str] = {
    "cosmos_predict2": "cosmos",
    "lycoris": "lycoris",
    "optim": "optim",
}


def _sync_key(profile_id: str) -> str:
    return _PROFILE_ID_TO_SYNC_KEY.get(profile_id, profile_id)


def profile_installed_by_id(profile_id: str) -> bool:
    return profile_installed(_sync_key(profile_id))


def list_optional_profiles() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_id in OPTIONAL_PROFILE_IDS:
        sync_key = _sync_key(profile_id)
        rows.append(
            {
                "id": profile_id,
                "label": PROFILE_LABELS.get(sync_key, profile_id),
                "description": PROFILE_DESCRIPTIONS.get(sync_key, ""),
                "command": rengu_init_command([sync_key]),
                "installed": profile_installed_by_id(profile_id),
            }
        )
    return rows


def install_optional_profile(
    profile_id: str,
    *,
    execute: bool,
    confirm: bool,
) -> dict[str, Any]:
    if profile_id not in OPTIONAL_PROFILE_IDS:
        raise ValueError(
            f"Unknown profile {profile_id!r}; choose from: {', '.join(OPTIONAL_PROFILE_IDS)}"
        )
    sync_key = _sync_key(profile_id)
    argv = uv_sync_argv(normalize_install_profiles([sync_key]))
    command_str = " ".join(argv)
    if not execute:
        return {
            "ok": True,
            "executed": False,
            "profile": profile_id,
            "command": command_str,
            "installed": profile_installed_by_id(profile_id),
            "message": "Copy and run at repo root.",
        }
    if not confirm:
        raise ValueError("Pass confirm=true to run uv sync from the UI.")
    ensure_profiles([sync_key], reason=f"Maintenance install ({profile_id})")
    return {
        "ok": profile_installed_by_id(profile_id),
        "executed": True,
        "profile": profile_id,
        "command": command_str,
        "installed": profile_installed_by_id(profile_id),
        "message": "uv sync finished."
        if profile_installed_by_id(profile_id)
        else "uv sync finished but imports still missing.",
    }
