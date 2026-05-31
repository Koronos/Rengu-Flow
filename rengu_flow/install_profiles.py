"""Backwards-compatible shim. Profile logic now lives in ``rengu_flow.install.profiles``."""

from __future__ import annotations

from rengu_flow.install.profiles import (
    ALL_PROFILE_NAMES,
    PROFILE_DESCRIPTIONS,
    PROFILE_EXTRAS,
    PROFILE_GIT_REQUIREMENTS,
    PROFILE_IMPORT_CHECKS,
    PROFILE_LABELS,
    normalize_profiles,
    profile_metadata,
    rengu_init_command,
    uv_sync_argv,
)

__all__ = [
    "ALL_PROFILE_NAMES",
    "PROFILE_DESCRIPTIONS",
    "PROFILE_EXTRAS",
    "PROFILE_GIT_REQUIREMENTS",
    "PROFILE_IMPORT_CHECKS",
    "PROFILE_LABELS",
    "normalize_profiles",
    "profile_metadata",
    "rengu_init_command",
    "uv_sync_argv",
]
