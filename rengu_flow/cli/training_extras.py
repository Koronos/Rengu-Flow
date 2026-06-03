"""Backwards-compatible shim. Logic now lives in ``rengu_flow.install.manager``."""

from __future__ import annotations

from rengu_flow.install.manager import (
    ensure_profiles,
    ensure_training_extras,
    missing_profiles,
    profile_installed,
    profiles_for_config_dict,
    profiles_for_config_path,
)

__all__ = [
    "ensure_profiles",
    "ensure_training_extras",
    "missing_profiles",
    "profile_installed",
    "profiles_for_config_dict",
    "profiles_for_config_path",
]
