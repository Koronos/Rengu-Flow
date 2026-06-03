"""Centralized dependency installer for rengu-flow.

All optional-dependency logic lives here: profile definitions, the uv command runners,
persisted install state, and the additive on-demand manager. CLI/UI modules delegate to this
package (some keep thin re-export shims for backwards-compatible import paths).

Design rules:
- Never destructive: every ``uv sync`` is run with ``--inexact`` so packages outside the
  selected resolution (other extras, user-installed custom optimizers/schedulers, git packages)
  are preserved.
- On demand: install a profile only when its modules are not importable yet.
- Self-healing: successfully installed profiles are recorded; ``self_heal`` re-ensures them so an
  external exact sync (e.g. ``uv run`` syncing to base) can be recovered from.
"""

from __future__ import annotations

from rengu_flow.install.manager import (
    ensure_profiles,
    ensure_training_extras,
    ensure_ui_dependencies,
    missing_profiles,
    profile_installed,
    profiles_for_config_dict,
    profiles_for_config_path,
    self_heal,
)
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
    "ensure_profiles",
    "ensure_training_extras",
    "ensure_ui_dependencies",
    "missing_profiles",
    "normalize_profiles",
    "profile_installed",
    "profile_metadata",
    "profiles_for_config_dict",
    "profiles_for_config_path",
    "rengu_init_command",
    "self_heal",
    "uv_sync_argv",
]
