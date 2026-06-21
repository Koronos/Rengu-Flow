"""Backwards-compatible shim. Logic now lives in ``rengu_flow.install.manager``."""

from __future__ import annotations

from rengu_flow.install.manager import (
    ensure_profiles,
    ensure_training_extras,
)

__all__ = [
    "ensure_profiles",
    "ensure_training_extras",
]
