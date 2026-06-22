"""Maintenance gate: always returns False (maintenance API is force-disabled)."""

from __future__ import annotations


def maintenance_enabled() -> bool:
    """Always returns False — destructive maintenance API is intentionally disabled.

    The git-submodule/pip/db-reset machinery has been removed. Set ``_MAINTENANCE_DISABLED``
    back to False and restore the full module if you need to re-enable it.
    """
    return False
