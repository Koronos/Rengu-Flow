"""Shared UTC timestamp helper."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
