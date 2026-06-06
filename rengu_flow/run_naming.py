"""Training run folder names from optional ``run_name`` in config."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

RUN_NAME_MAX_LEN = 80
_INVALID_RUN_NAME_CHARS = re.compile(r"[/\\:\0]")
_SAFE_RUN_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def format_run_timestamp(when: datetime | None = None) -> str:
    """UTC timestamp used as the default run folder suffix."""
    dt = when or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d_%H-%M-%S")


def normalize_run_name(raw: Any) -> str | None:
    """Return a trimmed run name, or ``None`` when unset/blank."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    name = raw.strip()
    return name or None


def sanitize_run_name(name: str) -> str:
    """Make a user label safe for a single path segment (no slashes)."""
    base = _SAFE_RUN_NAME.sub("_", name.strip()).strip("._")
    if not base:
        return ""
    if len(base) > RUN_NAME_MAX_LEN:
        base = base[:RUN_NAME_MAX_LEN].rstrip("._")
    return base or ""


def build_run_folder_name(
    run_name: str | None,
    *,
    timestamp: str | None = None,
) -> str:
    """Folder name under ``output_dir`` for a new training run.

    With ``run_name``: ``{timestamp}_{sanitized_name}`` — date first so folders sort
    chronologically by name and are easy to locate by date.
    Without: ``{timestamp}`` only (date-only default).
    """
    ts = timestamp or format_run_timestamp()
    label = normalize_run_name(run_name)
    if not label:
        return ts
    safe = sanitize_run_name(label)
    if not safe:
        return ts
    return f"{ts}_{safe}"


def collect_run_name_validation_errors(config: dict[str, Any]) -> list[str]:
    """Validation issues for optional top-level ``run_name``."""
    if "run_name" not in config:
        return []
    raw = config["run_name"]
    if raw is None:
        return []
    if not isinstance(raw, str):
        return ["run_name must be a string when set."]
    name = raw.strip()
    if not name:
        return []
    if _INVALID_RUN_NAME_CHARS.search(name):
        return [
            "run_name must not contain /, \\, :, or null characters "
            "(used in output folder and TensorBoard)."
        ]
    if len(name) > RUN_NAME_MAX_LEN:
        return [f"run_name must be at most {RUN_NAME_MAX_LEN} characters."]
    safe = sanitize_run_name(name)
    if not safe:
        return [
            "run_name must contain at least one letter, digit, dot, underscore, or hyphen "
            "after removing unsafe characters."
        ]
    return []
