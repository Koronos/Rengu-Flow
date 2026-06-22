"""Shared helpers for optimizer and scheduler form modules."""

from __future__ import annotations

import json
from typing import Any


def extras_dict_from_form(raw: Any) -> dict[str, Any]:
    """Parse optimizer/scheduler extra_params from a form value (dict, JSON string, or empty)."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def normalize_param_type(value: Any) -> str:
    """Normalize an optimizer or scheduler type value to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def is_custom_param_type(value: Any, *, known_builtins: frozenset[str]) -> bool:
    """True when the value is an FQN path or a name not in the builtin registry set."""
    name = normalize_param_type(value)
    if not name:
        return False
    if "." in name:
        return True
    return name.lower() not in known_builtins
