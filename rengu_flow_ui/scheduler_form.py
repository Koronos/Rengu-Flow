"""LR scheduler section: KV parameters, form pruning, and TOML round-trip."""

from __future__ import annotations

import json
from typing import Any

from rengu_flow.optim.resolver import scheduler_registry
from rengu_flow_ui.optim_kv_defaults import scheduler_kv_defaults

# Flat form keys owned by the config schema (not merged from lr_scheduler_args.extra_params).
SCHEMA_SCHEDULER_PATHS: frozenset[str] = frozenset(
    {
        "lr_scheduler",
        "warmup_steps",
        "lr_scheduler_args.extra_params",
    }
)

KNOWN_BUILTIN_SCHEDULER_TYPES: frozenset[str] = frozenset(k.lower() for k in scheduler_registry)


def normalize_scheduler_type(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_custom_scheduler_type(value: Any) -> bool:
    """True for FQN paths and other non-registry scheduler names."""
    name = normalize_scheduler_type(value)
    if not name:
        return False
    if "." in name:
        return True
    return name.lower() not in KNOWN_BUILTIN_SCHEDULER_TYPES


def when_scheduler_selected() -> dict[str, Any]:
    return {"form_nonempty": "lr_scheduler"}


def visibility_for_scheduler_path(path: str) -> dict[str, Any] | None:
    """Return a visibility clause for a schema field path, or None for always visible."""
    if path in ("lr_scheduler_args.extra_params", "warmup_steps"):
        return when_scheduler_selected()
    return None


def paths_relevant_for_scheduler_type(sched_type: Any) -> frozenset[str]:
    """Flat form keys that should be kept for this lr_scheduler value."""
    name = normalize_scheduler_type(sched_type)
    keep: set[str] = {"lr_scheduler"}
    if name and name.lower() != "none":
        keep.add("lr_scheduler_args.extra_params")
        keep.add("warmup_steps")
    return frozenset(keep)


def prune_scheduler_form(form: dict[str, Any], sched_type: Any | None = None) -> dict[str, Any]:
    """Drop scheduler-related keys that do not apply to the current lr_scheduler."""
    sched_type = sched_type if sched_type is not None else form.get("lr_scheduler")
    allowed = paths_relevant_for_scheduler_type(sched_type)
    out = dict(form)
    for key in list(out):
        if key == "lr_scheduler" or key == "warmup_steps" or key.startswith("lr_scheduler_args."):
            if key not in allowed:
                out.pop(key, None)
    return out


def defaults_for_scheduler_type_change(sched_type: Any) -> dict[str, Any]:
    """Suggested KV defaults when the user picks a scheduler type."""
    name = normalize_scheduler_type(sched_type)
    if not name:
        return {}
    kv = scheduler_kv_defaults(name)
    return {"lr_scheduler_args.extra_params": kv}


def _extras_dict_from_form(raw: Any) -> dict[str, Any]:
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


def collect_scheduler_extra_params(form: dict[str, Any]) -> dict[str, Any]:
    """Gather scheduler args from flat form keys into the KV editor dict."""
    extras = _extras_dict_from_form(form.get("lr_scheduler_args.extra_params"))
    extras.pop("warmup_steps", None)
    for key, value in form.items():
        if not key.startswith("lr_scheduler_args."):
            continue
        sub = key.split(".", 1)[1]
        if key in SCHEMA_SCHEDULER_PATHS:
            continue
        extras[sub] = value
    return extras


def split_scheduler_extras(form: dict[str, Any]) -> dict[str, Any]:
    """Move scheduler parameters into lr_scheduler_args.extra_params for the form UI."""
    out = dict(form)
    if not normalize_scheduler_type(out.get("lr_scheduler")):
        return out
    raw_kv = _extras_dict_from_form(out.get("lr_scheduler_args.extra_params"))
    if "warmup_steps" in raw_kv and "warmup_steps" not in out:
        out["warmup_steps"] = raw_kv["warmup_steps"]
    extras = collect_scheduler_extra_params(out)
    for key in list(out):
        if key.startswith("lr_scheduler_args.") and key not in SCHEMA_SCHEDULER_PATHS:
            out.pop(key, None)
    out["lr_scheduler_args.extra_params"] = extras
    return out


def merge_scheduler_extras(form: dict[str, Any]) -> dict[str, Any]:
    """Expand lr_scheduler_args.extra_params back into flat keys before TOML render."""
    out = dict(form)
    raw = out.pop("lr_scheduler_args.extra_params", None)
    if raw is None or raw == "":
        return out
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return out
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            return out
    if not isinstance(raw, dict):
        return out
    for sub, value in raw.items():
        if value is None or value == "":
            continue
        if sub == "warmup_steps":
            out["warmup_steps"] = value
            continue
        out[f"lr_scheduler_args.{sub}"] = value
    return out


def attach_scheduler_visibility(fields: list[dict[str, Any]]) -> None:
    """Set ``visibility`` on scheduler section fields (idempotent)."""
    for field in fields:
        clause = visibility_for_scheduler_path(field["path"])
        if clause is None:
            continue
        existing = field.get("visibility")
        if existing:
            field["visibility"] = {"all": [existing, clause]}
        else:
            field["visibility"] = clause
