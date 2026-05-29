"""Optimizer section: KV parameters, form pruning, and TOML round-trip."""

from __future__ import annotations

import json
from typing import Any

from rengu_flow.registry.optimizers import (
    OPTIMIZER_ALIASES,
    VENDOR_OPTIMIZER_ALIASES,
    optimizer_registry,
)
from rengu_flow_ui.optim_kv_defaults import optimizer_extra_params_defaults

# Flat form keys owned by the config schema (not merged from optimizer.extra_params).
SCHEMA_OPTIMIZER_PATHS: frozenset[str] = frozenset(
    {
        "optimizer.type",
        "optimizer.extra_params",
    }
)

KNOWN_BUILTIN_OPTIMIZER_TYPES: frozenset[str] = frozenset(
    {k.lower() for k in optimizer_registry}
    | {k.lower() for k in OPTIMIZER_ALIASES}
    | {k.lower() for k in VENDOR_OPTIMIZER_ALIASES}
)

ADAM_LIKE_OPTIMIZER_TYPES: frozenset[str] = frozenset(
    {
        "adam",
        "adamw",
        "adamw8bit",
        "adamw_optimi",
        "stableadamw",
        "adamw8bitkahan",
        "genericoptim",
        "automagic",
        "offload",
        "prodigy",
    }
)


def normalize_optimizer_type(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_custom_optimizer_type(value: Any) -> bool:
    """True for FQN paths, pytorch_optimizer class names, and other non-registry types."""
    name = normalize_optimizer_type(value)
    if not name:
        return False
    if "." in name:
        return True
    return name.lower() not in KNOWN_BUILTIN_OPTIMIZER_TYPES


def when_optimizer_selected() -> dict[str, Any]:
    return {"form_nonempty": "optimizer.type"}


def visibility_for_optimizer_path(path: str) -> dict[str, Any] | None:
    """Return a visibility clause for a schema field path, or None for always visible."""
    if path == "optimizer.extra_params":
        return when_optimizer_selected()
    return None


def paths_relevant_for_optimizer_type(opt_type: Any) -> frozenset[str]:
    """Flat form keys that should be kept for this optimizer.type value."""
    name = normalize_optimizer_type(opt_type)
    keep: set[str] = {"optimizer.type"}
    if name:
        keep.add("optimizer.extra_params")
    return frozenset(keep)


def prune_optimizer_form(form: dict[str, Any], opt_type: Any | None = None) -> dict[str, Any]:
    """Drop optimizer.* keys that do not apply to the current optimizer.type."""
    opt_type = opt_type if opt_type is not None else form.get("optimizer.type")
    allowed = paths_relevant_for_optimizer_type(opt_type)
    out = dict(form)
    for key in list(out):
        if key.startswith("optimizer.") and key not in allowed:
            out.pop(key, None)
    return out


def defaults_for_optimizer_type_change(opt_type: Any) -> dict[str, Any]:
    """Suggested KV defaults when the user picks an optimizer type."""
    name = normalize_optimizer_type(opt_type)
    if not name:
        return {}
    if is_custom_optimizer_type(name):
        return {"optimizer.extra_params": {}}
    kv = optimizer_extra_params_defaults(name)
    return {"optimizer.extra_params": kv}


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


def collect_optimizer_extra_params(form: dict[str, Any]) -> dict[str, Any]:
    """Gather optimizer args from flat form keys into the KV editor dict."""
    extras = _extras_dict_from_form(form.get("optimizer.extra_params"))
    for key, value in form.items():
        if not key.startswith("optimizer."):
            continue
        sub = key.split(".", 1)[1]
        if key in SCHEMA_OPTIMIZER_PATHS:
            continue
        extras[sub] = value
    return extras


def split_optimizer_extras(form: dict[str, Any]) -> dict[str, Any]:
    """Move optimizer parameters into optimizer.extra_params for the form UI."""
    out = dict(form)
    if not normalize_optimizer_type(out.get("optimizer.type")):
        return out
    extras = collect_optimizer_extra_params(out)
    for key in list(out):
        if key.startswith("optimizer.") and key not in SCHEMA_OPTIMIZER_PATHS:
            out.pop(key, None)
    out["optimizer.extra_params"] = extras
    return out


def merge_optimizer_extras(form: dict[str, Any]) -> dict[str, Any]:
    """Expand optimizer.extra_params back into flat optimizer.* keys before TOML render."""
    out = dict(form)
    raw = out.pop("optimizer.extra_params", None)
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
        if sub == "type":
            continue
        if value is None or value == "":
            continue
        out[f"optimizer.{sub}"] = value
    return out


def _betas_list_from_optimizer(optimizer: dict[str, Any]) -> list[Any] | None:
    betas = optimizer.get("betas")
    if betas is None:
        return None
    if isinstance(betas, (list, tuple)):
        return list(betas)
    return None


def collect_optimizer_betas_validation_errors(config: dict[str, Any]) -> list[str]:
    """Adam-style optimizers require exactly two betas when the key is set."""
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, dict):
        return []
    opt_type = normalize_optimizer_type(optimizer.get("type"))
    if not opt_type or opt_type.lower() not in ADAM_LIKE_OPTIMIZER_TYPES:
        return []
    betas = _betas_list_from_optimizer(optimizer)
    if betas is None:
        return []
    if len(betas) != 2:
        return [
            "optimizer.betas must contain exactly two floats (beta1 and beta2) "
            f"for optimizer type {opt_type!r}; got {len(betas)} value(s)."
        ]
    for i, item in enumerate(betas):
        try:
            float(item)
        except (TypeError, ValueError):
            return [f"optimizer.betas[{i}] must be a number, got {item!r}."]
    return []


def attach_optimizer_visibility(fields: list[dict[str, Any]]) -> None:
    """Set ``visibility`` on optimizer section fields (idempotent)."""
    for field in fields:
        clause = visibility_for_optimizer_path(field["path"])
        if clause is None:
            continue
        existing = field.get("visibility")
        if existing:
            field["visibility"] = {"all": [existing, clause]}
        else:
            field["visibility"] = clause
