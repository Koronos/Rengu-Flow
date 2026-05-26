"""Convert between TOML text and form-friendly dicts."""

from __future__ import annotations

import json
from typing import Any

import toml

from renga_flow.config import set_config_defaults
from renga_flow.registry.model_capabilities import normalize_model_type
from renga_flow_ui.field_visibility import prune_form_for_model


def _get_nested(data: dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    cur: Any = data
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = data
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    if value is None or value == "":
        cur.pop(parts[-1], None)
    else:
        cur[parts[-1]] = value


def _dtype_to_str(obj: Any) -> Any:
    if hasattr(obj, "__class__") and obj.__class__.__name__ in (
        "dtype",
        "bfloat16",
        "float16",
        "float32",
    ):
        name = str(obj).replace("torch.", "")
        for key in ("bfloat16", "float16", "float32", "float8_e4m3fn", "float8_e5m2"):
            if key in name:
                return key
        return name
    if isinstance(obj, dict):
        return {k: _dtype_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dtype_to_str(v) for v in obj]
    return obj


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_config(value, path))
        else:
            out[path] = value
    return out


def unflatten_form(form: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    skip = {"_has_adapter"}
    for key, value in form.items():
        if key in skip:
            continue
        if value is None or value == "":
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        if "." in key:
            _set_nested(config, key, _parse_json_value(value))
        else:
            config[key] = _parse_json_value(value)
    if not form.get("_has_adapter"):
        config.pop("adapter", None)
    return config


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return value
    return value


def parse_toml(content: str) -> dict[str, Any]:
    config = toml.loads(content)
    config = _dtype_to_str(config)
    if "model" in config and "type" in config["model"]:
        config["model"]["type"] = normalize_model_type(config["model"]["type"]) or config["model"]["type"]
    form = flatten_config(config)
    form["_has_adapter"] = "adapter" in config and bool(config.get("adapter"))
    return form


def form_to_config(form: dict[str, Any]) -> dict[str, Any]:
    return unflatten_form(form)


def form_to_toml(form: dict[str, Any]) -> str:
    config = form_to_config(prune_form_for_model(form))
    return toml.dumps(config)


def toml_to_form(content: str) -> dict[str, Any]:
    return parse_toml(content)


def form_values_for_ui(form: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Copy flat form dict and fill schema defaults for keys absent from TOML."""
    out = dict(form)
    for section in schema.get("sections", []):
        for field in section.get("fields", []):
            path = field["path"]
            if path not in out and "default" in field:
                out[path] = field["default"]
    if "_has_adapter" not in out:
        out["_has_adapter"] = bool(form.get("adapter"))
    return out


def apply_defaults_preview(config: dict[str, Any]) -> dict[str, Any]:
    """Apply framework defaults for display (dtypes become strings)."""
    import copy

    cfg = copy.deepcopy(config)
    try:
        set_config_defaults(cfg)
    except Exception:
        pass
    return _dtype_to_str(cfg)
