"""Convert between TOML text and form-friendly dicts."""

from __future__ import annotations

import json
from typing import Any

import toml
import tomlkit

from rengu_flow.config import set_config_defaults
from rengu_flow.registry.model_capabilities import normalize_model_type
from rengu_flow_ui.adapter_form import prune_adapter_form
from rengu_flow_ui.field_visibility import field_visible, prune_form_for_model
from rengu_flow_ui.optimizer_form import (
    KNOWN_BUILTIN_OPTIMIZER_TYPES,
    merge_optimizer_extras,
    prune_optimizer_form,
    split_optimizer_extras,
)
from rengu_flow_ui.scheduler_form import (
    KNOWN_BUILTIN_SCHEDULER_TYPES,
    merge_scheduler_extras,
    prune_scheduler_form,
    split_scheduler_extras,
)


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


# Paths whose value IS a dict (e.g. a resolution->batch map): kept as a single
# form value instead of being flattened into dotted per-key paths, so the form
# widget receives the whole map.
_LEAF_DICT_PATHS = frozenset({"micro_batch_size_per_gpu", "image_micro_batch_size_per_gpu"})


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and path in _LEAF_DICT_PATHS:
            out[path] = {str(k): v for k, v in value.items()}
        elif isinstance(value, dict):
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
    # tomlkit is TOML-1.0 compliant, so it accepts mixed int/float arrays (e.g. a legacy
    # `betas = [0, 0.999]`) that the strict `toml` lib rejects outright. Parsing with it lets
    # such configs load into the form; re-rendering writes them back homogeneously via
    # `_homogenize_numeric_arrays`, keeping the on-disk TOML loadable by the strict trainer.
    config = tomlkit.loads(content).unwrap()
    config = _dtype_to_str(config)
    if "model" in config and "type" in config["model"]:
        config["model"]["type"] = normalize_model_type(config["model"]["type"]) or config["model"]["type"]
    form = flatten_config(config)
    form["_has_adapter"] = "adapter" in config and bool(config.get("adapter"))
    opt_type = form.get("optimizer.type")
    if isinstance(opt_type, str) and opt_type.lower() in KNOWN_BUILTIN_OPTIMIZER_TYPES:
        form["optimizer.type"] = opt_type.lower()
    sched_type = form.get("lr_scheduler")
    if isinstance(sched_type, str) and sched_type.lower() in KNOWN_BUILTIN_SCHEDULER_TYPES:
        form["lr_scheduler"] = sched_type.lower()
    return split_scheduler_extras(split_optimizer_extras(form))


def normalize_dataset_value(value: Any) -> Any:
    """Single path string when one entry; list when several (matches TOML conventions)."""
    if isinstance(value, list):
        paths = [x.strip() for x in value if isinstance(x, str) and x.strip()]
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0]
        return paths
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return value


def coerce_preview_prompts_for_toml(config: dict[str, Any]) -> None:
    """toml.dumps cannot encode a mixed list of strings and tables under preview.prompts."""
    preview = config.get("preview")
    if not isinstance(preview, dict):
        return
    prompts = preview.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        return
    if all(isinstance(p, str) for p in prompts):
        return
    out: list[Any] = []
    for item in prompts:
        if isinstance(item, str):
            out.append({"prompt": item})
        else:
            out.append(item)
    preview["prompts"] = out


def _homogenize_numeric_arrays(obj: Any) -> Any:
    """Promote mixed int/float arrays to all-float so ``toml.dumps`` stays homogeneous.

    The form round-trips through JSON, where ``0.0`` collapses to the integer ``0``; an
    array like ``[0.0, 0.999]`` (e.g. adakaon betas) then reaches us as ``[0, 0.999]``.
    ``toml`` writes that verbatim and refuses to load it back ("Not a homogeneous array").
    Pure-int arrays (resolutions, etc.) are left untouched; only arrays that already mix
    ints with floats are promoted.
    """
    if isinstance(obj, dict):
        return {k: _homogenize_numeric_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [_homogenize_numeric_arrays(v) for v in obj]
        numeric = [v for v in items if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeric) == len(items) and items and any(isinstance(v, float) for v in numeric):
            return [float(v) for v in items]
        return items
    return obj


def form_to_config(form: dict[str, Any]) -> dict[str, Any]:
    merged = merge_scheduler_extras(
        merge_optimizer_extras(prune_scheduler_form(prune_optimizer_form(prune_adapter_form(form))))
    )
    config = unflatten_form(merged)
    if "dataset" in config:
        normalized = normalize_dataset_value(config["dataset"])
        if normalized is None:
            config.pop("dataset", None)
        else:
            config["dataset"] = normalized
    coerce_preview_prompts_for_toml(config)
    config = _homogenize_numeric_arrays(config)
    return config


def merge_form_into_config(base: dict[str, Any], form: dict[str, Any]) -> dict[str, Any]:
    """Overlay flat form onto a nested config dict (e.g. library save)."""
    import copy

    merged = copy.deepcopy(base)
    updated = form_to_config(form)
    for key, value in updated.items():
        merged[key] = value
    if not form.get("_has_adapter"):
        merged.pop("adapter", None)
    return merged


def form_to_toml(form: dict[str, Any], base_content: str | None = None) -> str:
    merged = dict(form)
    if base_content and base_content.strip():
        base_form = parse_toml(base_content)
        for key, value in base_form.items():
            if key not in merged:
                merged[key] = value
    config = form_to_config(prune_form_for_model(merged))
    return toml.dumps(config)


def toml_to_form(content: str) -> dict[str, Any]:
    return parse_toml(content)


def form_values_for_ui(form: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Copy flat form dict and fill schema defaults for keys absent from TOML."""
    out = split_scheduler_extras(split_optimizer_extras(dict(form)))
    capabilities = schema.get("registries", {}).get("model_capabilities")
    for section in schema.get("sections", []):
        for field in section.get("fields", []):
            path = field["path"]
            if path not in out and "default" in field:
                if field_visible(field, out, capabilities):
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
