"""Model-specific config rules shared by TOML validation and the web UI.

Single source for *what each model needs* under ``[model]``:

- **Required keys** — from ``ModelCapability.model_fields`` where ``required: true``
  (and optional ``model_validation`` overrides).
- **One-of groups** — e.g. ``llm_path`` or ``t5_path`` for Cosmos (TOML-only expert path).
- **Feature-gated training keys** — e.g. ``blocks_to_swap`` only when ``features.block_swap``.

The UI hides irrelevant fields via ``rengu_flow_ui/field_visibility.py``; this module
enforces the same intent at validate time so CLI and UI stay aligned.

To extend a model, edit its ``ModelCapability`` in ``model_capabilities.py`` — avoid
duplicating checks in ``validation.py``.
"""

from __future__ import annotations

from typing import Any

from rengu_flow.registry.model_capabilities import (
    ModelCapability,
    get_canonical_model_types,
    get_capability,
    normalize_model_type,
)


def _validation_error() -> type[Exception]:
    from rengu_flow.config.validation import ConfigValidationError

    return ConfigValidationError

# Training keys that only apply when the model capability sets the matching feature.
FEATURE_GATED_TRAINING_KEYS: dict[str, str] = {
    "blocks_to_swap": "block_swap",
    "block_swap_prefetch": "block_swap",
    "disable_block_swap_for_eval": "block_swap",
    "disable_block_swap_for_preview": "block_swap",
    "tread": "tread",
}


def _path_to_model_key(path: str) -> str:
    if path.startswith("model."):
        return path.split(".", 1)[1]
    return path


def required_model_keys(cap: ModelCapability) -> list[str]:
    """TOML keys under ``[model]`` required for this capability (from form registry)."""
    extra = list((cap.model_validation or {}).get("required") or [])
    in_one_of = {k for group in one_of_groups(cap) for k in group}
    from_fields: list[str] = []
    for spec in cap.model_fields:
        if not spec.get("required"):
            continue
        if spec.get("ui") is False:
            continue
        key = _path_to_model_key(spec["path"])
        if key in in_one_of:
            continue
        from_fields.append(key)
    # preserve order: fields first, then explicit extras
    seen: set[str] = set()
    out: list[str] = []
    for key in from_fields + extra:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def one_of_groups(cap: ModelCapability) -> list[list[str]]:
    return [list(g) for g in (cap.model_validation or {}).get("one_of") or []]


def validate_model_section(model: dict[str, Any], *, raw_type: str) -> None:
    """Validate ``config['model']`` for a registered pipeline type.

    Args:
        model: The ``[model]`` table (must include ``type``).
        raw_type: Value of ``model['type']`` as written in TOML (may be an alias).

    Raises:
        ConfigValidationError: On missing or incompatible keys.
    """
    canonical = normalize_model_type(raw_type)
    cap = get_capability(raw_type)
    if not cap:
        return

    ConfigValidationError = _validation_error()

    for key in required_model_keys(cap):
        if key not in model:
            raise ConfigValidationError(
                f"config['model'] must contain '{key}' for {canonical}."
            )

    for group in one_of_groups(cap):
        if not any(k in model for k in group):
            keys = "', '".join(group)
            raise ConfigValidationError(
                f"config['model'] must contain at least one of: '{keys}' (for {canonical})."
            )

    for spec in cap.model_fields:
        key = _path_to_model_key(spec["path"])
        if model.get(key) is not None:
            _check_field_value(spec, key, model[key], canonical)


def _check_field_value(spec: dict[str, Any], key: str, value: Any, canonical: str) -> None:
    """Enforce a field spec's fixed ``options`` and numeric ``min`` / ``gt`` bounds, so a bad
    value fails at validate time instead of deep in the pipeline (or silently)."""
    ConfigValidationError = _validation_error()
    options = spec.get("options")
    if options and value not in options:
        allowed = ", ".join(repr(o) for o in options)
        raise ConfigValidationError(
            f"model.{key} must be one of {allowed} for {canonical}, got {value!r}."
        )
    if spec.get("type") not in ("integer", "number") or ("min" not in spec and "gt" not in spec):
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"model.{key} must be a number, got {value!r}.")
    if spec.get("type") == "integer" and not float(value).is_integer():
        raise ConfigValidationError(f"model.{key} must be an integer, got {value!r}.")
    if "min" in spec and value < spec["min"]:
        raise ConfigValidationError(f"model.{key} must be >= {spec['min']}, got {value!r}.")
    if "gt" in spec and value <= spec["gt"]:
        hint = " Remove the key to use the default." if "default" not in spec else ""
        raise ConfigValidationError(f"model.{key} must be > {spec['gt']}, got {value!r}.{hint}")


def validate_tread(config: dict[str, Any], cap: ModelCapability) -> None:
    """Config-time mirror of the pipeline's ``[tread]`` checks (``Krea2Pipeline.to_layers``
    keeps its own; this surfaces the error before any model is loaded)."""
    tread = config.get("tread")
    if not tread:
        return
    ConfigValidationError = _validation_error()
    if not isinstance(tread, dict):
        raise ConfigValidationError("[tread] must be a table.")
    if tread.get("drop_ratio") is None:
        raise ConfigValidationError(
            "[tread] needs drop_ratio (fraction of image tokens routed around the middle "
            "blocks, e.g. 0.5); remove the [tread] table to turn routing off."
        )
    try:
        drop_ratio = float(tread["drop_ratio"])
        disable_after_frac = float(tread.get("disable_after_frac", 1.0))
        start_block = int(tread.get("start_block", 2))
        end_block = int(tread.get("end_block", -3))
    except (TypeError, ValueError) as e:
        raise ConfigValidationError(f"[tread] has a non-numeric value: {e}") from e
    if not 0.0 < drop_ratio < 1.0:
        raise ConfigValidationError(f"tread.drop_ratio must be in (0, 1), got {drop_ratio}.")
    if not 0.0 < disable_after_frac <= 1.0:
        raise ConfigValidationError(
            f"tread.disable_after_frac must be in (0, 1], got {disable_after_frac}."
        )
    n = cap.transformer_blocks
    if n:
        # Same normalization as training.token_routing.resolve_route (torch-free here).
        start = start_block if start_block >= 0 else n + start_block
        end = end_block if end_block >= 0 else n + end_block
        if not 0 < start < end < n - 1:
            raise ConfigValidationError(
                f"tread route [{start_block}, {end_block}] resolves to [{start}, {end}] on "
                f"{n} blocks; need 0 < start < end < {n - 1} (keep the first and last block "
                "unrouted)."
            )


def validate_training_keys_for_model(config: dict[str, Any]) -> None:
    """Warn via exception when training keys are set but unsupported for ``model.type``."""
    model = config.get("model") or {}
    raw_type = str(model.get("type", "")).lower()
    cap = get_capability(raw_type)
    if not cap:
        return
    ConfigValidationError = _validation_error()
    features = cap.features or {}
    for key, feature in FEATURE_GATED_TRAINING_KEYS.items():
        if key not in config:
            continue
        val = config[key]
        if val in (None, "", 0, False):
            continue
        if not features.get(feature):
            raise ConfigValidationError(
                f"'{key}' is only supported for models with feature '{feature}' "
                f"(not for {cap.type_id}). Remove it or change model.type."
            )


def validate_config_model_rules(config: dict[str, Any]) -> None:
    """Apply all model-capability rules to a full config dict."""
    if "model" not in config or "type" not in config["model"]:
        return
    raw_type = str(config["model"]["type"])
    # Require the canonical type id here. Aliases (e.g. "anima" → "cosmos_predict2")
    # are resolved by the registry/UI form, which rewrites model.type to canonical
    # before save; a raw alias reaching validation means the config was hand-written
    # against a legacy/unsupported name and must be rejected.
    if raw_type not in get_canonical_model_types():
        ConfigValidationError = _validation_error()
        registered = sorted(get_canonical_model_types())
        raise ConfigValidationError(
            f"Unknown model type {raw_type!r}. Use one of: {registered}."
        )
    validate_model_section(config["model"], raw_type=raw_type)
    validate_training_keys_for_model(config)
    cap = get_capability(raw_type)
    if cap and (cap.features or {}).get("tread"):
        validate_tread(config, cap)
