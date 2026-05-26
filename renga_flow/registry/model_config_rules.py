"""Model-specific config rules shared by TOML validation and the web UI.

Single source for *what each model needs* under ``[model]``:

- **Required keys** — from ``ModelCapability.model_fields`` where ``required: true``
  (and optional ``model_validation`` overrides).
- **One-of groups** — e.g. ``llm_path`` or ``t5_path`` for Cosmos (TOML-only expert path).
- **Feature-gated training keys** — e.g. ``blocks_to_swap`` only when ``features.block_swap``.

The UI hides irrelevant fields via ``renga_flow_ui/field_visibility.py``; this module
enforces the same intent at validate time so CLI and UI stay aligned.

To extend a model, edit its ``ModelCapability`` in ``model_capabilities.py`` — avoid
duplicating checks in ``validation.py``.
"""

from __future__ import annotations

from typing import Any

from renga_flow.registry.model_capabilities import (
    ModelCapability,
    get_canonical_model_types,
    get_capability,
    normalize_model_type,
)


def _validation_error() -> type[Exception]:
    from renga_flow.config.validation import ConfigValidationError

    return ConfigValidationError

# Training keys that only apply when the model capability sets the matching feature.
FEATURE_GATED_TRAINING_KEYS: dict[str, str] = {
    "blocks_to_swap": "block_swap",
    "disable_block_swap_for_eval": "block_swap",
    "disable_block_swap_for_preview": "block_swap",
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


def raw_type_rules(cap: ModelCapability, raw_type: str) -> dict[str, Any]:
    by_raw = (cap.model_validation or {}).get("by_raw_type") or {}
    return dict(by_raw.get(raw_type.lower()) or {})


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

    alias_rules = raw_type_rules(cap, raw_type)
    for key in alias_rules.get("required") or []:
        if key not in model:
            raise ConfigValidationError(
                f"config['model'] must contain '{key}' when type is '{raw_type}'."
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
    if get_capability(raw_type) is None:
        ConfigValidationError = _validation_error()
        registered = sorted(get_canonical_model_types())
        raise ConfigValidationError(
            f"Unknown model type {raw_type!r}. Use one of: {registered}."
        )
    validate_model_section(config["model"], raw_type=raw_type)
    validate_training_keys_for_model(config)
