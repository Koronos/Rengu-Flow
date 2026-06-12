"""Minimal validation of config: required sections for the standard flow."""

from __future__ import annotations

from typing import Any

from rengu_flow.config.dataset_library_ref import collect_script_dataset_library_ref_issues
from rengu_flow.registry.model_config_rules import validate_config_model_rules
from rengu_flow.run_naming import collect_run_name_validation_errors


class ConfigValidationError(ValueError):
    """Raised when required config keys are missing."""


_REQUIRED_TOP_LEVEL = ("model", "optimizer", "dataset")

_SECTION_HINTS: dict[str, str] = {
    "dataset": (
        "Set `dataset` to a dataset TOML path (or a list of paths to merge at train time)."
    ),
    "model": "Add a `[model]` table with `type` (e.g. `sdxl`, `cosmos_predict2`) and `dtype` (e.g. `bfloat16`).",
    "optimizer": "Add an `[optimizer]` table with at least `type` (e.g. `adamw`) and `lr`.",
}


def section_hints_for_empty_config() -> list[str]:
    """Hints shown when the config editor is empty (web UI)."""
    return [_SECTION_HINTS[k] for k in ("dataset", "model", "optimizer") if k in _SECTION_HINTS]


def format_validation_issues(issues: list[str]) -> str:
    """Single string for CLI / simple alerts."""
    if not issues:
        return ""
    if len(issues) == 1:
        return issues[0]
    return "Fix the following:\n" + "\n".join(f"• {line}" for line in issues)


def collect_validation_errors(
    config: dict[str, Any],
    *,
    for_script: bool = False,
) -> list[str]:
    """Return all validation problems (empty list if structurally ready for defaults).

    When ``for_script`` is true (CLI / ``rengu_flow.main``), UI-only
    ``rengu-flow-dataset:`` refs are reported so users export dataset TOML first.
    """
    issues: list[str] = []

    if not isinstance(config, dict):
        return ["Config must be a TOML table at the top level."]

    for section in _REQUIRED_TOP_LEVEL:
        if section not in config:
            hint = _SECTION_HINTS.get(section, "")
            issues.append(f"Missing `{section}`." + (f" {hint}" if hint else ""))

    dataset = config.get("dataset")
    if "dataset" in config:
        if dataset is None:
            issues.append("dataset is empty — set a path to your dataset TOML file.")
        elif isinstance(dataset, str):
            if not dataset.strip():
                issues.append("dataset is empty — set a path to your dataset TOML file.")
        elif isinstance(dataset, list):
            paths = [x for x in dataset if isinstance(x, str) and x.strip()]
            if not paths:
                issues.append(
                    "dataset list is empty — add one or more dataset TOML paths."
                )
        else:
            issues.append(
                "dataset must be a path string or a list of path strings."
            )

    model = config.get("model")
    if "model" in config and not isinstance(model, dict):
        issues.append("[model] must be a table (use `[model]` in TOML).")
    elif isinstance(model, dict):
        if "type" not in model or model.get("type") in (None, ""):
            issues.append(
                "model.type is required — choose a registered type such as `sdxl` or `cosmos_predict2`."
            )
        if "dtype" not in model or model.get("dtype") in (None, ""):
            issues.append("model.dtype is required — e.g. `bfloat16` or `float16`.")

    optimizer = config.get("optimizer")
    if "optimizer" in config and not isinstance(optimizer, dict):
        issues.append("[optimizer] must be a table.")
    elif isinstance(optimizer, dict) and "type" not in optimizer:
        issues.append("optimizer.type is required — e.g. `adamw`.")

    try:
        validate_config_model_rules(config)
    except ConfigValidationError as e:
        issues.append(str(e))

    try:
        from rengu_flow.training.optimizer_hooks import validate_fused_optimizer_config

        validate_fused_optimizer_config(config)
    except ValueError as e:
        issues.append(str(e))

    if "type" in config.get("optimizer", {}):
        optimizer = config["optimizer"]
        if optimizer.get("gradient_release") and config.get("pipeline_stages", 1) != 1:
            issues.append(
                "optimizer.gradient_release requires pipeline_stages = 1 (single-GPU pipeline)."
            )

    if config.get("cache_format") not in (None, "v2"):
        issues.append(
            "cache_format v1 is removed; omit cache_format or set cache_format = \"v2\"."
        )

    train_seed = config.get("train_seed")
    if train_seed is not None:
        try:
            int(train_seed)
        except (TypeError, ValueError):
            issues.append("train_seed must be an integer.")

    cache_root = config.get("cache_root")
    if cache_root is not None and (
        not isinstance(cache_root, str) or not str(cache_root).strip()
    ):
        issues.append("cache_root must be a non-empty path string when set.")

    adapter = config.get("adapter")
    if adapter is not None:
        if not isinstance(adapter, dict):
            issues.append("[adapter] must be a table when present.")
        else:
            from rengu_flow.networks.lycoris_meta import (
                LYCORIS_ADAPTER_TYPES,
                collect_lycoris_adapter_issues,
                is_lycoris_type,
            )

            if "type" not in adapter:
                issues.append(
                    "adapter.type is required when using an adapter — `lora`, `lokr`, "
                    "or a `lycoris_*` type."
                )
            elif adapter["type"] not in ("lora", "lokr") and not is_lycoris_type(adapter["type"]):
                known = ", ".join(f"`{t}`" for t in ("lora", "lokr", *LYCORIS_ADAPTER_TYPES))
                issues.append(
                    f"adapter.type {adapter['type']!r} is not supported. Available: {known}."
                )
            elif is_lycoris_type(adapter["type"]) and (
                lycoris_issues := collect_lycoris_adapter_issues(adapter)
            ):
                issues.extend(lycoris_issues)
            elif adapter["type"] == "lycoris_dylora" and config.get("activation_checkpointing"):
                # DyLoRA samples a random sub-rank per forward; checkpoint recompute
                # draws a different one and fails on mismatched tensor metadata.
                issues.append(
                    "lycoris_dylora requires activation_checkpointing = false "
                    "(its random sub-rank per forward breaks checkpoint recompute)."
                )
            elif "rank" not in adapter and "dim" not in adapter:
                issues.append("adapter.rank (or adapter.dim) is required for adapter training.")

    max_exports = config.get("max_model_exports_to_keep")
    if max_exports is not None:
        try:
            if int(max_exports) < 1:
                issues.append("max_model_exports_to_keep must be a positive integer.")
        except (TypeError, ValueError):
            issues.append("max_model_exports_to_keep must be a positive integer.")

    min_export_step = config.get("keep_exports_from_step")
    if min_export_step is not None:
        try:
            if int(min_export_step) < 0:
                issues.append("keep_exports_from_step must be >= 0.")
        except (TypeError, ValueError):
            issues.append("keep_exports_from_step must be an integer.")

    issues.extend(collect_run_name_validation_errors(config))

    if for_script:
        issues.extend(collect_script_dataset_library_ref_issues(config))

    return issues


def validate_config(config: dict[str, Any], *, for_script: bool = False) -> None:
    """Check that config has the minimum required sections for the orchestrator.

    Required: 'model', 'optimizer', 'dataset'. 'model' must have 'type' and 'dtype';
    'optimizer' must have 'type'.

    Per-model ``[model]`` keys and feature-gated training options are enforced via
    ``rengu_flow.registry.model_config_rules`` (same registry as the web UI).

    The 'adapter' section is optional. When absent, full-model finetuning is used
    (all trainable parameters; save via save_full_model). When present, adapter
    training (e.g. LoRA/LoKr) is used and only adapter weights are saved.

    Raises:
        ConfigValidationError: If any required key is missing.
    """
    issues = collect_validation_errors(config, for_script=for_script)
    if issues:
        raise ConfigValidationError(format_validation_issues(issues))
