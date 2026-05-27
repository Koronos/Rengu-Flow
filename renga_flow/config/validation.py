"""Minimal validation of config: required sections for the standard flow."""

from __future__ import annotations

from typing import Any

from renga_flow.config.dataset_library_ref import collect_script_dataset_library_ref_issues
from renga_flow.registry.model_config_rules import validate_config_model_rules


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

    When ``for_script`` is true (CLI / ``renga_flow.main``), UI-only
    ``renga-flow-dataset:`` refs are reported so users export dataset TOML first.
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

    if "type" in config.get("optimizer", {}):
        optimizer = config["optimizer"]
        if optimizer.get("gradient_release") and config.get("pipeline_stages", 1) != 1:
            issues.append(
                "optimizer.gradient_release requires pipeline_stages = 1 (single-GPU pipeline)."
            )

    cache_format = config.get("cache_format")
    if cache_format is not None and cache_format not in ("v1", "v2"):
        issues.append("cache_format must be `v1` or `v2`.")

    adapter = config.get("adapter")
    if adapter is not None:
        if not isinstance(adapter, dict):
            issues.append("[adapter] must be a table when present.")
        else:
            if "type" not in adapter:
                issues.append("adapter.type is required when using an adapter — `lora` or `lokr`.")
            elif adapter["type"] not in ("lora", "lokr"):
                issues.append(
                    f"adapter.type must be `lora` or `lokr`, not {adapter['type']!r}."
                )
            elif "rank" not in adapter and "dim" not in adapter:
                issues.append("adapter.rank (or adapter.dim) is required for LoRA / LoKr training.")

    if for_script:
        issues.extend(collect_script_dataset_library_ref_issues(config))

    return issues


def validate_config(config: dict[str, Any], *, for_script: bool = False) -> None:
    """Check that config has the minimum required sections for the orchestrator.

    Required: 'model', 'optimizer', 'dataset'. 'model' must have 'type' and 'dtype';
    'optimizer' must have 'type'.

    Per-model ``[model]`` keys and feature-gated training options are enforced via
    ``renga_flow.registry.model_config_rules`` (same registry as the web UI).

    The 'adapter' section is optional. When absent, full-model finetuning is used
    (all trainable parameters; save via save_full_model). When present, adapter
    training (e.g. LoRA/LoKr) is used and only adapter weights are saved.

    Raises:
        ConfigValidationError: If any required key is missing.
    """
    issues = collect_validation_errors(config, for_script=for_script)
    if issues:
        raise ConfigValidationError(format_validation_issues(issues))
