"""Parse, merge, and fingerprint augmentation configuration."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from rengu_flow.data.augmentation.errors import (
    AugmentationConfigError,
    AugmentationStrategyNotImplementedError,
)
from rengu_flow.data.augmentation.names import (
    ALL_PRESET_NAMES,
    DEFERRED_PRESET_NAMES,
    DEFERRED_PRESET_STRATEGIES,
    IMPLEMENTED_STRATEGIES,
    KNOWN_STRATEGIES,
    MVP_PRESET_NAMES,
    SEED_MODES,
)
from rengu_flow.data.augmentation.presets import get_preset_strategies


def _merge_strategy_entry(
    base: dict[str, Any] | None, override: dict[str, Any]
) -> dict[str, Any]:
    out = deepcopy(base) if base else {}
    for key, val in override.items():
        out[key] = val
    if "enabled" not in out:
        out["enabled"] = True
    return out


def _global_augmentation_defaults(dataset_config: dict) -> dict[str, Any]:
    dataset_section = dataset_config.get("dataset") or {}
    aug = dataset_section.get("augmentation") or {}
    if not isinstance(aug, dict):
        raise AugmentationConfigError("'dataset.augmentation' must be a table.")
    return dict(aug)


def _directory_augmentation_raw(directory_config: dict) -> dict[str, Any]:
    aug = directory_config.get("augmentation")
    if aug is None:
        return {}
    if not isinstance(aug, dict):
        raise AugmentationConfigError(
            f"directory {directory_config.get('path')!r}: 'augmentation' must be a table."
        )
    return dict(aug)


def merge_directory_augmentation(
    directory_config: dict, dataset_config: dict
) -> dict[str, Any]:
    """Merge global defaults with per-directory augmentation table (before resolve)."""
    global_aug = _global_augmentation_defaults(dataset_config)
    dir_aug = _directory_augmentation_raw(directory_config)
    merged: dict[str, Any] = {}
    for key in ("enabled", "preset", "seed_mode", "branches_per_image"):
        if key in dir_aug:
            merged[key] = dir_aug[key]
        elif key in global_aug:
            merged[key] = global_aug[key]
    strategies: dict[str, Any] = {}
    if "strategies" in global_aug and isinstance(global_aug["strategies"], dict):
        strategies.update(deepcopy(global_aug["strategies"]))
    if "strategies" in dir_aug and isinstance(dir_aug["strategies"], dict):
        for name, params in dir_aug["strategies"].items():
            if not isinstance(params, dict):
                raise AugmentationConfigError(
                    f"strategies.{name} must be a parameter table."
                )
            strategies[name] = _merge_strategy_entry(
                strategies.get(name), params
            )
    if strategies:
        merged["strategies"] = strategies
    if "enable_strategies" in dir_aug:
        merged["enable_strategies"] = list(dir_aug["enable_strategies"])
    elif "enable_strategies" in global_aug:
        merged["enable_strategies"] = list(global_aug["enable_strategies"])
    return merged


def resolve_augmentation_config(
    directory_config: dict, dataset_config: dict
) -> dict[str, Any]:
    """Resolve merged augmentation config per spec merge algorithm."""
    raw = merge_directory_augmentation(directory_config, dataset_config)
    enabled = bool(raw.get("enabled", False))
    preset = str(raw.get("preset", "none")).strip().lower() or "none"
    if preset not in ALL_PRESET_NAMES:
        raise AugmentationConfigError(
            f"Unknown augmentation preset {preset!r}. "
            f"Known presets: {', '.join(sorted(ALL_PRESET_NAMES))}."
        )
    seed_mode = str(raw.get("seed_mode", "deterministic_per_image")).strip()
    if seed_mode not in SEED_MODES:
        raise AugmentationConfigError(
            f"Invalid seed_mode {seed_mode!r}; use {', '.join(sorted(SEED_MODES))}."
        )
    branches_per_image = int(raw.get("branches_per_image", 1) or 0)
    if branches_per_image < 0:
        raise AugmentationConfigError("branches_per_image must be 0 or a positive integer.")

    if not enabled:
        return {
            "enabled": False,
            "preset": preset,
            "seed_mode": seed_mode,
            "branches_per_image": branches_per_image,
            "strategies": {},
        }

    if preset in DEFERRED_PRESET_NAMES:
        missing = DEFERRED_PRESET_STRATEGIES.get(preset, frozenset())
        raise AugmentationStrategyNotImplementedError(
            f"Preset {preset!r} is not available in the MVP build "
            f"(requires strategies: {', '.join(sorted(missing))}). "
            f"Use an MVP preset: {', '.join(sorted(MVP_PRESET_NAMES - {'none', 'custom'}))}."
        )

    strategies: dict[str, dict[str, Any]] = {}
    if preset not in ("none", "custom"):
        strategies = get_preset_strategies(preset)

    user_strategies = raw.get("strategies") or {}
    if not isinstance(user_strategies, dict):
        raise AugmentationConfigError("'strategies' must be a map of strategy name → params.")
    for name, params in user_strategies.items():
        strategies[name] = _merge_strategy_entry(strategies.get(name), params)

    if preset in ("none", "custom") and not user_strategies and enabled:
        if preset == "custom":
            raise AugmentationConfigError(
                "preset 'custom' with enabled=true requires explicit 'strategies' entries."
            )

    enable_list = raw.get("enable_strategies")
    if enable_list is not None:
        if not isinstance(enable_list, list) or not enable_list:
            raise AugmentationConfigError(
                "enable_strategies must be a non-empty list of strategy names."
            )
        allow = set()
        for name in enable_list:
            if not isinstance(name, str):
                raise AugmentationConfigError("enable_strategies entries must be strings.")
            if name not in strategies:
                raise AugmentationConfigError(
                    f"enable_strategies lists {name!r} which is not in the resolved preset."
                )
            allow.add(name)
        strategies = {k: v for k, v in strategies.items() if k in allow}
        if not strategies:
            raise AugmentationConfigError(
                "enable_strategies intersection with preset is empty."
            )

    resolved_strategies: dict[str, dict[str, Any]] = {}
    for name, entry in strategies.items():
        if name not in KNOWN_STRATEGIES:
            suggestions = sorted(
                s for s in KNOWN_STRATEGIES if name[:3] in s or s[:3] in name
            )[:5]
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise AugmentationConfigError(f"Unknown strategy name {name!r}.{hint}")
        if entry.get("enabled") is False:
            continue
        if name not in IMPLEMENTED_STRATEGIES:
            raise AugmentationStrategyNotImplementedError(
                f"Strategy {name!r} is not implemented in the MVP build."
            )
        # Drop any legacy per-strategy 'sampling' key gracefully; everything is probabilistic now.
        params = {k: v for k, v in entry.items() if k not in ("enabled", "sampling")}
        resolved_strategies[name] = {
            "enabled": True,
            "params": params,
        }

    if enabled and not resolved_strategies:
        raise AugmentationConfigError(
            "Augmentation is enabled but no strategies are active after merge."
        )

    return {
        "enabled": enabled,
        "preset": preset,
        "seed_mode": seed_mode,
        "branches_per_image": branches_per_image,
        "strategies": resolved_strategies,
    }


def augmentation_fingerprint(resolved: dict[str, Any]) -> str:
    """Stable hash input for latent cache invalidation."""
    payload = {
        "enabled": resolved.get("enabled"),
        "preset": resolved.get("preset"),
        "seed_mode": resolved.get("seed_mode"),
        "branches_per_image": resolved.get("branches_per_image"),
        "strategies": resolved.get("strategies"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def is_augmentation_enabled(resolved: dict[str, Any]) -> bool:
    return bool(resolved.get("enabled") and resolved.get("strategies"))


def validate_augmentation_for_directory(
    directory_config: dict,
    dataset_config: dict,
) -> dict[str, Any]:
    """Resolve and validate augmentation (images only; deterministic cache)."""
    resolved = resolve_augmentation_config(directory_config, dataset_config)
    if is_augmentation_enabled(resolved):
        if resolved.get("seed_mode") == "stochastic":
            raise AugmentationConfigError(
                f"Directory {directory_config.get('path')!r}: seed_mode 'stochastic' is not "
                "compatible with fixed latent caching in this release. Use "
                "'deterministic_per_image'."
            )
        frame_buckets = directory_config.get(
            "frame_buckets", dataset_config.get("frame_buckets", [1])
        )
        if any(int(f) > 1 for f in frame_buckets):
            raise AugmentationConfigError(
                f"Directory {directory_config.get('path')!r}: image augmentation is not "
                "supported for video frame_buckets > 1 in this release."
            )
    return resolved
