"""Augmentation catalog for the web UI (single source with training code)."""

from __future__ import annotations

from typing import Any

from rengu_flow.data.augmentation.names import (
    ALL_PRESET_NAMES,
    AUG_MVP_VERSION,
    DEFERRED_PRESET_NAMES,
    ENUMERABLE_STRATEGIES,
    IMPLEMENTED_STRATEGIES,
    MVP_PRESET_NAMES,
    SEED_MODES,
    VARIANT_SAMPLING_MODES,
)
from rengu_flow.data.augmentation.presets import get_preset_strategies
from rengu_flow.data.augmentation.registry import GEOMETRIC_ORDER, PHOTOMETRIC_ORDER

# Parameter fields mirror defaults in registry.py / presets.py.
_STRATEGY_PARAM_FIELDS: dict[str, list[dict[str, Any]]] = {
    "horizontal_flip": [
        {"path": "probability", "label": "Probability", "type": "number", "default": 0.5, "min": 0, "max": 1, "step": 0.05},
        {"path": "sampling", "label": "Sampling", "type": "select", "options": ["probability", "enumerated"], "default": "probability"},
    ],
    "color_jitter": [
        {"path": "brightness", "label": "Brightness", "type": "number", "default": 0.05, "min": 0, "step": 0.01},
        {"path": "contrast", "label": "Contrast", "type": "number", "default": 0.05, "min": 0, "step": 0.01},
        {"path": "saturation", "label": "Saturation", "type": "number", "default": 0.05, "min": 0, "step": 0.01},
        {"path": "hue", "label": "Hue", "type": "number", "default": 0.02, "min": 0, "step": 0.005},
    ],
    "gamma": [
        {"path": "gamma_min", "label": "Gamma min", "type": "number", "default": 0.95, "min": 0.1, "max": 2, "step": 0.01},
        {"path": "gamma_max", "label": "Gamma max", "type": "number", "default": 1.05, "min": 0.1, "max": 2, "step": 0.01},
    ],
    "jpeg_simulation": [
        {"path": "quality_min", "label": "Quality min", "type": "integer", "default": 85, "min": 1, "max": 100},
        {"path": "quality_max", "label": "Quality max", "type": "integer", "default": 98, "min": 1, "max": 100},
    ],
    "temperature_tint": [
        {"path": "warm_cool_range", "label": "Warm/cool range", "type": "number", "default": 0.05, "min": 0, "step": 0.01},
    ],
    "chromatic_aberration": [
        {"path": "shift_px", "label": "Shift (px)", "type": "integer", "default": 1, "min": 0, "max": 8},
    ],
    "gaussian_noise": [
        {"path": "sigma", "label": "Sigma", "type": "number", "default": 2.0, "min": 0, "step": 0.5},
    ],
    "crop_jitter": [
        {"path": "fraction", "label": "Fraction", "type": "number", "default": 0.02, "min": 0, "max": 0.2, "step": 0.005},
    ],
    "small_rotation": [
        {"path": "max_degrees", "label": "Max degrees", "type": "number", "default": 2.0, "min": 0, "max": 15, "step": 0.5},
    ],
    "film_grain": [
        {"path": "intensity", "label": "Intensity", "type": "number", "default": 0.04, "min": 0, "step": 0.01},
    ],
    "lab_jitter": [
        {"path": "delta_l", "label": "Delta L", "type": "number", "default": 3.0, "min": 0, "step": 0.5},
        {"path": "delta_a", "label": "Delta A", "type": "number", "default": 2.0, "min": 0, "step": 0.5},
        {"path": "delta_b", "label": "Delta B", "type": "number", "default": 2.0, "min": 0, "step": 0.5},
    ],
    "split_toning": [
        {"path": "strength", "label": "Strength", "type": "number", "default": 0.15, "min": 0, "max": 1, "step": 0.01},
    ],
}

# UI help for strategy override parameters (shown in dataset augmentation editor).
_STRATEGY_PARAM_HELP: dict[str, dict[str, str]] = {
    "horizontal_flip": {
        "probability": "Chance to apply a horizontal mirror when sampling = probability.",
        "sampling": (
            "Probability: one random branch per image in cache. "
            "Enumerated: cache both original and mirrored rows."
        ),
    },
    "color_jitter": {
        "brightness": "Max random shift in pixel brightness (fraction of full range).",
        "contrast": "Max random contrast scaling around the image mean.",
        "saturation": "Max random saturation scaling.",
        "hue": "Max random hue rotation (small values keep realism).",
    },
    "gamma": {
        "gamma_min": "Lower bound for random gamma correction.",
        "gamma_max": "Upper bound for random gamma correction.",
    },
    "jpeg_simulation": {
        "quality_min": "Lowest random JPEG quality (1–100) when simulating recompression.",
        "quality_max": "Highest random JPEG quality when simulating recompression.",
    },
    "temperature_tint": {
        "warm_cool_range": "How far white balance may shift toward warm or cool.",
    },
    "chromatic_aberration": {
        "shift_px": "RGB channel shift in pixels at image borders.",
    },
    "gaussian_noise": {
        "sigma": "Standard deviation of additive Gaussian noise (0–255 scale).",
    },
    "crop_jitter": {
        "fraction": "Random crop/resize jitter as a fraction of image size.",
    },
    "small_rotation": {
        "max_degrees": "Maximum absolute rotation in degrees.",
    },
    "film_grain": {
        "intensity": "Strength of synthetic film grain overlay.",
    },
    "lab_jitter": {
        "delta_l": "Max random shift in CIELAB L channel.",
        "delta_a": "Max random shift in CIELAB a channel.",
        "delta_b": "Max random shift in CIELAB b channel.",
    },
    "split_toning": {
        "strength": "Blend strength for shadow/highlight colour shifts.",
    },
}

_STRATEGY_SUMMARY_HELP: dict[str, str] = {
    "horizontal_flip": "Mirror images; enumerated sampling caches both orientations.",
    "color_jitter": "Random brightness, contrast, saturation, and hue shifts.",
    "gamma": "Random gamma correction within min/max bounds.",
    "jpeg_simulation": "Re-encode through JPEG at random quality to mimic compression artifacts.",
    "temperature_tint": "Shift white balance warmer or cooler.",
    "chromatic_aberration": "Separate RGB channels slightly for lens fringe effect.",
    "gaussian_noise": "Add light Gaussian noise.",
    "crop_jitter": "Small random crop and resize before bucketing.",
    "small_rotation": "Rotate within ± max_degrees.",
    "film_grain": "Overlay synthetic grain.",
    "lab_jitter": "Jitter colours in LAB space.",
    "split_toning": "Tint shadows and highlights differently.",
}


def _parameters_with_help(strategy: str) -> list[dict[str, Any]]:
    fields = _STRATEGY_PARAM_FIELDS.get(strategy, [])
    hints = _STRATEGY_PARAM_HELP.get(strategy, {})
    out: list[dict[str, Any]] = []
    for field in fields:
        merged = dict(field)
        if merged.get("path") in hints and "help" not in merged:
            merged["help"] = hints[merged["path"]]
        out.append(merged)
    return out


_PRESET_LABELS: dict[str, str] = {
    "none": "None",
    "custom": "Custom",
    "easy": "Easy",
    "anime": "Anime",
    "anime_mixed": "Anime mixed",
    "manga_mixed": "Manga mixed",
    "manga_bw": "Manga B&W",
    "photo_safe": "Photo safe",
    "realism_general": "Realism general",
    "bw_photo": "B&W photo",
    "sepia": "Sepia",
    "photo_cinematic": "Photo cinematic (deferred)",
    "retro_scan": "Retro scan (deferred)",
    "manga_print": "Manga print (deferred)",
}


def _strategy_catalog_entry(name: str, *, category: str) -> dict[str, Any]:
    return {
        "name": name,
        "label": name.replace("_", " ").title(),
        "category": category,
        "implemented": name in IMPLEMENTED_STRATEGIES,
        "enumerable": name in ENUMERABLE_STRATEGIES,
        "help": _STRATEGY_SUMMARY_HELP.get(name, ""),
        "parameters": _parameters_with_help(name),
    }


def get_augmentation_catalog() -> dict[str, Any]:
    """Return presets, strategies, and field metadata for the dataset editor UI."""
    strategies: list[dict[str, Any]] = []
    for name in GEOMETRIC_ORDER:
        strategies.append(_strategy_catalog_entry(name, category="geometric"))
    for name in PHOTOMETRIC_ORDER:
        strategies.append(_strategy_catalog_entry(name, category="photometric"))

    presets: list[dict[str, Any]] = []
    for name in sorted(ALL_PRESET_NAMES):
        try:
            body = get_preset_strategies(name)
        except KeyError:
            body = {}
        presets.append(
            {
                "name": name,
                "label": _PRESET_LABELS.get(name, name.replace("_", " ").title()),
                "available": name in MVP_PRESET_NAMES,
                "deferred": name in DEFERRED_PRESET_NAMES,
                "strategies": sorted(body.keys()),
                "strategy_defaults": body,
            }
        )

    return {
        "version": AUG_MVP_VERSION,
        "seed_modes": sorted(SEED_MODES),
        "variant_sampling_modes": sorted(VARIANT_SAMPLING_MODES),
        "presets": presets,
        "strategies": strategies,
        "directory_fields": [
            {"path": "enabled", "label": "Enable augmentation", "type": "boolean", "default": False},
            {"path": "preset", "label": "Preset", "type": "select", "default": "none"},
            {"path": "seed_mode", "label": "Seed mode", "type": "select", "default": "deterministic_per_image"},
            {"path": "variant_sampling", "label": "Variant sampling", "type": "select", "default": "probability"},
            {"path": "max_branches_per_image", "label": "Max branches per image", "type": "integer", "min": 1},
            {"path": "enable_strategies", "label": "Enable strategies", "type": "string_list"},
            {"path": "strategies", "label": "Strategy overrides", "type": "strategy_map"},
        ],
        "global_fields": [
            {"path": "enabled", "label": "Enable augmentation", "type": "boolean", "default": False},
            {"path": "preset", "label": "Preset", "type": "select", "default": "none"},
        ],
    }
