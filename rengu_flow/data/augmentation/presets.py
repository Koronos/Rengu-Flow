"""Default strategy sets per preset name."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Each preset: strategy_name -> {enabled, **params}
_PRESET_BODIES: dict[str, dict[str, dict[str, Any]]] = {
    "none": {},
    "custom": {},
    "easy": {
        "color_jitter": {
            "enabled": True,
            "brightness": 0.03,
            "contrast": 0.03,
            "saturation": 0.03,
            "hue": 0.01,
        },
        "gamma": {"enabled": True, "gamma_min": 0.97, "gamma_max": 1.03},
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.04},
    },
    "anime_mixed": {
        "color_jitter": {
            "enabled": True,
            "brightness": 0.03,
            "contrast": 0.03,
            "saturation": 0.02,
            "hue": 0.008,
        },
        "gamma": {"enabled": True, "gamma_min": 0.97, "gamma_max": 1.03},
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.03},
        "horizontal_flip": {"enabled": False, "probability": 0.0},
    },
    "manga_mixed": {
        "color_jitter": {
            "enabled": True,
            "brightness": 0.025,
            "contrast": 0.025,
            "saturation": 0.015,
            "hue": 0.005,
        },
        "gamma": {"enabled": True, "gamma_min": 0.98, "gamma_max": 1.02},
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.025},
        "horizontal_flip": {"enabled": False, "probability": 0.0},
    },
    "anime": {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "color_jitter": {
            "enabled": True,
            "brightness": 0.04,
            "contrast": 0.04,
            "saturation": 0.03,
            "hue": 0.01,
        },
        "gamma": {"enabled": True, "gamma_min": 0.96, "gamma_max": 1.04},
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.04},
    },
    "manga_bw": {
        "gamma": {"enabled": True, "gamma_min": 0.98, "gamma_max": 1.02},
    },
    "photo_safe": {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "color_jitter": {
            "enabled": True,
            "brightness": 0.05,
            "contrast": 0.05,
            "saturation": 0.04,
            "hue": 0.015,
        },
        "gamma": {"enabled": True, "gamma_min": 0.95, "gamma_max": 1.05},
        "jpeg_simulation": {
            "enabled": True,
            "quality_min": 85,
            "quality_max": 98,
        },
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.05},
    },
    "realism_general": {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "color_jitter": {
            "enabled": True,
            "brightness": 0.06,
            "contrast": 0.06,
            "saturation": 0.05,
            "hue": 0.02,
        },
        "gamma": {"enabled": True, "gamma_min": 0.9, "gamma_max": 1.1},
        "jpeg_simulation": {
            "enabled": True,
            "quality_min": 80,
            "quality_max": 95,
        },
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.06},
        "gaussian_noise": {"enabled": True, "sigma": 2.0},
        "chromatic_aberration": {"enabled": True, "shift_px": 1.5},
    },
    "bw_photo": {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "gamma": {"enabled": True, "gamma_min": 0.94, "gamma_max": 1.06},
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.03},
        "film_grain": {"enabled": True, "intensity": 0.04},
    },
    "sepia": {
        "gamma": {"enabled": True, "gamma_min": 0.96, "gamma_max": 1.04},
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.08},
        "split_toning": {"enabled": True, "strength": 0.15},
    },
    "photo_cinematic": {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "color_jitter": {
            "enabled": True,
            "brightness": 0.05,
            "contrast": 0.05,
            "saturation": 0.04,
            "hue": 0.015,
        },
        "gamma": {"enabled": True, "gamma_min": 0.95, "gamma_max": 1.05},
        "jpeg_simulation": {
            "enabled": True,
            "quality_min": 85,
            "quality_max": 98,
        },
        "temperature_tint": {"enabled": True, "warm_cool_range": 0.05},
        "local_tone_mapping": {"enabled": True, "strength": 0.12},
        "bloom": {"enabled": True, "intensity": 0.08},
        "chromatic_aberration": {"enabled": True, "shift_px": 1.5},
        "split_toning": {"enabled": True, "strength": 0.12},
        "cross_process_lut": {"enabled": True, "strength": 0.1},
    },
    "retro_scan": {
        "paper_texture": {"enabled": True, "opacity": 0.08},
        "moire": {"enabled": True, "strength": 0.05},
        "vhs_analogue": {"enabled": False, "strength": 0.05},
        "scan_dust": {"enabled": False, "strength": 0.03},
    },
    "manga_print": {
        "dithering": {"enabled": True, "strength": 0.1},
        "halftone": {"enabled": True, "frequency": 0.05},
        "screentone": {"enabled": True, "density": 0.05},
    },
}


def get_preset_strategies(preset_name: str) -> dict[str, dict[str, Any]]:
    """Return a deep copy of the preset strategy map."""
    if preset_name not in _PRESET_BODIES:
        raise KeyError(preset_name)
    return deepcopy(_PRESET_BODIES[preset_name])
