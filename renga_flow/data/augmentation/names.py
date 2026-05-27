"""Canonical augmentation strategy names (string identifiers)."""

from __future__ import annotations

# Tier A + B + horizontal_flip (MVP).
IMPLEMENTED_STRATEGIES = frozenset(
    {
        "horizontal_flip",
        "color_jitter",
        "gamma",
        "jpeg_simulation",
        "temperature_tint",
        "chromatic_aberration",
        "gaussian_noise",
        "crop_jitter",
        "small_rotation",
        "film_grain",
        "lab_jitter",
        "split_toning",
    }
)

# Strategies with a documented finite branch set (may use sampling=enumerated).
ENUMERABLE_STRATEGIES = frozenset({"horizontal_flip"})

# Presets that ship in config but require not-yet-implemented strategies.
DEFERRED_PRESET_STRATEGIES: dict[str, frozenset[str]] = {
    "photo_cinematic": frozenset(
        {
            "local_tone_mapping",
            "bloom",
            "cross_process_lut",
        }
    ),
    "retro_scan": frozenset({"paper_texture", "moire", "vhs_analogue", "scan_dust"}),
    "manga_print": frozenset({"dithering", "halftone", "screentone"}),
}

KNOWN_STRATEGIES = IMPLEMENTED_STRATEGIES | frozenset(
    s for strategies in DEFERRED_PRESET_STRATEGIES.values() for s in strategies
) | frozenset(
    {
        "vertical_flip",
        "gaussian_blur",
        "motion_blur",
        "unsharp_mask",
        "channel_dropout",
        "scale_translate",
        "shear",
        "perspective",
        "random_erasing",
        "clahe",
        "vignette",
        "multiscale_jitter",
        "exposure_bracket_fusion",
        "bloom",
        "lens_distortion",
        "posterize",
        "dithering",
        "halftone",
        "paper_texture",
        "moire",
        "vhs_analogue",
        "crt",
        "bilateral",
        "micro_contrast",
        "fft_band",
        "cross_process_lut",
        "orton",
        "dehaze",
        "chromatic_noise",
        "rain_overlay",
        "fog_overlay",
        "lens_flare",
        "bokeh_disk",
        "sensor_banding",
        "scan_dust",
        "screentone",
        "meta_external",
        "local_tone_mapping",
    }
)

MVP_PRESET_NAMES = frozenset(
    {
        "none",
        "custom",
        "easy",
        "anime",
        "anime_mixed",
        "manga_mixed",
        "manga_bw",
        "photo_safe",
        "realism_general",
        "bw_photo",
        "sepia",
    }
)

DEFERRED_PRESET_NAMES = frozenset({"photo_cinematic", "retro_scan", "manga_print"})

ALL_PRESET_NAMES = MVP_PRESET_NAMES | DEFERRED_PRESET_NAMES

SEED_MODES = frozenset({"deterministic_per_image", "stochastic"})
VARIANT_SAMPLING_MODES = frozenset({"probability", "enumerated"})
AUG_MVP_VERSION = "aug_mvp=1"
