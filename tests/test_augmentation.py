"""Tests for dataset augmentation (MVP)."""

from __future__ import annotations


import pytest
from PIL import Image

from rengu_flow.data.augmentation import (
    apply_augmentation,
    augmentation_fingerprint,
    augmentation_seed_for_image,
    expand_variant_keys,
    resolve_augmentation_config,
)
from rengu_flow.data.augmentation.errors import (
    AugmentationConfigError,
    AugmentationStrategyNotImplementedError,
)
from rengu_flow.data.dataset_config import (
    DatasetConfigError,
    validate_dataset_config_for_real_data,
)


def _dataset_cfg(**directory_aug) -> dict:
    return {
        "resolutions": [1024],
        "frame_buckets": [1],
        "directory": [
            {
                "path": "/data/photos",
                "num_repeats": 1,
                "augmentation": directory_aug,
            }
        ],
    }


def test_resolve_easy_preset() -> None:
    resolved = resolve_augmentation_config(
        {"augmentation": {"enabled": True, "preset": "easy"}},
        {},
    )
    assert resolved["enabled"] is True
    assert "color_jitter" in resolved["strategies"]
    assert "horizontal_flip" not in resolved["strategies"]


def test_strategies_override_disables_flip() -> None:
    resolved = resolve_augmentation_config(
        {
            "augmentation": {
                "enabled": True,
                "preset": "photo_safe",
                "strategies": {"horizontal_flip": {"enabled": False}},
            }
        },
        {},
    )
    assert "horizontal_flip" not in resolved["strategies"]


def test_enable_strategies_intersection() -> None:
    resolved = resolve_augmentation_config(
        {
            "augmentation": {
                "enabled": True,
                "preset": "photo_safe",
                "enable_strategies": ["color_jitter", "gamma"],
            }
        },
        {},
    )
    assert set(resolved["strategies"]) == {"color_jitter", "gamma"}


def test_unknown_strategy_raises() -> None:
    with pytest.raises(AugmentationConfigError, match="Unknown strategy"):
        resolve_augmentation_config(
            {
                "augmentation": {
                    "enabled": True,
                    "preset": "none",
                    "strategies": {"not_a_real_strategy": {"brightness": 0.1}},
                }
            },
            {},
        )


def test_legacy_sampling_and_variant_sampling_ignored() -> None:
    # Old configs with variant_sampling / per-strategy sampling must degrade gracefully:
    # the keys are dropped, not rejected, and never leak into the resolved strategies.
    resolved = resolve_augmentation_config(
        {
            "augmentation": {
                "enabled": True,
                "preset": "none",
                "variant_sampling": "enumerated",
                "strategies": {
                    "color_jitter": {"brightness": 0.05, "sampling": "enumerated"},
                },
            }
        },
        {},
    )
    assert "color_jitter" in resolved["strategies"]
    assert "sampling" not in resolved["strategies"]["color_jitter"]
    assert "sampling" not in resolved["strategies"]["color_jitter"]["params"]
    assert "variant_sampling" not in resolved
    assert resolved["branches_per_image"] == 1


def test_deferred_preset_raises() -> None:
    with pytest.raises(AugmentationStrategyNotImplementedError, match="photo_cinematic"):
        resolve_augmentation_config(
            {"augmentation": {"enabled": True, "preset": "photo_cinematic"}},
            {},
        )


def test_deferred_preset_allowed_when_disabled() -> None:
    resolved = resolve_augmentation_config(
        {"augmentation": {"enabled": False, "preset": "photo_cinematic"}},
        {},
    )
    assert resolved["enabled"] is False
    assert resolved["preset"] == "photo_cinematic"
    assert resolved["strategies"] == {}


def test_stochastic_rejected() -> None:
    from rengu_flow.data.augmentation import validate_augmentation_for_directory

    with pytest.raises(AugmentationConfigError, match="stochastic"):
        validate_augmentation_for_directory(
            {
                "path": "/data/x",
                "augmentation": {
                    "enabled": True,
                    "preset": "easy",
                    "seed_mode": "stochastic",
                },
            },
            {},
        )


def test_video_frame_buckets_rejected() -> None:
    from rengu_flow.data.augmentation import validate_augmentation_for_directory

    with pytest.raises(AugmentationConfigError, match="video"):
        validate_augmentation_for_directory(
            {
                "path": "/data/v",
                "augmentation": {"enabled": True, "preset": "easy"},
                "frame_buckets": [9],
            },
            {"frame_buckets": [1]},
        )


def test_validate_dataset_config_wraps_augmentation() -> None:
    cfg = _dataset_cfg(enabled=True, preset="photo_cinematic")
    with pytest.raises(DatasetConfigError, match="photo_cinematic"):
        validate_dataset_config_for_real_data(cfg)


def test_fingerprint_changes_with_config() -> None:
    a = resolve_augmentation_config(
        {"augmentation": {"enabled": True, "preset": "easy"}}, {}
    )
    b = resolve_augmentation_config(
        {"augmentation": {"enabled": True, "preset": "photo_safe"}}, {}
    )
    assert augmentation_fingerprint(a) != augmentation_fingerprint(b)


def test_expand_branches_per_image() -> None:
    # branches_per_image = N → pristine original (None) + N augmented copies.
    base = {"enabled": True, "preset": "easy"}
    assert expand_variant_keys(
        resolve_augmentation_config({"augmentation": base}, {})
    ) == [None, "1"]  # default 1
    assert expand_variant_keys(
        resolve_augmentation_config({"augmentation": {**base, "branches_per_image": 3}}, {})
    ) == [None, "1", "2", "3"]
    assert expand_variant_keys(
        resolve_augmentation_config({"augmentation": {**base, "branches_per_image": 0}}, {})
    ) == [None]
    assert expand_variant_keys(
        resolve_augmentation_config({"augmentation": {"enabled": False, "preset": "easy"}}, {})
    ) == [None]


def test_branches_per_image_negative_rejected() -> None:
    with pytest.raises(AugmentationConfigError, match="branches_per_image"):
        resolve_augmentation_config(
            {"augmentation": {"enabled": True, "preset": "easy", "branches_per_image": -1}}, {}
        )


def test_apply_reproducible() -> None:
    resolved = resolve_augmentation_config(
        {"augmentation": {"enabled": True, "preset": "easy"}}, {}
    )
    img = Image.new("RGB", (64, 64), color=(128, 64, 200))
    # An augmented copy is deterministic for a given (seed, variant_key).
    out1, _ = apply_augmentation(img, None, 42, resolved, variant_key="1")
    out2, _ = apply_augmentation(img, None, 42, resolved, variant_key="1")
    assert list(out1.getdata()) == list(out2.getdata())


def test_pristine_original_untouched_when_enabled() -> None:
    # variant_key=None is the pristine original: returned byte-identical even with aug enabled.
    resolved = resolve_augmentation_config(
        {"augmentation": {"enabled": True, "preset": "easy", "branches_per_image": 2}}, {}
    )
    img = Image.new("RGB", (48, 48), color=(120, 200, 30))
    img.putpixel((3, 3), (10, 20, 30))
    original, _ = apply_augmentation(img, None, 7, resolved, variant_key=None)
    assert list(original.getdata()) == list(img.getdata())


def test_copies_are_augmented_and_distinct_per_key() -> None:
    # Each copy runs the strategy stack with a distinct deterministic seed, so copies differ
    # from the pristine original and from each other.
    resolved = resolve_augmentation_config(
        {
            "augmentation": {
                "enabled": True,
                "preset": "none",
                "branches_per_image": 2,
                "strategies": {"color_jitter": {"enabled": True, "brightness": 0.2}},
            }
        },
        {},
    )
    img = Image.new("RGB", (32, 32), color=(180, 90, 60))
    fp = augmentation_fingerprint(resolved)
    seed1 = augmentation_seed_for_image(("img.png", (32, 32)), fp, "1")
    seed2 = augmentation_seed_for_image(("img.png", (32, 32)), fp, "2")
    assert seed1 != seed2  # each copy keys a distinct deterministic seed
    copy1, _ = apply_augmentation(img, None, seed1, resolved, variant_key="1")
    copy2, _ = apply_augmentation(img, None, seed2, resolved, variant_key="2")
    assert list(copy1.getdata()) != list(img.getdata())  # augmented, not pristine
    assert list(copy1.getdata()) != list(copy2.getdata())  # distinct per seed


def test_horizontal_flip_geometry() -> None:
    # Geometric flip (probability 1.0) mirrors the image; a left-edge marker lands on the right.
    resolved = resolve_augmentation_config(
        {
            "augmentation": {
                "enabled": True,
                "preset": "none",
                "strategies": {"horizontal_flip": {"enabled": True, "probability": 1.0}},
            }
        },
        {},
    )
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.putpixel((0, 16), (0, 255, 0))
    flipped, _ = apply_augmentation(img, None, 99, resolved, variant_key="1")
    assert flipped.getpixel((31, 16)) == (0, 255, 0)
    assert flipped.getpixel((0, 16)) == (255, 0, 0)


def test_global_augmentation_inherited() -> None:
    resolved = resolve_augmentation_config(
        {"path": "/x"},
        {
            "dataset": {
                "augmentation": {"enabled": True, "preset": "easy"},
            }
        },
    )
    assert resolved["enabled"] is True
    assert resolved["preset"] == "easy"
