"""Tests for dataset augmentation (MVP)."""

from __future__ import annotations


import pytest
from PIL import Image

from rengu_flow.data.augmentation import (
    apply_augmentation,
    augmentation_fingerprint,
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


def test_enumerated_on_continuous_strategy_raises() -> None:
    with pytest.raises(AugmentationConfigError, match="enumerated"):
        resolve_augmentation_config(
            {
                "augmentation": {
                    "enabled": True,
                    "preset": "none",
                    "strategies": {
                        "color_jitter": {
                            "brightness": 0.05,
                            "sampling": "enumerated",
                        }
                    },
                }
            },
            {},
        )


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


def test_expand_enumerated_horizontal_flip() -> None:
    resolved = resolve_augmentation_config(
        {
            "augmentation": {
                "enabled": True,
                "preset": "photo_safe",
                "strategies": {
                    "horizontal_flip": {"sampling": "enumerated"},
                },
            }
        },
        {},
    )
    keys = expand_variant_keys(resolved)
    assert keys == [None, "horizontal_flip:mirror"]


def test_apply_reproducible() -> None:
    resolved = resolve_augmentation_config(
        {"augmentation": {"enabled": True, "preset": "easy"}}, {}
    )
    img = Image.new("RGB", (64, 64), color=(128, 64, 200))
    out1, _ = apply_augmentation(img, None, 42, resolved, variant_key=None)
    out2, _ = apply_augmentation(img, None, 42, resolved, variant_key=None)
    assert list(out1.getdata()) == list(out2.getdata())


def test_apply_enumerated_mirror_differs_from_identity() -> None:
    resolved = resolve_augmentation_config(
        {
            "augmentation": {
                "enabled": True,
                "preset": "none",
                "strategies": {
                    "horizontal_flip": {
                        "enabled": True,
                        "probability": 1.0,
                        "sampling": "enumerated",
                    }
                },
            }
        },
        {},
    )
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.putpixel((0, 16), (0, 255, 0))
    img.putpixel((31, 16), (0, 0, 255))
    identity, _ = apply_augmentation(
        img, None, 99, resolved, variant_key=None
    )
    mirrored, _ = apply_augmentation(
        img, None, 99, resolved, variant_key="horizontal_flip:mirror"
    )
    assert list(identity.getdata()) != list(mirrored.getdata())
    assert identity.getpixel((0, 16)) == (0, 255, 0)
    assert mirrored.getpixel((31, 16)) == (0, 255, 0)


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
