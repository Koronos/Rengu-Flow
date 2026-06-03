"""Aspect-ratio bucket computation in DirectoryDataset."""

from __future__ import annotations

import pytest

from rengu_flow.data.dataset import DirectoryDataset


def test_ar_buckets_without_explicit_keys_falls_back_to_defaults(tmp_path):
    """enable_ar_bucket on, but min_ar/max_ar/num_ar_buckets and ar_buckets all unset.

    Regression: dataset.py read ``dataset_config["min_ar"]`` directly and raised KeyError when a
    config enabled AR bucketing without spelling those keys out (and provided no explicit
    ar_buckets). It must fall back to the same defaults the UI schema advertises (0.5 .. 2.0,
    12 buckets) instead of crashing.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    dd = DirectoryDataset(
        {"path": str(img_dir), "num_repeats": 1, "shuffle_metadata": False},
        {"resolutions": [512], "frame_buckets": [1], "enable_ar_bucket": True},
        "sdxl",
        skip_dataset_validation=True,
    )
    assert dd.enable_ar_bucket is True
    assert len(dd.ars) == 12
    assert float(dd.ars.min()) == pytest.approx(0.5)
    assert float(dd.ars.max()) == pytest.approx(2.0)


def test_ar_buckets_explicit_keys_still_respected(tmp_path):
    """Explicit min_ar/max_ar/num_ar_buckets keep overriding the defaults."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    dd = DirectoryDataset(
        {"path": str(img_dir), "num_repeats": 1, "shuffle_metadata": False},
        {
            "resolutions": [512],
            "frame_buckets": [1],
            "enable_ar_bucket": True,
            "min_ar": 0.25,
            "max_ar": 4.0,
            "num_ar_buckets": 5,
        },
        "sdxl",
        skip_dataset_validation=True,
    )
    assert len(dd.ars) == 5
    assert float(dd.ars.min()) == pytest.approx(0.25)
    assert float(dd.ars.max()) == pytest.approx(4.0)
