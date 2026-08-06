"""Resuming a run must not rebuild the dataset metadata.

Building metadata enumerates the folder and reads every image header, so it used to dominate the
startup of every resume (and printed a "computing metadata" line each time) even when nothing had
changed. A cheap signature (scandir + config) now decides, so the rebuild happens only when the
source files or the dataset config actually changed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rengu_flow.data.dataset import DirectoryDataset

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)

DATASET_CONFIG = {
    "resolutions": [512],
    "frame_buckets": [1],
    "min_ar": 0.5,
    "max_ar": 2.0,
    "num_ar_buckets": 4,
}


@pytest.fixture
def img_dir(tmp_path):
    assert FIXTURE_JPG.is_file(), "smoke_cc0 fixture required"
    d = tmp_path / "images"
    d.mkdir()
    for stem in ("a", "b"):
        shutil.copy(FIXTURE_JPG, d / f"{stem}.jpg")
        (d / f"{stem}.txt").write_text(f"a photo of {stem}", encoding="utf-8")
    return d


def _dd(img_dir: Path, cache_root: Path, **dataset_extra) -> DirectoryDataset:
    return DirectoryDataset(
        {"path": str(img_dir), "num_repeats": 1},
        {**DATASET_CONFIG, **dataset_extra},
        "sdxl",
        skip_dataset_validation=True,
        training_config={"cache_root": str(cache_root)},
    )


def _cache_counting_rebuilds(dd: DirectoryDataset, monkeypatch) -> int:
    """Run cache_metadata, returning how many times the expensive rebuild ran."""
    calls = {"n": 0}
    real = DirectoryDataset._group_metadata_and_save_to_disk

    def spy(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(DirectoryDataset, "_group_metadata_and_save_to_disk", spy)
    dd.cache_metadata()
    monkeypatch.undo()
    return calls["n"]


def test_second_run_does_not_rebuild_metadata(tmp_path, img_dir, monkeypatch):
    cache_root = tmp_path / "cache"
    assert _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch) == 1  # cold build

    # A resume with nothing changed: no enumeration, no header reads, no regroup.
    assert _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch) == 0


def test_added_image_rebuilds_metadata(tmp_path, img_dir, monkeypatch):
    cache_root = tmp_path / "cache"
    _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch)

    shutil.copy(FIXTURE_JPG, img_dir / "c.jpg")
    assert _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch) == 1


def test_removed_image_rebuilds_metadata(tmp_path, img_dir, monkeypatch):
    cache_root = tmp_path / "cache"
    _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch)

    (img_dir / "b.jpg").unlink()
    assert _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch) == 1


def test_edited_caption_rebuilds_metadata(tmp_path, img_dir, monkeypatch):
    cache_root = tmp_path / "cache"
    _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch)

    # Same file list, different caption content: size and mtime both move.
    (img_dir / "a.txt").write_text("a completely different caption", encoding="utf-8")
    assert _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch) == 1


def test_changed_bucketing_config_rebuilds_metadata(tmp_path, img_dir, monkeypatch):
    """No file changed, but the buckets did — the cached metadata is grouped under old rules."""
    cache_root = tmp_path / "cache"
    _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch)

    changed = _dd(img_dir, cache_root, num_ar_buckets=7)
    assert _cache_counting_rebuilds(changed, monkeypatch) == 1


def test_regenerate_cache_always_rebuilds(tmp_path, img_dir, monkeypatch):
    cache_root = tmp_path / "cache"
    _cache_counting_rebuilds(_dd(img_dir, cache_root), monkeypatch)

    dd = _dd(img_dir, cache_root)
    calls = {"n": 0}
    real = DirectoryDataset._group_metadata_and_save_to_disk
    monkeypatch.setattr(
        DirectoryDataset,
        "_group_metadata_and_save_to_disk",
        lambda self, *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(self, *a, **k))[1],
    )
    dd.cache_metadata(regenerate_cache=True)
    assert calls["n"] == 1
