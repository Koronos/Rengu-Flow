"""Versioned smoke_cc0 fixture: 12 images, manifest, dump_dataset."""

import json
from pathlib import Path

import pytest

from renga_flow.data.dump_dataset import dump_dataset

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smoke_cc0"
IMAGES = FIXTURE / "images"
MANIFEST = FIXTURE / "manifest.json"
DATASET_TOML = Path(__file__).resolve().parent.parent / "examples" / "smoke_cc0_dataset.toml"


def test_manifest_has_twelve_entries():
    entries = json.loads(MANIFEST.read_text())
    assert len(entries) == 12
    for entry in entries:
        assert "upstream_png" in entry
        assert "output_stem" in entry
        assert "caption" in entry


def test_images_jpg_and_txt_pairs_exist():
    entries = json.loads(MANIFEST.read_text())
    for entry in entries:
        stem = entry["output_stem"]
        jpg = IMAGES / f"{stem}.jpg"
        txt = IMAGES / f"{stem}.txt"
        assert jpg.is_file(), stem
        assert txt.is_file(), stem
        assert txt.read_text(encoding="utf-8").strip() == entry["caption"].strip()


def test_license_and_attribution_present():
    assert (FIXTURE / "LICENSE").is_file()
    text = (FIXTURE / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "gb82" in text.lower()
    assert "CC0" in text or "cc0" in text.lower()


def test_dump_dataset_smoke_cc0(capsys):
    records = dump_dataset(DATASET_TOML)
    assert len(records) == 12
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 12
