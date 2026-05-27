"""Dataset form round-trip for augmentation tables."""

import json

import toml

from renga_flow_ui.dataset_form import form_to_toml, parse_toml


def test_parse_and_render_directory_augmentation() -> None:
    raw = """
resolutions = [1024]
frame_buckets = [1]

[dataset.augmentation]
enabled = false
preset = "none"

[[directory]]
path = "/data/a"
num_repeats = 1

[directory.augmentation]
enabled = true
preset = "easy"
seed_mode = "deterministic_per_image"
"""
    form = parse_toml(raw)
    row = form["_directories"][0]
    assert row["augmentation"]["enabled"] is True
    assert row["augmentation"]["preset"] == "easy"
    out = form_to_toml(form)
    cfg = toml.loads(out)
    assert cfg["directory"][0]["augmentation"]["preset"] == "easy"
    assert cfg["dataset"]["augmentation"]["enabled"] is False


def test_global_augmentation_json_round_trip() -> None:
    form = parse_toml(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/x'\nnum_repeats = 1\n"
    )
    form["_dataset_augmentation"] = json.dumps(
        {"enabled": True, "preset": "photo_safe"}, indent=2
    )
    cfg = toml.loads(form_to_toml(form))
    assert cfg["dataset"]["augmentation"]["preset"] == "photo_safe"
