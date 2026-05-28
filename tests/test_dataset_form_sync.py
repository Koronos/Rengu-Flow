"""Form ↔ TOML sync for dataset editor (render/parse API)."""

from __future__ import annotations

import toml

from renga_flow_ui.dataset_form import form_to_toml, parse_toml_to_form


def _roundtrip(form: dict) -> dict:
    return toml.loads(form_to_toml(form))


def test_global_integer_lists_roundtrip() -> None:
    form, _ = parse_toml_to_form(
        "resolutions = [768, 1024]\nframe_buckets = [1, 9]\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["resolutions"] = [512, 768]
    cfg = _roundtrip(form)
    assert cfg["resolutions"] == [512, 768]
    assert cfg["frame_buckets"] == [1, 9]


def test_global_booleans_and_numbers_roundtrip() -> None:
    form, _ = parse_toml_to_form(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["enable_ar_bucket"] = True
    form["min_ar"] = 0.6
    form["max_ar"] = 1.8
    form["num_ar_buckets"] = 8
    form["shuffle_tags"] = True
    form["cache_shuffle_num"] = 2
    form["subsample_ratio"] = 0.25
    cfg = _roundtrip(form)
    assert cfg["enable_ar_bucket"] is True
    assert cfg["min_ar"] == 0.6
    assert cfg["max_ar"] == 1.8
    assert cfg["num_ar_buckets"] == 8
    assert cfg["shuffle_tags"] is True
    assert cfg["cache_shuffle_num"] == 2
    assert cfg["subsample_ratio"] == 0.25


def test_global_ar_buckets_and_size_buckets_roundtrip() -> None:
    form, _ = parse_toml_to_form(
        "resolutions = [1024]\nframe_buckets = [1]\nenable_ar_bucket = true\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["ar_buckets"] = [0.75, 1.0, 1.33]
    form["size_buckets"] = [[512, 512, 1], [768, 768, 1]]
    cfg = _roundtrip(form)
    assert cfg["ar_buckets"] == [0.75, 1.0, 1.33]
    assert cfg["size_buckets"] == [[512, 512, 1], [768, 768, 1]]


def test_dataset_augmentation_roundtrip() -> None:
    import json

    form, _ = parse_toml_to_form(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["_dataset_augmentation"] = json.dumps(
        {"enabled": True, "preset": "photo_safe"}, indent=2
    )
    cfg = _roundtrip(form)
    assert cfg["dataset"]["augmentation"]["enabled"] is True
    assert cfg["dataset"]["augmentation"]["preset"] == "photo_safe"


def test_form_to_toml_tolerates_invalid_directories_field() -> None:
    form = {
        "resolutions": [1024],
        "frame_buckets": [1],
        "_directories": "not-json",
    }
    cfg = _roundtrip(form)
    assert cfg.get("directory") is None or cfg.get("directory") == []


def test_directory_boolean_and_ar_overrides_roundtrip() -> None:
    form, _ = parse_toml_to_form(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["_directories"] = [
        {
            "path": "/my/images",
            "num_repeats": 1,
            "shuffle_tags": True,
            "shuffle_metadata": False,
            "enable_ar_bucket": True,
            "ar_buckets": [1.0, 1.5],
        }
    ]
    cfg = _roundtrip(form)
    d = cfg["directory"][0]
    assert d["shuffle_tags"] is True
    assert d["shuffle_metadata"] is False
    assert d["enable_ar_bucket"] is True
    assert d["ar_buckets"] == [1.0, 1.5]


def test_directory_subsample_ratio_override_roundtrip() -> None:
    form, _ = parse_toml_to_form(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["_directories"] = [
        {
            "path": "/my/images",
            "num_repeats": 1,
            "subsample_ratio": 0.5,
        }
    ]
    cfg = _roundtrip(form)
    assert cfg["directory"][0]["subsample_ratio"] == 0.5


def test_directory_subsample_ratio_omitted_when_same_as_global() -> None:
    form, _ = parse_toml_to_form(
        "resolutions = [1024]\nframe_buckets = [1]\nsubsample_ratio = 0.25\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["subsample_ratio"] = 0.25
    form["_directories"] = [
        {
            "path": "/my/images",
            "num_repeats": 1,
            "subsample_ratio": 0.25,
        }
    ]
    cfg = _roundtrip(form)
    assert cfg["subsample_ratio"] == 0.25
    assert "subsample_ratio" not in cfg["directory"][0]


def test_subsample_ratio_omitted_from_toml_when_full_dataset() -> None:
    form, _ = parse_toml_to_form(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["subsample_ratio"] = 1
    cfg = _roundtrip(form)
    assert "subsample_ratio" not in cfg


def test_directory_path_and_overrides_roundtrip() -> None:
    form, _ = parse_toml_to_form(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/data/x'\nnum_repeats = 1\n"
    )
    form["_directories"] = [
        {
            "path": "/my/images",
            "num_repeats": 3,
            "directory_caption": "style: ",
            "shuffle_tags": True,
            "resolutions": [512],
        }
    ]
    cfg = _roundtrip(form)
    d = cfg["directory"][0]
    assert d["path"] == "/my/images"
    assert d["num_repeats"] == 3
    assert d["directory_caption"] == "style: "
    assert d["shuffle_tags"] is True
    assert d["resolutions"] == [512]
