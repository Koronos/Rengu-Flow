"""Tests for dataset TOML form round-trip."""

import toml

from renga_flow_ui.dataset_form import form_to_toml, form_values_for_ui, parse_toml
from renga_flow_ui.dataset_schema import get_dataset_schema


def test_parse_and_render_directories() -> None:
    raw = """
resolutions = [1024, 512]
frame_buckets = [1]

[[directory]]
path = "/data/a"
num_repeats = 2
directory_caption = "style: "

[[directory]]
path = "/data/b"
num_repeats = 1
"""
    form = parse_toml(raw)
    assert len(form["_directories"]) == 2
    assert form["_directories"][0]["path"] == "/data/a"
    out = form_to_toml(form)
    cfg = toml.loads(out)
    assert len(cfg["directory"]) == 2
    assert cfg["resolutions"] == [1024, 512]


def test_skips_empty_directory_rows() -> None:
    form = {
        "_directories": [
            {"path": "", "num_repeats": 1},
            {"path": "/only/this", "num_repeats": 3},
        ],
        "resolutions": "[1024]",
        "frame_buckets": "[1]",
    }
    cfg = toml.loads(form_to_toml(form))
    assert len(cfg["directory"]) == 1
    assert cfg["directory"][0]["path"] == "/only/this"
    assert cfg["directory"][0]["num_repeats"] == 3


def test_form_values_for_ui_fills_dataset_defaults() -> None:
    schema = get_dataset_schema()
    form = parse_toml("resolutions = [1024]\n\n[[directory]]\npath = '/x'\nnum_repeats = 1\n")
    assert "enable_ar_bucket" not in form
    filled = form_values_for_ui(form, schema)
    assert filled["enable_ar_bucket"] is False
    assert filled["frame_buckets"] == [1]


def test_parse_keeps_number_lists_as_arrays() -> None:
    form = parse_toml(
        "resolutions = [1024]\nenable_ar_bucket = true\nar_buckets = [1.0, 1.5]\n"
        "frame_buckets = [1]\n\n[[directory]]\npath = '/x'\nnum_repeats = 1\n"
    )
    assert form["ar_buckets"] == [1.0, 1.5]


def test_directory_per_folder_shuffle_override() -> None:
    raw = """
resolutions = [1024]
frame_buckets = [1]
shuffle_tags = false

[[directory]]
path = "/data/a"
num_repeats = 1
shuffle_tags = true
cache_shuffle_num = 3

[[directory]]
path = "/data/b"
num_repeats = 2
"""
    form = parse_toml(raw)
    assert form["_directories"][0]["shuffle_tags"] is True
    assert form["_directories"][0]["cache_shuffle_num"] == 3
    cfg = toml.loads(form_to_toml(form))
    assert cfg["directory"][0]["shuffle_tags"] is True
    assert cfg["directory"][0]["cache_shuffle_num"] == 3
    assert "shuffle_tags" not in cfg["directory"][1]


def test_parse_keeps_integer_lists_as_arrays() -> None:
    form = parse_toml("resolutions = [1024, 512]\nframe_buckets = [1, 9]\n")
    assert form["resolutions"] == [1024, 512]
    assert form["frame_buckets"] == [1, 9]
    assert isinstance(form["resolutions"], list)


def test_no_adapter_key_in_dataset_toml() -> None:
    form = parse_toml("resolutions = [1024]\nframe_buckets = [1]\n")
    assert "_has_adapter" not in form
    rendered = form_to_toml(form)
    assert "adapter" not in rendered
