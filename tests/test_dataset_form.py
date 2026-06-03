"""Tests for dataset TOML form round-trip."""

import toml

from rengu_flow_ui.dataset_form import form_to_toml, form_values_for_ui, parse_toml
from rengu_flow_ui.dataset_schema import get_dataset_schema


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


def test_renders_empty_path_directory_rows() -> None:
    form = {
        "_directories": [
            {"path": "", "num_repeats": 1},
            {"path": "/only/this", "num_repeats": 3},
        ],
        "resolutions": "[1024]",
        "frame_buckets": "[1]",
    }
    cfg = toml.loads(form_to_toml(form))
    assert len(cfg["directory"]) == 2
    assert cfg["directory"][0]["path"] == ""
    assert cfg["directory"][1]["path"] == "/only/this"
    assert cfg["directory"][1]["num_repeats"] == 3


def test_form_values_for_ui_fills_dataset_defaults() -> None:
    schema = get_dataset_schema()
    form = parse_toml("resolutions = [1024]\n\n[[directory]]\npath = '/x'\nnum_repeats = 1\n")
    assert "enable_ar_bucket" not in form
    filled = form_values_for_ui(form, schema)
    assert filled["enable_ar_bucket"] is False
    assert filled["frame_buckets"] == [1]
    assert filled["cache_shuffle_num"] == 1
    assert filled["subsample_ratio"] == 1


def test_subsample_ratio_schema_default_is_full_dataset() -> None:
    schema = get_dataset_schema()
    captions = next(s for s in schema["sections"] if s["id"] == "captions")
    by_path = {f["path"]: f for f in captions["fields"]}
    assert by_path["subsample_ratio"]["default"] == 1
    assert "show_if_set" not in by_path["subsample_ratio"]
    dir_by_path = {f["path"]: f for f in schema["directory_fields"]}
    assert dir_by_path["subsample_ratio"]["default"] == 1


def test_max_images_schema_fields() -> None:
    schema = get_dataset_schema()
    captions = next(s for s in schema["sections"] if s["id"] == "captions")
    by_path = {f["path"]: f for f in captions["fields"]}
    # Global default field exists, integer with no default (unset == no cap).
    assert by_path["max_images"]["type"] == "integer"
    assert by_path["max_images"]["min"] == 1
    assert "default" not in by_path["max_images"]
    assert by_path["static_sampling"]["default"] is False

    dir_by_path = {f["path"]: f for f in schema["directory_fields"]}
    assert dir_by_path["max_images"]["type"] == "integer"
    assert dir_by_path["max_images"]["min"] == 1
    assert dir_by_path["max_images"].get("show_if_set") is True
    # Static flag governs whichever limiter is active; it's an optional override.
    assert dir_by_path["static_sampling"]["type"] == "boolean"
    assert dir_by_path["static_sampling"].get("show_if_set") is True
    assert dir_by_path["static_sampling"]["default"] is False


def test_cache_shuffle_schema_default_and_shuffle_tags_gating() -> None:
    schema = get_dataset_schema()
    captions = next(s for s in schema["sections"] if s["id"] == "captions")
    by_path = {f["path"]: f for f in captions["fields"]}
    assert by_path["cache_shuffle_num"]["default"] == 1
    assert by_path["cache_shuffle_num"]["show_when_field"] == "shuffle_tags"
    assert by_path["cache_shuffle_delimiter"]["show_when_field"] == "shuffle_tags"
    dir_by_path = {f["path"]: f for f in schema["directory_fields"]}
    assert dir_by_path["cache_shuffle_num"]["default"] == 1
    assert dir_by_path["cache_shuffle_num"]["show_when_field"] == "shuffle_tags"


def test_parse_toml_to_form_skips_defaults_by_default() -> None:
    from rengu_flow_ui.dataset_form import parse_toml_to_form

    form, _notes = parse_toml_to_form(
        "resolutions = [768]\nframe_buckets = [1]\n\n[[directory]]\npath = '/x'\nnum_repeats = 1\n"
    )
    assert form["resolutions"] == [768]
    assert "enable_ar_bucket" not in form
    assert "shuffle_tags" not in form


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


def test_tag_dropout_rules_roundtrip() -> None:
    raw = """
resolutions = [1024]
frame_buckets = [1]
tag_dropout_enabled = true
tag_dropout_probability = 0.25

[[tag_dropout_rules]]
tags = ["hero", "sidekick"]
drop_probability = 0.1

[[tag_dropout_rules]]
tags_file = "extras.txt"
drop_probability = 0.5

[[directory]]
path = "/data/a"
num_repeats = 1
"""
    form = parse_toml(raw)
    assert form["tag_dropout_enabled"] is True
    rules_raw = form["tag_dropout_rules"]
    if isinstance(rules_raw, str):
        import json

        rules = json.loads(rules_raw)
    else:
        rules = rules_raw
    assert len(rules) == 2
    assert rules[0]["tags"] == ["hero", "sidekick"]
    assert rules[1]["tags_file"] == "extras.txt"

    form["tag_dropout_rules"] = [
        {"tags": ["hero", "sidekick"], "drop_probability": 0.1},
        {"tags_file": "extras.txt", "drop_probability": 0.5},
    ]
    cfg = toml.loads(form_to_toml(form))
    assert len(cfg["tag_dropout_rules"]) == 2
    assert cfg["tag_dropout_rules"][0]["drop_probability"] == 0.1


def test_form_to_toml_omits_incomplete_tag_dropout_rules() -> None:
    form = parse_toml(
        """
resolutions = [1024]
frame_buckets = [1]
tag_dropout_enabled = true
"""
    )
    form["tag_dropout_rules"] = [{"tags": [], "drop_probability": 0.1}]
    cfg = toml.loads(form_to_toml(form))
    assert "tag_dropout_rules" not in cfg
