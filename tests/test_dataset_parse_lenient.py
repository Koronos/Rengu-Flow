"""Lenient dataset TOML parsing for the UI (does not rewrite stored files)."""

from renga_flow_ui.dataset_form import form_to_toml, parse_toml_to_form
import toml


def test_invalid_toml_returns_empty_form_with_note() -> None:
    form, notes = parse_toml_to_form("not valid {{{ toml")
    assert notes
    assert "Could not parse TOML" in notes[0]
    assert form["_directories"] == []


def test_unknown_keys_stay_out_of_form_but_noted() -> None:
    raw = """
unknown_key = true
resolutions = [1024]
frame_buckets = [1]

[[directory]]
path = "/data/a"
num_repeats = 1
"""
    form, notes = parse_toml_to_form(raw)
    assert "unknown_key" not in form
    assert any("not supported in the form builder" in w for w in notes)
    assert form["_directories"][0]["path"] == "/data/a"


def test_includes_directory_without_path() -> None:
    raw = """
resolutions = [1024]
frame_buckets = [1]

[[directory]]
num_repeats = 2

[[directory]]
path = "/ok"
num_repeats = 1
"""
    form, notes = parse_toml_to_form(raw)
    assert len(form["_directories"]) == 2
    assert form["_directories"][0]["path"] == ""
    assert form["_directories"][0]["num_repeats"] == 2
    assert form["_directories"][1]["path"] == "/ok"
    assert any("path is empty" in w for w in notes)


def test_form_to_toml_preserves_empty_path_directory() -> None:
    form = {
        "_directories": [
            {"path": "", "num_repeats": 2},
            {"path": "/only/this", "num_repeats": 3},
        ],
        "resolutions": [1024],
        "frame_buckets": [1],
    }
    cfg = toml.loads(form_to_toml(form))
    assert len(cfg["directory"]) == 2
    assert cfg["directory"][0]["path"] == ""
    assert cfg["directory"][0]["num_repeats"] == 2
    assert cfg["directory"][1]["path"] == "/only/this"
