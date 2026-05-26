"""Tests for UI dataset library store."""

from pathlib import Path

import pytest

from renga_flow_ui import datasets_store


MINIMAL = """resolutions = [1024]
frame_buckets = [1]

[[directory]]
path = "/tmp/images"
num_repeats = 1
"""

SECOND = """resolutions = [512]
frame_buckets = [1]

[[directory]]
path = "/tmp/other"
num_repeats = 2
"""


def test_dataset_crud(ui_data_tmp: Path) -> None:
    datasets_store.write_dataset_text("ds_a", MINIMAL)
    assert "ds_a" in datasets_store.list_dataset_ids()
    assert datasets_store.dataset_exists("ds_a")
    assert "sdxl" not in datasets_store.read_dataset_text("ds_a")
    dup = datasets_store.duplicate_dataset("ds_a")
    assert dup != "ds_a"
    datasets_store.delete_dataset("ds_a")
    with pytest.raises(FileNotFoundError):
        datasets_store.read_dataset_text("ds_a")


def test_compose_merges_directories(ui_data_tmp: Path) -> None:
    datasets_store.write_dataset_text("one", MINIMAL)
    datasets_store.write_dataset_text("two", SECOND)
    datasets_store.compose_datasets("merged", ["one", "two"])
    cfg = datasets_store.parse_dataset_dict(datasets_store.read_dataset_text("merged"))
    assert len(cfg["directory"]) == 2
    paths = {d["path"] for d in cfg["directory"]}
    assert paths == {"/tmp/images", "/tmp/other"}
    assert cfg["resolutions"] == [1024]


def test_validate_includes_preview(ui_data_tmp: Path) -> None:
    datasets_store.write_dataset_text("v", MINIMAL)
    r = datasets_store.validate_dataset_text(MINIMAL)
    assert r["ok"] is True
    assert "preview" in r
    assert r["preview"]["directory_count"] == 1


def test_training_picker_uses_library_ref(ui_data_tmp: Path) -> None:
    datasets_store.write_dataset_text("pick_me", MINIMAL)
    picker = datasets_store.list_for_training_picker()
    lib = [p for p in picker if p["id"] == "pick_me"]
    assert len(lib) == 1
    assert lib[0]["path"].startswith("renga-flow-dataset:")
