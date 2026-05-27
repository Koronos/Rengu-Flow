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
    ds_a = datasets_store.insert_dataset(MINIMAL)
    assert ds_a in datasets_store.list_dataset_ids()
    assert datasets_store.dataset_exists(ds_a)
    assert "sdxl" not in datasets_store.read_dataset_text(ds_a)
    dup = datasets_store.duplicate_dataset(ds_a)
    assert dup != ds_a
    datasets_store.delete_dataset(ds_a)
    with pytest.raises(FileNotFoundError):
        datasets_store.read_dataset_text(ds_a)


def test_compose_merges_directories(ui_data_tmp: Path) -> None:
    one = datasets_store.insert_dataset(MINIMAL)
    two = datasets_store.insert_dataset(SECOND)
    merged = datasets_store.compose_datasets([one, two])
    cfg = datasets_store.parse_dataset_dict(datasets_store.read_dataset_text(merged))
    assert len(cfg["directory"]) == 2
    paths = {d["path"] for d in cfg["directory"]}
    assert paths == {"/tmp/images", "/tmp/other"}
    assert cfg["resolutions"] == [1024]


def test_validate_includes_preview(ui_data_tmp: Path) -> None:
    datasets_store.insert_dataset(MINIMAL)
    r = datasets_store.validate_dataset_text(MINIMAL)
    assert r["ok"] is True
    assert "preview" in r
    assert r["preview"]["directory_count"] == 1


def test_create_dataset_allocates_id(ui_data_tmp: Path) -> None:
    cid = datasets_store.create_dataset(MINIMAL)
    assert isinstance(cid, int)
    assert cid > 0
    assert datasets_store.dataset_exists(cid)


def test_compose_allocates_id(ui_data_tmp: Path) -> None:
    one = datasets_store.insert_dataset(MINIMAL)
    tid = datasets_store.compose_datasets([one])
    assert isinstance(tid, int)
    assert datasets_store.dataset_exists(tid)


def test_training_picker_uses_library_ref(ui_data_tmp: Path) -> None:
    did = datasets_store.insert_dataset(MINIMAL)
    picker = datasets_store.list_for_training_picker()
    lib = [p for p in picker if p["id"] == str(did)]
    assert len(lib) == 1
    assert lib[0]["path"].startswith("renga-flow-dataset:")


def test_training_picker_library_only(ui_data_tmp: Path) -> None:
    """Repo examples/*.toml must not appear in pickers (import into library first)."""
    datasets_store.insert_dataset(MINIMAL)
    picker = datasets_store.list_for_training_picker()
    assert picker
    assert all(p["path"].startswith("renga-flow-dataset:") for p in picker)
    assert all("(example" not in p.get("label", "").lower() for p in picker)


def test_dataset_name_stored_separately_from_toml(ui_data_tmp: Path) -> None:
    did = datasets_store.insert_dataset(MINIMAL, name="My portraits")
    row = datasets_store.read_dataset_for_ui(did)
    assert row["name"] == "My portraits"
    assert "My portraits" not in row["content"]
    datasets_store.update_dataset_text(did, MINIMAL, name="Renamed set")
    row2 = datasets_store.read_dataset_for_ui(did)
    assert row2["name"] == "Renamed set"
