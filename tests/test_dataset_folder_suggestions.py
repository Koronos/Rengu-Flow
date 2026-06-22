"""Tests for cross-dataset folder path suggestions."""

from __future__ import annotations

from rengu_flow_ui import datasets_store
from rengu_flow_ui.dataset_folder_suggestions import collect_folder_suggestions


def test_collect_folder_suggestions_existing_and_missing(ui_data_tmp, tmp_path) -> None:
    existing = tmp_path / "images_a"
    existing.mkdir()
    (existing / "a.png").write_bytes(b"x")

    missing = tmp_path / "gone_folder"

    ds1 = (
        "resolutions = [512]\nframe_buckets = [1]\n\n"
        f'[[directory]]\npath = "{existing.as_posix()}"\nnum_repeats = 1\n'
    )
    ds2 = (
        "resolutions = [512]\nframe_buckets = [1]\n\n"
        f'[[directory]]\npath = "{existing.as_posix()}"\nnum_repeats = 2\n\n'
        f'[[directory]]\npath = "{missing.as_posix()}"\nnum_repeats = 1\n'
    )
    ds_a = datasets_store.insert_dataset(ds1)
    datasets_store.insert_dataset(ds2)

    out = collect_folder_suggestions(exclude_dataset_id=99999)
    paths = {s["path"] for s in out["suggestions"]}
    assert existing.as_posix() in paths or str(existing.resolve()) in paths
    assert any(s["image_count"] >= 1 for s in out["suggestions"])
    assert any(s.get("preview_token") for s in out["suggestions"])

    missing_paths = {m["path"] for m in out["missing"]}
    assert missing.as_posix() in missing_paths

    excluded = collect_folder_suggestions(exclude_dataset_id=ds_a)
    for s in excluded["suggestions"]:
        assert ds_a not in s["source_datasets"]
