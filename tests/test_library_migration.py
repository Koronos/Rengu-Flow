"""Tests for dataset library export/import (migration mode)."""

from pathlib import Path

from rengu_flow_ui import datasets_store, db, library_db
from rengu_flow_ui.library_migration import (
    INDEX_SECTION,
    export_library,
    import_library,
)

DATASET_TOML = "resolutions = [1024]\nframe_buckets = [1]\n\n[[directory]]\npath = '/tmp/imgs'\nnum_repeats = 1\n"


def test_export_import_round_trip(ui_data_tmp: Path, tmp_path: Path) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML, name="My set")

    out = tmp_path / "lib"
    counts = export_library(out)
    assert counts["datasets"] == 1
    assert (out / "datasets" / f"{did}.toml").is_file()

    # Wipe and restore from the export.
    db.reset_ui_database()
    assert not library_db.dataset_exists(did)
    res = import_library(out)
    assert res["datasets"] == 1

    assert library_db.dataset_exists(did)
    row = datasets_store.read_dataset_for_ui(did)
    assert row["name"] == "My set"
    assert "frame_buckets" in row["content"]
    # The display name is not embedded in the dataset TOML body.
    assert "My set" not in row["content"]


def test_exported_dataset_has_index_section(ui_data_tmp: Path, tmp_path: Path) -> None:
    import toml

    did = datasets_store.insert_dataset(DATASET_TOML, name="My set")
    out = tmp_path / "lib"
    export_library(out)
    exported = toml.loads((out / "datasets" / f"{did}.toml").read_text(encoding="utf-8"))
    assert INDEX_SECTION in exported
    assert "frame_buckets" in exported


def test_import_skips_unrecognized_and_existing(ui_data_tmp: Path, tmp_path: Path) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML, name="My set")
    out = tmp_path / "lib"
    export_library(out)

    # A stray TOML without our index section must be ignored.
    (out / "datasets" / "stray.toml").write_text('foo = "bar"\n', encoding="utf-8")

    # Existing id is skipped without --overwrite.
    res = import_library(out)
    assert res["datasets"] == 0
    assert res["skipped"] >= 2  # stray file + already-present dataset

    # With overwrite, the existing row is replaced.
    res2 = import_library(out, overwrite=True)
    assert res2["datasets"] == 1
    assert library_db.dataset_exists(did)
