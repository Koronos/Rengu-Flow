"""Tests for library export/import (migration mode)."""

from pathlib import Path

import toml

from rengu_flow_ui import configs_store, datasets_store, db, library_db
from rengu_flow_ui.library_migration import (
    INDEX_SECTION,
    export_library,
    import_library,
)
from rengu_flow.config.validation import validate_config

CONFIG_TOML = (
    'dataset = "examples/minimal_dataset.toml"\n'
    'run_name = "demo"\n\n'
    '[model]\ntype = "sdxl"\ndtype = "bfloat16"\ncheckpoint_path = "/tmp/x.safetensors"\n\n'
    '[optimizer]\ntype = "adamw"\nlr = 1.0e-4\n'
)
DATASET_TOML = "resolutions = [1024]\nframe_buckets = [1]\n\n[[directory]]\npath = '/tmp/imgs'\nnum_repeats = 1\n"


def test_export_import_round_trip(ui_data_tmp: Path, tmp_path: Path) -> None:
    cid = configs_store.insert_config(CONFIG_TOML)
    did = datasets_store.insert_dataset(DATASET_TOML, name="My set")

    out = tmp_path / "lib"
    counts = export_library(out)
    assert counts == {"configs": 1, "datasets": 1}
    assert (out / "configs" / f"{cid}.toml").is_file()
    assert (out / "datasets" / f"{did}.toml").is_file()

    # Wipe and restore from the export.
    db.reset_ui_database()
    assert not library_db.config_exists(cid)
    res = import_library(out)
    assert res["configs"] == 1
    assert res["datasets"] == 1

    assert library_db.config_exists(cid)
    assert 'type = "sdxl"' in library_db.read_config_text(cid)
    row = datasets_store.read_dataset_for_ui(did)
    assert row["name"] == "My set"
    assert "frame_buckets" in row["content"]
    # The display name is not embedded in the dataset TOML body.
    assert "My set" not in row["content"]


def test_exported_config_still_validates(ui_data_tmp: Path, tmp_path: Path) -> None:
    cid = configs_store.insert_config(CONFIG_TOML)
    out = tmp_path / "lib"
    export_library(out)
    exported = toml.loads((out / "configs" / f"{cid}.toml").read_text(encoding="utf-8"))
    # The index section is present but ignored by the validator.
    assert INDEX_SECTION in exported
    validate_config(exported)  # must not raise


def test_import_skips_unrecognized_and_existing(ui_data_tmp: Path, tmp_path: Path) -> None:
    cid = configs_store.insert_config(CONFIG_TOML)
    out = tmp_path / "lib"
    export_library(out)

    # A stray TOML without our index section must be ignored.
    (out / "configs" / "stray.toml").write_text('foo = "bar"\n', encoding="utf-8")

    # Existing id is skipped without --overwrite.
    res = import_library(out)
    assert res["configs"] == 0
    assert res["skipped"] >= 2  # stray file + already-present config

    # With overwrite, the existing row is replaced.
    res2 = import_library(out, overwrite=True)
    assert res2["configs"] == 1
    assert library_db.config_exists(cid)
