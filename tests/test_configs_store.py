"""Tests for UI training config store and staging."""

from pathlib import Path

import pytest
import toml

from rengu_flow_ui import configs_store, datasets_store, library_db


MINIMAL_TOML = """
dataset = "rengu-flow-dataset:my_dataset"
output_dir = "output"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"

[optimizer]
type = "adamw"
lr = 1.0e-4

epochs = 1
micro_batch_size_per_gpu = 1
"""

DATASET_TOML = """resolutions = [1024]
frame_buckets = [1]

[[directory]]
path = "/tmp/img"
num_repeats = 1
"""


def test_safe_id_sanitizes() -> None:
    assert configs_store._safe_id("  foo bar!!  ") == "foo_bar"


def test_config_library_crud(ui_data_tmp: Path) -> None:
    cid = configs_store.insert_config(MINIMAL_TOML)
    assert cid in configs_store.list_config_ids()
    assert configs_store.config_exists(cid)
    dup = configs_store.duplicate_config(cid)
    assert dup != cid
    configs_store.delete_config(cid)
    with pytest.raises(FileNotFoundError):
        configs_store.read_config_text(cid)


def test_materialize_staging_resolves_library_dataset(ui_data_tmp: Path) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML)
    ref = library_db.dataset_library_ref(did)
    content = MINIMAL_TOML.replace("rengu-flow-dataset:my_dataset", ref)
    cid = configs_store.insert_config(content)

    staging = configs_store.materialize_staging(
        configs_store.read_config_text(cid),
        "job-abc",
    )
    assert staging.name == "train.toml"
    cfg = toml.loads(staging.read_text(encoding="utf-8"))
    assert Path(cfg["dataset"]).is_absolute()
    assert (staging.parent / f"{did}.dataset.toml").is_file()


def test_materialize_staging_absolute_dataset_unchanged(ui_data_tmp: Path) -> None:
    abs_ds = ui_data_tmp / "abs.toml"
    abs_ds.write_text(DATASET_TOML, encoding="utf-8")
    content = MINIMAL_TOML.replace(
        'dataset = "rengu-flow-dataset:my_dataset"',
        f'dataset = "{abs_ds}"',
    )
    out = configs_store.materialize_staging(content, "job-abs")
    cfg = toml.loads(out.read_text(encoding="utf-8"))
    assert cfg["dataset"] == str(abs_ds.resolve())


def test_validate_rejects_bad_toml() -> None:
    assert configs_store.validate_toml_text("not valid {{{")["ok"] is False


def test_materialize_staging_merges_multiple_datasets(ui_data_tmp: Path) -> None:
    did_a = datasets_store.insert_dataset(DATASET_TOML)
    did_b = datasets_store.insert_dataset(
        DATASET_TOML.replace('path = "/tmp/img"', 'path = "/tmp/img2"')
    )
    ref_a = library_db.dataset_library_ref(did_a)
    ref_b = library_db.dataset_library_ref(did_b)
    content = MINIMAL_TOML.replace(
        'dataset = "rengu-flow-dataset:my_dataset"',
        f"dataset = [{ref_a!r}, {ref_b!r}]",
    )
    out = configs_store.materialize_staging(content, "job-merge")
    cfg = toml.loads(out.read_text(encoding="utf-8"))
    merged_path = Path(cfg["dataset"])
    assert merged_path.is_file()
    merged = toml.loads(merged_path.read_text(encoding="utf-8"))
    assert len(merged["directory"]) == 2


def test_validate_accepts_dataset_list(ui_data_tmp: Path, minimal_config: dict) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML)
    ref = library_db.dataset_library_ref(did)
    minimal_config["dataset"] = [ref, ref]
    r = configs_store.validate_toml_text(toml.dumps(minimal_config))
    assert r["ok"] is True


def test_validate_accepts_minimal(ui_data_tmp: Path, minimal_config: dict) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML)
    minimal_config["dataset"] = library_db.dataset_library_ref(did)
    text = toml.dumps(minimal_config)
    r = configs_store.validate_toml_text(text)
    assert r["ok"] is True
    assert "config" in r
    assert isinstance(r["config"]["model"]["dtype"], str)
