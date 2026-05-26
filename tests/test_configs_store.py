"""Tests for UI training config store and staging."""

from pathlib import Path

import pytest
import toml

from renga_flow_ui import configs_store, datasets_store, library_db


MINIMAL_TOML = """
dataset = "renga-flow-dataset:my_dataset"
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
    configs_store.write_config_text("run_cfg", MINIMAL_TOML)
    assert "run_cfg" in configs_store.list_config_ids()
    assert configs_store.config_exists("run_cfg")
    dup = configs_store.duplicate_config("run_cfg")
    assert dup != "run_cfg"
    configs_store.delete_config("run_cfg")
    with pytest.raises(FileNotFoundError):
        configs_store.read_config_text("run_cfg")


def test_materialize_staging_resolves_library_dataset(ui_data_tmp: Path) -> None:
    datasets_store.write_dataset_text("my_dataset", DATASET_TOML)
    configs_store.write_config_text("run_cfg", MINIMAL_TOML)

    staging = configs_store.materialize_staging(
        configs_store.read_config_text("run_cfg"),
        "job-abc",
    )
    assert staging.name == "train.toml"
    cfg = toml.loads(staging.read_text(encoding="utf-8"))
    assert Path(cfg["dataset"]).is_absolute()
    assert (staging.parent / "my_dataset.dataset.toml").is_file()


def test_materialize_staging_absolute_dataset_unchanged(ui_data_tmp: Path) -> None:
    abs_ds = ui_data_tmp / "abs.toml"
    abs_ds.write_text(DATASET_TOML, encoding="utf-8")
    content = MINIMAL_TOML.replace(
        'dataset = "renga-flow-dataset:my_dataset"',
        f'dataset = "{abs_ds}"',
    )
    out = configs_store.materialize_staging(content, "job-abs")
    cfg = toml.loads(out.read_text(encoding="utf-8"))
    assert cfg["dataset"] == str(abs_ds.resolve())


def test_validate_rejects_bad_toml() -> None:
    assert configs_store.validate_toml_text("not valid {{{")["ok"] is False


def test_validate_accepts_minimal(minimal_config: dict) -> None:
    minimal_config["dataset"] = library_db.dataset_library_ref("ds1")
    datasets_store.write_dataset_text("ds1", DATASET_TOML)
    text = toml.dumps(minimal_config)
    r = configs_store.validate_toml_text(text)
    assert r["ok"] is True
    assert "config" in r
    assert isinstance(r["config"]["model"]["dtype"], str)
