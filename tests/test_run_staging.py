"""Tests for run TOML validation and job staging (run_staging)."""

from pathlib import Path

from rengu_flow_ui import datasets_store, library_db, run_staging
import toml


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


def test_materialize_staging_resolves_library_dataset(ui_data_tmp: Path) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML)
    ref = library_db.dataset_library_ref(did)
    content = MINIMAL_TOML.replace("rengu-flow-dataset:my_dataset", ref)

    staging = run_staging.materialize_staging(content, "job-abc")
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
    out = run_staging.materialize_staging(content, "job-abs")
    cfg = toml.loads(out.read_text(encoding="utf-8"))
    assert cfg["dataset"] == str(abs_ds.resolve())


def test_materialize_staging_does_not_persist_defaults(ui_data_tmp: Path) -> None:
    """RF-03: set_config_defaults() mutates the config in place — it converts dtype strings
    into torch.dtype objects (serialized as "torch.bfloat16") and injects alpha=rank into
    [adapter]. Persisting those into the staged train.toml makes the trainer abort
    (KeyError 'torch.bfloat16' / "Remove alpha from [adapter]"). The staged file must keep
    the user's values verbatim (only the dataset path is resolved)."""
    abs_ds = ui_data_tmp / "abs.toml"
    abs_ds.write_text(DATASET_TOML, encoding="utf-8")
    content = (
        MINIMAL_TOML.replace(
            'dataset = "rengu-flow-dataset:my_dataset"',
            f'dataset = "{abs_ds}"',
        )
        + '\n[adapter]\ntype = "lora"\nrank = 8\n'
    )
    out = run_staging.materialize_staging(content, "job-defaults")
    raw = out.read_text(encoding="utf-8")
    cfg = toml.loads(raw)
    assert cfg["model"]["dtype"] == "bfloat16"
    assert "torch." not in raw
    assert "alpha" not in cfg["adapter"]
    assert cfg["adapter"]["rank"] == 8


def test_validate_rejects_bad_toml() -> None:
    assert run_staging.validate_toml_text("not valid {{{")["ok"] is False


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
    out = run_staging.materialize_staging(content, "job-merge")
    cfg = toml.loads(out.read_text(encoding="utf-8"))
    merged_path = Path(cfg["dataset"])
    assert merged_path.is_file()
    merged = toml.loads(merged_path.read_text(encoding="utf-8"))
    assert len(merged["directory"]) == 2


def test_validate_accepts_dataset_list(ui_data_tmp: Path, minimal_config: dict) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML)
    ref = library_db.dataset_library_ref(did)
    minimal_config["dataset"] = [ref, ref]
    r = run_staging.validate_toml_text(toml.dumps(minimal_config))
    assert r["ok"] is True


def test_validate_accepts_minimal(ui_data_tmp: Path, minimal_config: dict) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML)
    minimal_config["dataset"] = library_db.dataset_library_ref(did)
    text = toml.dumps(minimal_config)
    r = run_staging.validate_toml_text(text)
    assert r["ok"] is True
    assert "config" in r
    assert isinstance(r["config"]["model"]["dtype"], str)
