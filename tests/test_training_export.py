"""Training export ZIP bundles for CLI use."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import toml

from rengu_flow.config.dataset_library_ref import dataset_library_ref
from rengu_flow_ui import datasets_store
from rengu_flow_ui.training_export import (
    absolutize_dataset_config,
    build_training_export_zip,
    resolve_media_path,
)

DATASET_TOML = """
resolutions = [512]
frame_buckets = [1]

[[directory]]
path = "examples/minimal_images"
num_repeats = 1
""".strip()


def test_resolve_media_path_relative_to_repo(examples_dir: Path) -> None:
    rel = "examples/minimal_dataset.toml"
    got = resolve_media_path(rel, dataset_toml_path=None)
    assert got == str((examples_dir.parent / rel).resolve()) or Path(got).is_absolute()


def test_export_dataset_toml_absolutizes_directory_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img = tmp_path / "imgs"
    img.mkdir()
    content = f"""
resolutions = [512]
frame_buckets = [1]

[[directory]]
path = "{img.name}"
num_repeats = 1
""".strip()
    cfg = absolutize_dataset_config(
        toml.loads(content),
        dataset_toml_path=tmp_path / "ds.toml",
    )
    assert Path(cfg["directory"][0]["path"]).is_absolute()
    assert Path(cfg["directory"][0]["path"]).resolve() == img.resolve()


def test_build_zip_resolves_library_ref(ui_data_tmp: Path) -> None:
    did = datasets_store.insert_dataset(DATASET_TOML, name="Export me")
    ref = dataset_library_ref(did, "Export me")
    train = f"""
dataset = "{ref}"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "path/to/model.safetensors"

[optimizer]
type = "adamw"
lr = 1.0e-4
""".strip()
    zip_bytes, filename = build_training_export_zip(train, bundle_stem="my_run")
    assert filename == "my_run.zip"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "my_run.toml" in names
        assert f"datasets/dataset_{did}.toml" in names
        main = toml.loads(zf.read("my_run.toml").decode())
        assert main["dataset"] == f"datasets/dataset_{did}.toml"
        ds = toml.loads(zf.read(f"datasets/dataset_{did}.toml").decode())
        assert "name" not in ds
        assert Path(ds["directory"][0]["path"]).is_absolute()


def test_build_zip_with_file_path_dataset(examples_dir: Path) -> None:
    ds_path = examples_dir / "minimal_dataset.toml"
    train = f'dataset = "{ds_path.as_posix()}"\n[model]\ntype = "sdxl"\ndtype = "bfloat16"\ncheckpoint_path = "x"\n[optimizer]\ntype = "adamw"\nlr = 1e-4\n'
    zip_bytes, _ = build_training_export_zip(train, bundle_stem="file_ref")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        main = toml.loads(zf.read("file_ref.toml").decode())
        assert main["dataset"].startswith("datasets/")
        assert main["dataset"].endswith(".toml")


def test_export_config_api_returns_zip(ui_client) -> None:
    content = (
        'dataset = "examples/minimal_dataset.toml"\n[model]\ntype = "sdxl"\ndtype = "bfloat16"\ncheckpoint_path = "x"\n[optimizer]\ntype = "adamw"\nlr = 1e-4\n'
    )
    r = ui_client.post(
        "/api/v1/configs/export-bundle",
        json={"content": content, "name": "my_run"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers.get("content-disposition", "")
