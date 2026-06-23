"""Import script-mode run folders into the UI job registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rengu_flow_ui import datasets_store, job_import, library_db
from rengu_flow_ui.job_import import JobImportError


def _make_run_dir(parent: Path, name: str = "20250101_12-00-00") -> Path:
    run = parent / name
    run.mkdir(parents=True)
    (run / "train.toml").write_text(
        'dataset = "dataset.toml"\noutput_dir = "output"\n\n'
        '[model]\ntype = "sdxl"\ndtype = "bfloat16"\n'
        'checkpoint_path = "/tmp/x.safetensors"\n\n'
        '[optimizer]\ntype = "adamw"\nlr = 1.0e-4\n',
        encoding="utf-8",
    )
    (run / "dataset.toml").write_text(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/tmp/images'\nnum_repeats = 1\n",
        encoding="utf-8",
    )
    (run / "status.json").write_text(
        json.dumps(
            {
                "step": 100,
                "loss": 0.1,
                "epoch": 1,
                "examples": 100,
                "updated_at": "2025-01-01T12:30:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run / "global_step100").mkdir()
    return run


def test_preview_and_import_run(ui_data_tmp: Path) -> None:
    run = _make_run_dir(ui_data_tmp / "output")
    preview = job_import.preview_import(str(run))
    assert preview["ok"] is True
    assert preview["run"]["name"] == run.name
    assert preview["already_imported"] is False

    job = job_import.import_run(str(run))
    assert job.state == "finished"
    assert job.run_dir == str(run.resolve())
    # The run carries its config inline as a snapshot.
    assert 'type = "sdxl"' in job.config_content
    # The imported dataset is stored and named after the run.
    ds_names = [d["name"] for d in library_db.list_datasets_summary()]
    assert f"{run.name} dataset" in ds_names

    with pytest.raises(JobImportError, match="already"):
        job_import.import_run(str(run))

    job2 = job_import.import_run(str(run), allow_duplicate=True)
    assert job2.id != job.id


def test_import_via_api(ui_client, ui_data_tmp: Path) -> None:
    run = _make_run_dir(ui_data_tmp / "runs_out", "script_run")
    r = ui_client.post("/api/v1/jobs/import/preview", json={"run_path": str(run)})
    assert r.status_code == 200
    assert r.json()["run"]["name"] == "script_run"

    r2 = ui_client.post(
        "/api/v1/jobs/import",
        json={"run_path": str(run), "import_dataset": True},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["state"] == "finished"
    assert body["run_dir"] == str(run.resolve())

    r3 = ui_client.get("/api/v1/jobs/import/candidates", params={"output_dir": str(run.parent)})
    assert r3.status_code == 200
    names = [x["name"] for x in r3.json()["runs"]]
    assert "script_run" in names
    imported = next(x for x in r3.json()["runs"] if x["name"] == "script_run")
    assert imported["already_imported"] is True


def test_import_resolves_relative_paths_to_absolute(ui_data_tmp: Path) -> None:
    """A run with a relative dataset path / relative directory path becomes absolute."""
    run = ui_data_tmp / "output" / "rel_run"
    run.mkdir(parents=True)
    # Config references the dataset via a *relative* path.
    (run / "train.toml").write_text(
        'dataset = "rel_dataset.toml"\noutput_dir = "output"\n\n'
        '[model]\ntype = "sdxl"\ndtype = "bfloat16"\n'
        'checkpoint_path = "/tmp/x.safetensors"\n\n'
        '[optimizer]\ntype = "adamw"\nlr = 1.0e-4\n',
        encoding="utf-8",
    )
    # Dataset TOML uses a *relative* [[directory]].path.
    (run / "rel_dataset.toml").write_text(
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = 'tests/fixtures/smoke_cc0/images'\nnum_repeats = 1\n",
        encoding="utf-8",
    )
    (run / "status.json").write_text(
        json.dumps({"step": 1, "updated_at": "2025-01-01T12:30:00+00:00"}),
        encoding="utf-8",
    )
    (run / "global_step1").mkdir()

    import toml as _toml

    from rengu_flow_ui.paths import resolve_repo_path

    job = job_import.import_run(str(run))

    # The imported config_content carries an ABSOLUTE dataset path.
    cfg = _toml.loads(job.config_content)
    ds_val = cfg["dataset"]
    assert Path(ds_val).is_absolute(), ds_val
    # Forward-slash (PLATFORM.config_path) so the config is valid TOML on Windows; == str() on POSIX.
    assert ds_val == resolve_repo_path("rel_dataset.toml").as_posix()
    # output_dir is left untouched.
    assert cfg["output_dir"] == "output"

    # The inserted library dataset has an ABSOLUTE [[directory]].path.
    summaries = library_db.list_datasets_summary()
    ds = next(d for d in summaries if d["name"] == f"{run.name} dataset")
    text = datasets_store.read_dataset_text(ds["id"])
    dcfg = _toml.loads(text)
    dir_path = dcfg["directory"][0]["path"]
    assert Path(dir_path).is_absolute(), dir_path
    assert dir_path == resolve_repo_path("tests/fixtures/smoke_cc0/images").as_posix()


def test_resolve_config_dataset_paths_idempotent_and_refs() -> None:
    """List shape preserved, library refs and absolute paths untouched."""
    abs_ds = job_import.resolve_repo_path("a.toml").as_posix()
    text = (
        f'dataset = ["rel.toml", "{abs_ds}", "rengu-flow-dataset:3"]\n'
        'output_dir = "output"\n'
    )
    out = job_import.resolve_config_dataset_paths(text)
    import toml as _toml

    cfg = _toml.loads(out)
    vals = cfg["dataset"]
    assert isinstance(vals, list)
    # Resolved paths come back forward-slash (PLATFORM.config_path); as_posix() == str() on POSIX.
    assert vals[0] == job_import.resolve_repo_path("rel.toml").as_posix()
    assert vals[1] == abs_ds  # already absolute, unchanged
    assert vals[2] == "rengu-flow-dataset:3"  # library ref untouched
    # Idempotent.
    assert job_import.resolve_config_dataset_paths(out) == out


def test_import_rejects_empty_dir(ui_data_tmp: Path) -> None:
    empty = ui_data_tmp / "empty"
    empty.mkdir()
    with pytest.raises(JobImportError):
        job_import.preview_import(str(empty))
