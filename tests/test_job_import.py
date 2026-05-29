"""Import script-mode run folders into the UI job registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rengu_flow_ui import configs_store, datasets_store, db, job_import
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
    assert job.config_id == run.name
    assert configs_store.config_exists(run.name)
    assert datasets_store.dataset_exists(f"{run.name}_dataset")

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
        json={"run_path": str(run), "import_config": True, "import_dataset": True},
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


def test_import_rejects_empty_dir(ui_data_tmp: Path) -> None:
    empty = ui_data_tmp / "empty"
    empty.mkdir()
    with pytest.raises(JobImportError):
        job_import.preview_import(str(empty))
