"""Run-folder TOML helpers for resume / continue."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from renga_flow_ui.run_config import describe_run_config, read_run_config_text, resume_checkpoint_arg


def _make_run(parent: Path, name: str = "20250101_12-00-00") -> Path:
    run = parent / "output" / name
    run.mkdir(parents=True)
    (run / "train.toml").write_text(
        'dataset = "dataset.toml"\noutput_dir = "output"\nepochs = 2\n\n'
        '[model]\ntype = "sdxl"\ndtype = "bfloat16"\n'
        'checkpoint_path = "/tmp/x.safetensors"\n\n'
        '[optimizer]\ntype = "adamw"\nlr = 1.0e-4\n',
        encoding="utf-8",
    )
    (run / "dataset.toml").write_text(
        "resolutions = [1024]\n\n[[directory]]\npath = '/tmp/x'\nnum_repeats = 1\n",
        encoding="utf-8",
    )
    (run / "global_step10").mkdir()
    return run


def test_read_run_config_and_resume_arg(ui_data_tmp: Path) -> None:
    run = _make_run(ui_data_tmp)
    text = read_run_config_text(run)
    assert "epochs = 2" in text
    arg = resume_checkpoint_arg(run)
    assert arg == run.name or arg.endswith(run.name)

    desc = describe_run_config(run)
    assert desc["resume_from"] == arg
    assert desc["epochs"] == 2


def test_continue_run_api(ui_client, ui_data_tmp: Path) -> None:
    run = _make_run(ui_data_tmp)
    r = ui_client.get("/api/v1/runs/config", params={"run_path": str(run)})
    assert r.status_code == 200
    base = r.json()["content"]

    updated = base.replace("epochs = 2", "epochs = 10")
    r2 = ui_client.post(
        "/api/v1/jobs/continue-run",
        json={
            "run_path": str(run),
            "content": updated,
            "enqueue": True,
            "start_immediately": False,
        },
    )
    assert r2.status_code == 200
    job = r2.json()
    assert str(job["resume_from"]).endswith(run.name)
    assert job["source_run_dir"] == str(run.resolve())
