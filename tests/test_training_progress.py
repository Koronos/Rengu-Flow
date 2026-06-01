"""Unit tests for training progress / ETA helpers."""

from __future__ import annotations

import json
from pathlib import Path

from rengu_flow.control.status_file import read_status_file, write_status_file
from rengu_flow.training_progress import (
    TrainingProgressTracker,
    format_eta,
    format_training_log_line,
    resolve_target_steps,
)
from rengu_flow_ui import training_hub


def test_resolve_target_steps_prefers_max_steps() -> None:
    assert resolve_target_steps(500, 1000) == 500
    assert resolve_target_steps(None, 1000) == 1000
    assert resolve_target_steps(None, None) is None


def test_format_eta() -> None:
    assert format_eta(45) == "45s"
    assert format_eta(90) == "1m 30s"
    assert format_eta(3661) == "1h 1m"
    assert format_eta(0) == "<1s"
    assert format_eta(None) is None


def test_tracker_ema_and_eta_fixed_durations() -> None:
    tracker = TrainingProgressTracker(max_steps=10, ema_alpha=0.5)
    for dt in (2.0, 2.0, 2.0):
        tracker.record_step_duration(dt)
    m = tracker.metrics(step=3)
    assert m["max_steps"] == 10
    assert m["steps_remaining"] == 7
    assert m["percent"] == 30.0
    assert m["steps_per_second"] == 0.5
    assert m["steps_per_second_ema"] == 0.5
    assert m["eta_sec"] == 14
    assert m["eta"] == "14s"


def test_format_training_log_line() -> None:
    line = format_training_log_line(
        step=10,
        loss=0.42,
        epoch=1,
        metrics={
            "max_steps": 100,
            "percent": 10.0,
            "steps_remaining": 90,
            "steps_per_second": 0.85,
            "steps_per_second_ema": 0.82,
            "eta": "1m 46s",
        },
    )
    assert "step=10/100 (10.0%)" in line
    assert "loss=0.420000" in line
    assert "speed=0.85 step/s (ema 0.82)" in line
    assert "remaining=90" in line
    assert "eta=1m 46s" in line
    assert "epoch=1" in line


def test_write_status_file_includes_progress_fields(tmp_path: Path) -> None:
    write_status_file(
        tmp_path,
        step=5,
        examples=50,
        epoch=1,
        loss=0.1,
        progress={"percent": 50.0, "eta": "10s", "steps_per_second": 1.0},
    )
    data = read_status_file(tmp_path)
    assert data is not None
    assert data["percent"] == 50.0
    assert data["eta"] == "10s"
    assert data["steps_per_second"] == 1.0


def test_compute_run_progress_merges_marker_speed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train.toml").write_text(
        "max_steps = 100\n[model]\ntype = \"sdxl\"\n",
        encoding="utf-8",
    )
    # Speed/ETA fields now ride on the latest @@RFPROG@@ marker payload.
    marker = {
        "step": 40,
        "loss": 0.2,
        "epoch": 1,
        "phase": "training",
        "max_steps": 100,
        "percent": 40.0,
        "steps_remaining": 60,
        "step_time_sec": 2.0,
        "steps_per_second_ema": 0.5,
        "eta": "2m",
    }
    prog = training_hub.compute_run_progress(run_dir, marker=marker)
    assert prog is not None
    assert prog["step_time_sec"] == 2.0
    assert prog["steps_per_second_ema"] == 0.5
    assert prog["eta"] == "2m"
    assert prog["steps_remaining"] == 60
