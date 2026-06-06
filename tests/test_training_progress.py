"""Unit tests for training progress / ETA helpers."""

from __future__ import annotations

from pathlib import Path

from rengu_flow.control.status_file import read_status_file, write_status_file
from rengu_flow.training_progress import (
    TrainingProgressTracker,
    budget_display_epoch,
    format_eta,
    format_training_log_line,
    resolve_target_steps,
)
from rengu_flow_ui import training_hub


def test_budget_display_epoch_caps_at_configured_epochs():
    # 15-epoch budget, 100 steps/epoch (full multi-res epoch) -> total 1500 steps.
    spe, epochs = 100, 15
    assert budget_display_epoch(1, spe, epochs) == 1
    assert budget_display_epoch(100, spe, epochs) == 1
    assert budget_display_epoch(101, spe, epochs) == 2
    assert budget_display_epoch(1500, spe, epochs) == 15
    # Past the budget (short staged epochs would overshoot) stays capped at 15.
    assert budget_display_epoch(4500, spe, epochs) == 15


def test_budget_display_epoch_handles_zero_steps_per_epoch():
    assert budget_display_epoch(42, 0, 15) == 42


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
    # Without an EMA step time we fall back to the instant step/s rate.
    assert "speed=0.85 step/s" in line
    assert "remaining=90" in line
    assert "eta=1m 46s" in line
    assert "epoch=1" in line


def test_format_training_log_line_prefers_smoothed() -> None:
    line = format_training_log_line(
        step=10,
        loss=0.42,
        epoch=1,
        metrics={
            "max_steps": 100,
            "loss_avg": 0.40,
            "step_time_sec_ema": 2.5,
            "steps_per_second": 0.85,
        },
    )
    # Kohya-style display: smoothed avr_loss and EMA s/it, not the jumpy instant values.
    assert "avr_loss=0.400000" in line
    assert "speed=2.50 s/it" in line
    assert "loss=0.420000" not in line


def test_tracker_loss_moving_average_windowed() -> None:
    tracker = TrainingProgressTracker(max_steps=10, loss_window=2)
    tracker.record_loss(1.0)
    assert tracker.loss_avg == 1.0
    tracker.record_loss(3.0)
    assert tracker.loss_avg == 2.0  # mean(1, 3)
    tracker.record_loss(5.0)
    assert tracker.loss_avg == 4.0  # window slid to mean(3, 5)
    m = tracker.metrics(step=3)
    assert m["loss_avg"] == 4.0
    assert "step_time_sec_ema" not in m  # no durations recorded yet


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
