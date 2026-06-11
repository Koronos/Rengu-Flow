"""Unit tests for training progress / ETA helpers."""

from __future__ import annotations

from pathlib import Path

from rengu_flow.control.status_file import read_status_file, write_status_file
from rengu_flow.training_progress import (
    EpochSchedule,
    TrainingProgressTracker,
    budget_display_epoch,
    budget_reached_target,
    format_eta,
    format_training_log_line,
    plan_final_saves,
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


def test_epoch_schedule_current_matches_budget_display():
    sched = EpochSchedule(100, 15)
    for step in (1, 100, 101, 1500, 4500):
        assert sched.current(step) == budget_display_epoch(step, 100, 15)
    assert sched.total_steps == 1500


def test_epoch_schedule_completed_at_boundaries():
    sched = EpochSchedule(100, 15)
    # Completes only at exact multiples of steps_per_epoch, named by the COMPLETED epoch
    # (so the first one is epoch 1, not epoch 2 — the off-by-one that mislabelled saves).
    assert sched.completed_at(1) is None
    assert sched.completed_at(99) is None
    assert sched.completed_at(100) == 1
    assert sched.completed_at(101) is None
    assert sched.completed_at(200) == 2
    assert sched.completed_at(1500) == 15
    # Past the budget (e.g. max_steps lands beyond epochs*spe) does not invent epoch 16.
    assert sched.completed_at(1600) is None


def test_epoch_schedule_first_completed_epoch_is_one():
    # Regression for "first saved epoch is named 2": the sequence of completed epochs over a
    # full run must start at 1 and have no gaps.
    sched = EpochSchedule(10, 5)
    completed = [sched.completed_at(s) for s in range(1, sched.total_steps + 1)]
    assert [c for c in completed if c is not None] == [1, 2, 3, 4, 5]


def test_epoch_schedule_zero_steps_per_epoch_is_safe():
    sched = EpochSchedule(0, 15)
    assert sched.current(42) == 42
    assert sched.completed_at(42) is None
    assert sched.total_steps == 0


def test_budget_reached_target_max_steps_vs_epochs():
    assert budget_reached_target(8820, 15, 8820) == ("step8820", "Reached max_steps=8820")
    assert budget_reached_target(None, 15, 1500) == ("epoch15", "Reached epochs=15")


def test_plan_final_saves_writes_final_checkpoint_after_an_earlier_one():
    # Regression for the reported bug: a resume checkpoint at step 8197 must NOT suppress the
    # final checkpoint at step 8820 (the old sticky "checkpointed" flag did exactly that).
    write_ckpt, export = plan_final_saves(
        step=8820, last_checkpoint_step=8197, last_save_step=8197, final_model_name="step8820"
    )
    assert write_ckpt is True
    assert export == "step8820"


def test_plan_final_saves_skips_when_already_saved_at_this_step():
    # A periodic export + checkpoint landed exactly on the final step -> no redundant final saves.
    write_ckpt, export = plan_final_saves(
        step=8820, last_checkpoint_step=8820, last_save_step=8820, final_model_name="epoch15"
    )
    assert write_ckpt is False
    assert export is None


def test_plan_final_saves_with_no_final_model_name():
    write_ckpt, export = plan_final_saves(
        step=100, last_checkpoint_step=-1, last_save_step=-1, final_model_name=None
    )
    assert write_ckpt is True  # nothing checkpointed yet -> still write the final checkpoint
    assert export is None  # nothing to export


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


def test_loader_announces_new_latent_shapes_once(capsys):
    """With announce_new_shapes on, each distinct latent shape prints one heads-up."""
    import torch

    from rengu_flow.data.loader import PipelineDataLoader

    loader = object.__new__(PipelineDataLoader)
    loader.announce_new_shapes = True
    loader._seen_latent_shapes = set()
    loader.auto_budget_base = None
    loader.auto_budget_max_latent_tokens = None

    def batch(h, w, b=1):
        return ((torch.zeros(b, 16, 1, h, w), torch.zeros(b)), (torch.zeros(b), None))

    loader._maybe_announce_shape(batch(64, 64))
    loader._maybe_announce_shape(batch(64, 64))      # repeat -> silent
    loader._maybe_announce_shape(batch(64, 64, b=2))  # batch dim ignored -> silent
    loader._maybe_announce_shape(batch(128, 96))      # new shape -> printed
    out = capsys.readouterr().out
    assert out.count("[compile] new latent shape") == 2
    assert "1x16x1x64x64" in out and "(2 seen)" in out

    loader.announce_new_shapes = False
    loader._maybe_announce_shape(batch(32, 32))
    assert "[compile]" not in capsys.readouterr().out
