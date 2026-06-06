"""Train hub unified list and progress."""

from __future__ import annotations

from pathlib import Path

from rengu_flow_ui import training_hub


def test_sort_runs_orders_states() -> None:
    items = [
        {"state": "finished", "finished_at": "2020-01-02", "started_at": ""},
        {"state": "pending", "queue_position": 1, "started_at": ""},
        {"state": "running", "started_at": ""},
    ]
    ordered = training_hub._sort_runs(items)
    assert [r["state"] for r in ordered] == ["running", "pending", "finished"]


def test_compute_run_progress_from_marker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "train.toml").write_text(
        "max_steps = 100\n[model]\ntype = \"sdxl\"\n[optimizer]\ntype = \"adamw\"\n",
        encoding="utf-8",
    )
    # Live progress is supplied by the latest parsed @@RFPROG@@ marker payload.
    marker = {"phase": "training", "step": 25, "loss": 0.5, "epoch": 1}
    prog = training_hub.compute_run_progress(run_dir, marker=marker)
    assert prog is not None
    assert prog["step"] == 25
    assert prog["max_steps"] == 100
    assert prog["percent"] == 25.0
    assert prog["phase"] == "training"


def test_compute_run_progress_caching_marker_without_run_dir() -> None:
    # During caching the run folder does not exist yet; the caching marker (from the log)
    # must still surface so the progress bar shows caching progress.
    marker = {"phase": "caching", "current": 30, "total": 120}
    prog = training_hub.compute_run_progress(None, marker=marker)
    assert prog is not None
    assert prog["phase"] == "caching"
    assert prog["current"] == 30
    assert prog["total"] == 120


def test_compute_run_progress_none_without_dir_or_marker() -> None:
    assert training_hub.compute_run_progress(None, marker=None) is None


def test_train_runs_api(ui_client) -> None:
    r = ui_client.get("/api/v1/train/runs?page=1&page_size=10")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "stats" in data


def test_train_active_api(ui_client) -> None:
    r = ui_client.get("/api/v1/train/active")
    assert r.status_code == 200
    assert "active" in r.json()
