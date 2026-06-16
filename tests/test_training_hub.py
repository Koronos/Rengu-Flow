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


def test_compute_run_progress_uses_manifest_without_parsing_tensorboard(tmp_path, monkeypatch) -> None:
    """With a manifest present, the runs list must never parse TensorBoard event files — that
    per-run EventAccumulator parse is what made listing cost seconds per run."""
    from rengu_track.run import RunManifest, write_manifest

    run_dir = tmp_path / "run_m"
    run_dir.mkdir()
    (run_dir / "train.toml").write_text('max_steps = 100\n[model]\ntype = "sdxl"\n', encoding="utf-8")
    write_manifest(run_dir, RunManifest(run_id="run_m", last_scalars={"train/loss": 0.3}, last_step=42))
    # Event files exist, but a manifest means they must not be read.
    (run_dir / "events.out.tfevents.1").write_bytes(b"x")

    def _boom(*_a, **_k):
        raise AssertionError("parsed TensorBoard despite a manifest being present")

    monkeypatch.setattr(training_hub.metrics_tb, "read_scalars", _boom)
    prog = training_hub.compute_run_progress(run_dir)
    assert prog is not None
    assert prog["step"] == 42
    assert prog["loss"] == 0.3


def test_enrich_backfills_manifest_for_terminal_run(ui_data_tmp: Path) -> None:
    """A terminal run without a manifest gets one written on enrich, so later loads skip the TB
    parse; active runs are left to the trainer's own manifest writer."""
    from rengu_flow_ui import db
    from rengu_track import read_manifest

    log = ui_data_tmp / "logs" / "bf.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("hi\n", encoding="utf-8")

    run_dir = ui_data_tmp / "output" / "run_bf"
    run_dir.mkdir(parents=True)
    job = db.create_job(config_path="x.toml", log_path=str(log), output_dir=str(ui_data_tmp / "output"))
    db.update_job(job.id, state="finished", run_dir=str(run_dir), pid=None)
    assert read_manifest(run_dir) is None

    prog = {"step": 7, "loss": 0.2, "val_loss": None, "val_gap": None, "run_name_label": None}
    training_hub._backfill_manifest_from_progress(db.get_job(job.id), run_dir, prog)
    man = read_manifest(run_dir)
    assert man is not None and man.last_step == 7 and man.last_scalars.get("train/loss") == 0.2

    # An active run is the trainer's to own — no backfill.
    run_active = ui_data_tmp / "output" / "run_active"
    run_active.mkdir(parents=True)
    job2 = db.create_job(config_path="x.toml", log_path=str(log), output_dir=str(ui_data_tmp / "output"))
    db.update_job(job2.id, state="running", run_dir=str(run_active), pid=10_000_000)
    training_hub._backfill_manifest_from_progress(db.get_job(job2.id), run_active, prog)
    assert read_manifest(run_active) is None


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
