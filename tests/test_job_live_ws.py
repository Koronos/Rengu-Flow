"""WebSocket live stream for active training jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from rengu_flow_ui import db, jobs, live_stream


def _running_job(ui_data_tmp: Path, *, run_dir: Path, log_path: Path) -> db.JobRecord:
    job = db.create_job(
        config_path="configs/x.toml",
        log_path=str(log_path),
        output_dir=str(run_dir.parent),
    )
    db.update_job(
        job.id,
        state="running",
        run_dir=str(run_dir),
        pid=os.getpid(),
    )
    return db.get_job(job.id)


from rengu_flow.control.progress_stream import format_progress_marker


def test_snapshot_job_live_progress_and_log(ui_data_tmp: Path) -> None:
    run_dir = ui_data_tmp / "output" / "run_ws"
    run_dir.mkdir(parents=True)
    log_path = ui_data_tmp / "logs" / "live.log"
    # Progress now flows from a throttled @@RFPROG@@ marker in the captured log; the
    # marker line is stripped from the displayed log chunk.
    marker = format_progress_marker({"phase": "training", "step": 3, "loss": 0.25, "epoch": 1})
    log_path.write_text(f"line one\n{marker}\n", encoding="utf-8")
    (run_dir / "train.toml").write_text(
        "max_steps = 10\n[model]\ntype = \"sdxl\"\n",
        encoding="utf-8",
    )
    job = _running_job(ui_data_tmp, run_dir=run_dir, log_path=log_path)

    snap = live_stream.snapshot_job_live(str(job.id))
    assert snap["log_chunk"] == "line one\n"
    assert snap["progress"]["step"] == 3
    assert snap["state"] == "running"


def test_tail_log_strips_progress_markers(ui_data_tmp: Path) -> None:
    run_dir = ui_data_tmp / "output" / "run_strip"
    run_dir.mkdir(parents=True)
    log_path = ui_data_tmp / "logs" / "strip.log"
    marker = format_progress_marker({"phase": "training", "step": 7})
    log_path.write_text(f"keep me\n{marker}\nkeep me too\n", encoding="utf-8")
    job = _running_job(ui_data_tmp, run_dir=run_dir, log_path=log_path)

    text, offset = jobs.tail_log(str(job.id))
    assert "keep me\n" in text
    assert "keep me too\n" in text
    assert "@@RFPROG@@" not in text
    assert offset == log_path.stat().st_size


def test_job_live_ws_delivers_progress(ui_client, ui_data_tmp: Path) -> None:
    run_dir = ui_data_tmp / "output" / "run_ws_api"
    run_dir.mkdir(parents=True)
    log_path = ui_data_tmp / "logs" / "live_api.log"
    marker = format_progress_marker({"phase": "training", "step": 1, "loss": 0.1, "epoch": 0})
    log_path.write_text(f"{marker}\n", encoding="utf-8")
    (run_dir / "train.toml").write_text("max_steps = 5\n[model]\ntype = \"sdxl\"\n", encoding="utf-8")
    job = _running_job(ui_data_tmp, run_dir=run_dir, log_path=log_path)

    with ui_client.websocket_connect(f"/api/v1/jobs/{job.id}/live/ws") as ws:
        seen: set[str] = set()
        for _ in range(8):
            raw = ws.receive_text()
            msg = json.loads(raw)
            seen.add(msg["type"])
            if msg["type"] == "progress":
                assert msg["progress"]["step"] == 1
                break
        assert "progress" in seen


def test_system_stats_ws_pushes_stats(ui_client) -> None:
    """The global host-stats socket pushes a system_stats message (replaces HTTP polling)."""
    with ui_client.websocket_connect("/api/v1/system/stats/ws") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "system_stats"
        assert msg["stats"]["ok"] is True
        assert "summary" in msg["stats"]
        assert "detail" in msg["stats"]
