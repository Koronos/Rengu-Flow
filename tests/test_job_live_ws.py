"""WebSocket live stream for active training jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from rengu_flow_ui import db, live_stream


def _running_job(ui_data_tmp: Path, *, run_dir: Path, log_path: Path) -> db.JobRecord:
    job = db.create_job(
        config_path="configs/x.toml",
        config_id=None,
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


def test_snapshot_job_live_progress_and_log(ui_data_tmp: Path) -> None:
    run_dir = ui_data_tmp / "output" / "run_ws"
    run_dir.mkdir(parents=True)
    log_path = ui_data_tmp / "logs" / "live.log"
    log_path.write_text("line one\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"step": 3, "loss": 0.25, "epoch": 1, "phase": "training"}),
        encoding="utf-8",
    )
    (run_dir / "train.toml").write_text(
        "max_steps = 10\n[model]\ntype = \"sdxl\"\n",
        encoding="utf-8",
    )
    job = _running_job(ui_data_tmp, run_dir=run_dir, log_path=log_path)

    snap = live_stream.snapshot_job_live(str(job.id))
    assert snap["log_chunk"] == "line one\n"
    assert snap["progress"]["step"] == 3
    assert snap["state"] == "running"


def test_job_live_ws_delivers_progress(ui_client, ui_data_tmp: Path) -> None:
    run_dir = ui_data_tmp / "output" / "run_ws_api"
    run_dir.mkdir(parents=True)
    log_path = ui_data_tmp / "logs" / "live_api.log"
    log_path.write_text("", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"step": 1, "loss": 0.1, "epoch": 0, "phase": "training"}),
        encoding="utf-8",
    )
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
