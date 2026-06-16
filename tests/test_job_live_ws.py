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


def test_tail_log_reads_incrementally_and_resets_on_truncation(ui_data_tmp: Path) -> None:
    run_dir = ui_data_tmp / "output" / "run_incr"
    run_dir.mkdir(parents=True)
    log_path = ui_data_tmp / "logs" / "incr.log"
    log_path.write_text("first\n", encoding="utf-8")
    job = _running_job(ui_data_tmp, run_dir=run_dir, log_path=log_path)

    text, offset = jobs.tail_log(str(job.id))
    assert text == "first\n"
    assert offset == log_path.stat().st_size

    # A tail from the saved offset returns ONLY the bytes appended since (not the whole file).
    with log_path.open("a", encoding="utf-8") as f:
        f.write("second\n")
    text2, offset2 = jobs.tail_log(str(job.id), offset)
    assert text2 == "second\n"
    assert offset2 == log_path.stat().st_size

    # Truncation/rotation: file shrinks below the saved offset -> restart from the top.
    log_path.write_text("fresh\n", encoding="utf-8")
    text3, offset3 = jobs.tail_log(str(job.id), offset2)
    assert text3 == "fresh\n"
    assert offset3 == log_path.stat().st_size


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


def test_iter_log_frames_bounds_frame_bytes() -> None:
    assert jobs.iter_log_frames("") == []
    # 3 MB ASCII split into sub-limit frames; nothing dropped or reordered.
    big = "x" * (3 * 1024 * 1024)
    frames = jobs.iter_log_frames(big)
    assert "".join(frames) == big
    assert all(len(f.encode()) <= jobs.LOG_WS_FRAME_BYTES for f in frames)
    # Worst-case multi-byte UTF-8 still stays under the 1 MB WS frame limit.
    multi = "é" * 500_000
    assert all(len(f.encode()) < 1_048_576 for f in jobs.iter_log_frames(multi))


def test_log_tail_start_offset(ui_data_tmp: Path) -> None:
    log_path = ui_data_tmp / "logs" / "tail.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("X" * (jobs.LOG_WS_TAIL_BYTES + 100_000), encoding="utf-8")
    job = db.create_job(
        config_path="x.toml", log_path=str(log_path), output_dir=str(ui_data_tmp / "output")
    )
    # Seeks to exactly tail_bytes before EOF; the read from there is just the recent tail.
    off = jobs.log_tail_start_offset(job.id)
    assert off == log_path.stat().st_size - jobs.LOG_WS_TAIL_BYTES
    chunk, eof = jobs.tail_log(job.id, off)
    assert len(chunk) <= jobs.LOG_WS_TAIL_BYTES
    assert eof == log_path.stat().st_size

    # A short log is sent in full (offset 0); an unknown job never raises.
    short = ui_data_tmp / "logs" / "short.log"
    short.write_text("hi\n", encoding="utf-8")
    j2 = db.create_job(
        config_path="x.toml", log_path=str(short), output_dir=str(ui_data_tmp / "output")
    )
    assert jobs.log_tail_start_offset(j2.id) == 0
    assert jobs.log_tail_start_offset("does-not-exist") == 0


def test_logs_ws_bounds_frame_size_for_large_log(ui_client, ui_data_tmp: Path) -> None:
    """A multi-MB log must stream in sub-limit frames instead of one oversized frame.

    A single multi-MB frame previously tripped the 1 MB WS limit (1009 close), silently dropping
    the client to HTTP polling. We assert the FIRST delivered frame is bounded; reading just one
    frame off a running job avoids waiting on the server-side close (which hangs the TestClient).
    """
    run_dir = ui_data_tmp / "output" / "run_big_ws"
    run_dir.mkdir(parents=True)
    log_path = ui_data_tmp / "logs" / "big_ws.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("training output line\n" * 200_000, encoding="utf-8")  # ~4 MB
    job = _running_job(ui_data_tmp, run_dir=run_dir, log_path=log_path)

    with ui_client.websocket_connect(f"/api/v1/jobs/{job.id}/logs/ws") as ws:
        first = ws.receive_text()

    # The first frame is a bounded slice, well under the 1 MB WS frame limit.
    assert 0 < len(first.encode()) <= jobs.LOG_WS_FRAME_BYTES
    assert len(first.encode()) < 1_048_576


def test_system_stats_ws_pushes_stats(ui_client) -> None:
    """The global host-stats socket pushes a system_stats message (replaces HTTP polling)."""
    with ui_client.websocket_connect("/api/v1/system/stats/ws") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "system_stats"
        assert msg["stats"]["ok"] is True
        assert "summary" in msg["stats"]
        assert "detail" in msg["stats"]
