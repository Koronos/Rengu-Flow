"""WebSocket live stream for running jobs: progress + log tail (+ periodic metrics).

Live progress is derived from throttled ``@@RFPROG@@`` markers the trainer prints to
stdout (parsed from the captured log), not from status.json (which is no longer
written). Marker lines are stripped from the log text shown to the client.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rengu_flow.control.progress_stream import parse_last_progress_marker
from rengu_flow_ui import jobs, metrics_tb, training_hub

ACTIVE_STATES = training_hub.ACTIVE_STATES

# Seconds between loop iterations (status/log tail).
_TICK_SEC = 0.5
# Send TB scalars / preview images every N ticks when still active.
_METRICS_EVERY_TICKS = 10


def _latest_marker(job_id: str) -> dict[str, Any] | None:
    """Parse the last complete @@RFPROG@@ marker from the job's raw log tail.

    Only reads the last 64 KB of the log file instead of the entire thing — the marker
    is emitted at most ~1/s and is <200 bytes, so this always captures the latest one.
    """
    try:
        raw = jobs.read_raw_log_tail(job_id)
    except KeyError:
        return None
    if not raw:
        return None
    return parse_last_progress_marker(raw)


def snapshot_job_live(job_id: str, *, log_offset: int = 0) -> dict[str, Any]:
    """One poll cycle: progress, optional log chunk, job state (sync, for tests)."""
    job = jobs.poll_job(job_id)
    marker = _latest_marker(job_id)
    if job.kind == "prep":
        # Prep jobs have no run folder / TB metrics: the marker payload IS the progress.
        run_dir = Path(job.run_dir) if job.run_dir else None
        progress = marker
    else:
        run_dir = training_hub.resolve_job_run_dir(job)
        progress = training_hub.compute_run_progress(run_dir, marker=marker)
    chunk, new_offset = jobs.tail_log(job_id, log_offset)
    out: dict[str, Any] = {
        "job_id": job_id,
        "state": job.state,
        "run_dir": str(run_dir) if run_dir else job.run_dir,
        "kind": job.kind,
        "progress": progress,
        "log_offset": new_offset,
    }
    if chunk:
        out["log_chunk"] = chunk
    return out


def _metrics_payload(run_dir: Path) -> dict[str, Any]:
    return {
        "scalars": metrics_tb.read_scalars(run_dir),
        "preview_images": training_hub.list_run_preview_images(run_dir),
    }


async def run_job_live_ws(send_json, job_id: str) -> None:
    """Drive a job live WebSocket until disconnect or run finishes.

    ``send_json`` is an async callable accepting a dict (serialized to JSON text).
    Message shapes:
      - ``{"type": "progress", "state", "run_dir", "progress"}``
      - ``{"type": "log_line", "chunk": "<utf-8 text>"}``
      - ``{"type": "metrics", "scalars", "preview_images"}``
      - ``{"type": "run_finished", "state"}``
      - ``{"type": "error", "message"}``
    """
    log_offset = 0
    last_progress_json: str | None = None
    tick = 0

    while True:
        try:
            snap = await asyncio.to_thread(snapshot_job_live, job_id, log_offset=log_offset)
        except KeyError:
            await send_json({"type": "error", "message": "job not found"})
            return

        log_offset = int(snap["log_offset"])
        state = snap["state"]
        run_dir_str = snap.get("run_dir")
        run_dir = Path(run_dir_str).resolve() if run_dir_str else None

        chunk = snap.get("log_chunk")
        if chunk:
            await send_json({"type": "log_line", "chunk": chunk})

        # Progress is driven by the marker stream; emit whenever it changes. The last
        # parsed marker persists across ticks (parse_last_progress_marker re-reads the
        # full log), so the bar holds its last-known value between marker emits.
        progress = snap.get("progress")
        progress_json = json.dumps(progress, sort_keys=True, default=str) if progress else None
        if progress_json != last_progress_json:
            last_progress_json = progress_json
            await send_json(
                {
                    "type": "progress",
                    "state": state,
                    "run_dir": run_dir_str,
                    "progress": progress,
                }
            )

        tick += 1
        if (
            tick % _METRICS_EVERY_TICKS == 0
            and snap.get("kind") != "prep"
            and run_dir is not None
            and run_dir.is_dir()
        ):
            metrics = await asyncio.to_thread(_metrics_payload, run_dir)
            await send_json({"type": "metrics", **metrics})

        if state not in ACTIVE_STATES:
            await send_json({"type": "run_finished", "state": state})
            return

        await asyncio.sleep(_TICK_SEC)


# Seconds between host-stats pushes on the global system-stats socket.
_SYSTEM_STATS_SEC = 2.0


async def run_system_stats_ws(send_json, *, interval_sec: float = _SYSTEM_STATS_SEC) -> None:
    """Push host CPU/RAM/GPU stats over a global WebSocket, replacing per-client HTTP polling.

    Stats are app-global (not per-job), so this rides a dedicated socket rather than the per-job
    live stream. ``collect_system_stats`` blocks (psutil CPU sampling + an ``nvidia-smi`` subprocess),
    so it runs in a thread to keep the event loop responsive. The loop ends when the client
    disconnects (``send_json`` raises ``WebSocketDisconnect``, handled by the endpoint).
    """
    from rengu_flow_ui.system_stats import collect_system_stats

    while True:
        stats = await asyncio.to_thread(collect_system_stats)
        await send_json({"type": "system_stats", "stats": stats})
        await asyncio.sleep(interval_sec)
