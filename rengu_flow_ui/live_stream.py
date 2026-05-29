"""WebSocket live stream for running jobs: progress + log tail (+ periodic metrics)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rengu_flow_ui import jobs, metrics_tb, training_hub

ACTIVE_STATES = training_hub.ACTIVE_STATES

# Seconds between loop iterations (status/log tail).
_TICK_SEC = 0.5
# Send TB scalars / preview images every N ticks when still active.
_METRICS_EVERY_TICKS = 10


def _status_mtime(run_dir: Path | None) -> float:
    if run_dir is None:
        return 0.0
    path = run_dir / "status.json"
    try:
        return path.stat().st_mtime if path.is_file() else 0.0
    except OSError:
        return 0.0


def snapshot_job_live(job_id: str, *, log_offset: int = 0) -> dict[str, Any]:
    """One poll cycle: progress, optional log chunk, job state (sync, for tests)."""
    job = jobs.poll_job(job_id)
    run_dir = training_hub.resolve_job_run_dir(job)
    progress = training_hub.compute_run_progress(run_dir)
    chunk, new_offset = jobs.tail_log(job_id, log_offset)
    out: dict[str, Any] = {
        "job_id": job_id,
        "state": job.state,
        "run_dir": str(run_dir) if run_dir else job.run_dir,
        "progress": progress,
        "log_offset": new_offset,
        "status_mtime": _status_mtime(run_dir),
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
    last_status_mtime = -1.0
    last_progress_json: str | None = None
    tick = 0
    no_status_ticks = 0

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

        status_mtime = float(snap.get("status_mtime") or 0.0)
        progress = snap.get("progress")
        progress_json = json.dumps(progress, sort_keys=True, default=str) if progress else None

        status_changed = status_mtime != last_status_mtime
        progress_changed = progress_json != last_progress_json
        if status_changed:
            last_status_mtime = status_mtime
        if status_mtime <= 0:
            no_status_ticks += 1
        else:
            no_status_ticks = 0

        # Without status.json, refresh progress periodically (TB fallback is slower).
        periodic_progress = status_mtime <= 0 and no_status_ticks % 4 == 0

        if progress_changed or status_changed or periodic_progress:
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
        if tick % _METRICS_EVERY_TICKS == 0 and run_dir is not None and run_dir.is_dir():
            metrics = await asyncio.to_thread(_metrics_payload, run_dir)
            await send_json({"type": "metrics", **metrics})

        if state not in ACTIVE_STATES:
            await send_json({"type": "run_finished", "state": state})
            return

        await asyncio.sleep(_TICK_SEC)
