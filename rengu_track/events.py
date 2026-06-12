"""Append-only per-run event timeline (``run_events.jsonl``).

A shared primitive written by BOTH clients of the tracking core: the trainer (lifecycle +
live ``[preview]`` reload) and the UI (config edits, continue/resume) — UI mutations happen
while no trainer process is running, so the UI records them itself. One newline-delimited JSON
record per event; appends use ``O_APPEND`` so concurrent writers never interleave a line
(records are far below ``PIPE_BUF``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rengu_track.run import flatten_dict, now_iso

EVENTS_NAME = "run_events.jsonl"

# Event types (the run's lifecycle + every config mutation).
EVENT_RUN_STARTED = "run_started"
EVENT_RESUMED = "resumed"
EVENT_RESTARTED_FROM_SCRATCH = "restarted_from_scratch"
EVENT_CONFIG_RELOADED = "config_reloaded"
EVENT_CONFIG_EDITED = "config_edited"
EVENT_STOP_REQUESTED = "stop_requested"
EVENT_FINISHED = "finished"
EVENT_FAILED = "failed"


def append_event(
    run_dir: str | Path,
    event_type: str,
    *,
    step: int | None = None,
    payload: dict[str, Any] | None = None,
    source: str = "trainer",
) -> dict[str, Any]:
    """Append one event record to ``run_dir/run_events.jsonl`` and return it.

    ``source`` distinguishes trainer- vs UI-originated events. The append is atomic per line.
    """
    record: dict[str, Any] = {
        "ts": now_iso(),
        "type": event_type,
        "step": step,
        "source": source,
        "payload": payload or {},
    }
    line = json.dumps(record, separators=(",", ":")) + "\n"
    path = Path(run_dir) / EVENTS_NAME
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    return record


def read_events(run_dir: str | Path) -> list[dict[str, Any]]:
    """Return all event records in order; skips malformed/partial lines."""
    path = Path(run_dir) / EVENTS_NAME
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def config_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Deep diff of two config dicts as dotted keys → {added, removed, changed:{k:[old,new]}}."""
    old_flat = flatten_dict(old or {})
    new_flat = flatten_dict(new or {})
    old_keys, new_keys = set(old_flat), set(new_flat)
    return {
        "added": {k: new_flat[k] for k in sorted(new_keys - old_keys)},
        "removed": {k: old_flat[k] for k in sorted(old_keys - new_keys)},
        "changed": {
            k: [old_flat[k], new_flat[k]]
            for k in sorted(old_keys & new_keys)
            if old_flat[k] != new_flat[k]
        },
    }
