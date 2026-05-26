"""Read TensorBoard scalar events from a run directory (passive, no trainer load)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_scalars(run_dir: str | Path, tag_prefix: str = "train/") -> dict[str, list[dict[str, Any]]]:
    """Return {tag: [{step, value, wall_time}, ...]} from event files."""
    root = Path(run_dir)
    if not root.is_dir():
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return _read_scalars_fallback(root, tag_prefix)

    acc = EventAccumulator(str(root), size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for tag in acc.Tags().get("scalars", []):
        if tag_prefix and not tag.startswith(tag_prefix):
            continue
        events = acc.Scalars(tag)
        out[tag] = [
            {"step": e.step, "value": float(e.value), "wall_time": e.wall_time}
            for e in events
        ]
    return out


def _read_scalars_fallback(run_dir: Path, tag_prefix: str) -> dict[str, list[dict[str, Any]]]:
    """Minimal parser when tensorboard is not installed."""
    del run_dir, tag_prefix
    return {}
