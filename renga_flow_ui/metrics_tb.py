"""Read TensorBoard scalar events from a run directory (passive, no trainer load)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# run_dir key -> (latest event mtime, parsed scalars)
_scalar_cache: dict[str, tuple[float, dict[str, list[dict[str, Any]]]]] = {}


def invalidate_scalars_cache(run_dir: str | Path | None = None) -> None:
    """Drop cached scalars for one run or all runs."""
    if run_dir is None:
        _scalar_cache.clear()
        return
    _scalar_cache.pop(str(Path(run_dir).resolve()), None)


def _latest_event_mtime(run_dir: Path) -> float:
    latest = 0.0
    for pattern in ("events.out.tfevents.*", "events.out.tfevents.*.*"):
        for event_file in run_dir.glob(pattern):
            if event_file.is_file():
                try:
                    latest = max(latest, event_file.stat().st_mtime)
                except OSError:
                    continue
    return latest


def read_scalars(run_dir: str | Path, tag_prefix: str = "train/") -> dict[str, list[dict[str, Any]]]:
    """Return {tag: [{step, value, wall_time}, ...]} from event files (cached by mtime)."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        return {}
    mtime = _latest_event_mtime(root)
    key = str(root)
    cached = _scalar_cache.get(key)
    if cached is not None and cached[0] >= mtime:
        return _filter_by_prefix(cached[1], tag_prefix)

    data = _load_scalars(root)
    _scalar_cache[key] = (mtime, data)
    return _filter_by_prefix(data, tag_prefix)


def _filter_by_prefix(
    data: dict[str, list[dict[str, Any]]],
    tag_prefix: str,
) -> dict[str, list[dict[str, Any]]]:
    if not tag_prefix:
        return data
    return {tag: points for tag, points in data.items() if tag.startswith(tag_prefix)}


def _load_scalars(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return _read_scalars_fallback(run_dir)

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        out[tag] = [
            {"step": e.step, "value": float(e.value), "wall_time": e.wall_time}
            for e in events
        ]
    return out


def _read_scalars_fallback(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Minimal parser when tensorboard is not installed."""
    del run_dir
    return {}
