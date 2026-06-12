"""Consumer client: read a run's tracking data back for the UI.

Scalars come from TensorBoard event files (via the event accumulator, cached by mtime);
structured metadata and the timeline come from ``run.json`` / ``run_events.jsonl``.
``compare_runs`` assembles the cross-run comparison payload — one manifest row per run, the
aligned scalar series, and each run's event timeline — for the UI's comparison view.

Reading manifests/events never imports torch; only ``read_scalars`` touches the (deferred)
tensorboard event accumulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rengu_track.events import read_events
from rengu_track.run import MANIFEST_NAME, read_manifest

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


def read_scalars(run_dir: str | Path, tag_prefix: str = "") -> dict[str, list[dict[str, Any]]]:
    """Return {tag: [{step, value, wall_time}, ...]} from event files (cached by mtime).

    Default returns ALL tags (train/eval/val/system); pass a prefix to filter.
    """
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
        return {}

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = [
            {"step": e.step, "value": float(e.value), "wall_time": e.wall_time}
            for e in acc.Scalars(tag)
        ]
    return out


# --- cross-run comparison ---------------------------------------------------------------------


def run_row(run_dir: str | Path) -> dict[str, Any] | None:
    """One manifest-derived comparison row (no scalars), or None if the run has no manifest."""
    manifest = read_manifest(run_dir)
    if manifest is None:
        return None
    return {
        "run_id": manifest.run_id,
        "name": manifest.name,
        "status": manifest.status,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "hparams": manifest.hparams_flat,
        "summary": manifest.summary,
        "system_summary": manifest.system_summary,
        "lineage": manifest.lineage,
        "hardware": manifest.hardware,
    }


def list_run_dirs(output_dir: str | Path) -> list[Path]:
    """Run directories under ``output_dir`` that carry a tracking manifest."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    return sorted(
        (child for child in root.iterdir() if child.is_dir() and (child / MANIFEST_NAME).is_file()),
        key=lambda p: p.name,
    )


def _hparam_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union of hparam keys across runs, flagging which ones differ (for highlighting)."""
    seen: dict[str, set] = {}
    for row in rows:
        for key, value in row["hparams"].items():
            seen.setdefault(key, set()).add(repr(value))
    return [{"key": key, "varies": len(values) > 1} for key, values in sorted(seen.items())]


def compare_runs(
    run_dirs: list[str | Path],
    *,
    tag_prefix: str = "",
    include_series: bool = True,
) -> dict[str, Any]:
    """Assemble the comparison payload: manifest rows, hparam columns, series, timelines."""
    rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, list[dict[str, Any]]]] = {}
    timelines: dict[str, list[dict[str, Any]]] = {}
    for run_dir in run_dirs:
        row = run_row(run_dir)
        if row is None:
            continue
        rid = row["run_id"]
        rows.append(row)
        timelines[rid] = read_events(run_dir)
        if include_series:
            series[rid] = read_scalars(run_dir, tag_prefix)
    return {
        "runs": rows,
        "columns": _hparam_columns(rows),
        "series": series,
        "timelines": timelines,
    }
