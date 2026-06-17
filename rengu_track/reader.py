"""Consumer client: read a run's tracking data back for the UI.

Scalars come from TensorBoard event files (via the event accumulator, cached by mtime);
structured metadata and the timeline come from ``run.json`` / ``run_events.jsonl``.
``compare_runs`` assembles the cross-run comparison payload — one manifest row per run, the
aligned scalar series, and each run's event timeline — for the UI's comparison view.

Reading manifests/events never imports torch; only ``read_scalars`` touches the (deferred)
tensorboard event accumulator.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Callable

from rengu_track.events import read_events
from rengu_track.run import MANIFEST_NAME, read_manifest

# Preview frames are written by the trainer as ``step{NNNNNNNN}_{prompt}.ext`` (step first,
# zero-padded, so a file browser sorts them chronologically). Parsing here — not in the UI —
# keeps the step/prompt structure authoritative.
_PREVIEW_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_PREVIEW_RE = re.compile(r"^step(\d+)_(.+)\.(?:png|jpe?g|webp)$", re.IGNORECASE)

# run_dir key -> (latest event mtime, parsed scalars)
_scalar_cache: dict[str, tuple[float, dict[str, list[dict[str, Any]]]]] = {}

# A short-lived lock PER run dir, plus a meta-lock guarding the cache/lock dicts. The Compare view
# mounts one chart per metric, each firing a series request at once for the SAME run set; without
# this the concurrent requests all miss the not-yet-populated cache and each launches a full
# EventAccumulator.Reload() of the same event files — N×M parallel re-parses that pin the GIL.
# Per-run (not global) locks collapse that stampede to one parse per run while still letting
# DIFFERENT runs parse in parallel, so an N-run comparison loads concurrently instead of serially.
_cache_lock = threading.Lock()
_run_locks: dict[str, threading.Lock] = {}


def _run_lock(key: str) -> threading.Lock:
    with _cache_lock:
        lock = _run_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _run_locks[key] = lock
        return lock

# Default cap on points returned per scalar tag. Charts are a few hundred px wide, so loading
# every step is wasted work; the full series stays cached and is downsampled on the way out.
DEFAULT_MAX_POINTS = 500

# Offered for runs that have no manifest tag index (TB-only / pre-tracking). The lazy series
# fetch reads these from the event files on demand; tags a run doesn't have render empty.
DEFAULT_COMPARE_METRICS = (
    "train/loss",
    "train/grad_norm",
    "val/loss",
    "val/gap",
    "system/vram_used_gb",
)


def _downsample(series: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    """Uniform-stride downsample to <= max_points, always keeping the first and last point."""
    n = len(series)
    if max_points <= 0 or n <= max_points:
        return series
    if max_points <= 2:
        return [series[0], series[-1]]
    step = (n - 1) / (max_points - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_points)})
    return [series[i] for i in idxs]


def invalidate_scalars_cache(run_dir: str | Path | None = None) -> None:
    """Drop cached scalars (and the persistent accumulator) for one run or all runs."""
    if run_dir is None:
        _scalar_cache.clear()
        _accumulators.clear()
        return
    key = str(Path(run_dir).resolve())
    _scalar_cache.pop(key, None)
    _accumulators.pop(key, None)


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


def read_scalars(
    run_dir: str | Path,
    tag_prefix: str = "",
    *,
    max_points: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {tag: [{step, value, wall_time}, ...]} from event files (cached by mtime).

    Default returns ALL tags (train/eval/val/system); pass a prefix to filter. ``max_points``
    downsamples each series on return (the full series stays cached for other callers).
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        return {}
    mtime = _latest_event_mtime(root)
    key = str(root)

    def _cached_fresh() -> dict[str, list[dict[str, Any]]] | None:
        with _cache_lock:
            entry = _scalar_cache.get(key)
        return entry[1] if entry is not None and entry[0] >= mtime else None

    data = _cached_fresh()
    if data is None:
        # Only one thread per run parses; others wait here, then take the result from cache via the
        # re-check below. Different runs hold different locks, so they parse concurrently.
        with _run_lock(key):
            data = _cached_fresh()
            if data is None:
                data = _load_scalars(root)
                with _cache_lock:
                    _scalar_cache[key] = (mtime, data)

    # Filter/downsample outside the lock: both build fresh lists from the cached data without
    # mutating it, so they're safe to run concurrently once the parse is cached.
    filtered = _filter_by_prefix(data, tag_prefix)
    if max_points is None:
        return filtered
    return {tag: _downsample(points, max_points) for tag, points in filtered.items()}


def _filter_by_prefix(
    data: dict[str, list[dict[str, Any]]],
    tag_prefix: str,
) -> dict[str, list[dict[str, Any]]]:
    if not tag_prefix:
        return data
    return {tag: points for tag, points in data.items() if tag.startswith(tag_prefix)}


# Cap on scalar events kept PER TAG when reading an event file (TensorBoard's own default). The
# previous ``0`` meant "unlimited", so a long run that logs a scalar every step balloons memory and
# parse time without bound. Charts only render a few hundred points (the series is downsampled to
# ``max_points`` on the way out), so a 10k-per-tag reservoir is already far more than is shown.
_SCALARS_PER_TAG_CAP = 10000

# Persistent EventAccumulator per run dir. ``Reload()`` is INCREMENTAL — it continues from the last
# file offset, so re-reading a growing (live) run costs ~1ms instead of re-parsing the whole stream.
# This is what makes TensorBoard fast on these same files; recreating the accumulator every call (as
# before) re-decoded the entire multi-MB stream (~4.5s, dominated by embedded preview-image bytes)
# on every poll of a running run. Accessed only under the caller's per-run lock (not thread-safe).
_accumulators: dict[str, Any] = {}


def _load_scalars(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    key = str(run_dir)
    acc = _accumulators.get(key)
    if acc is None:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return {}
        acc = EventAccumulator(key, size_guidance={"scalars": _SCALARS_PER_TAG_CAP})
        _accumulators[key] = acc
    try:
        acc.Reload()  # incremental: only events appended since the last call are parsed
    except Exception:
        # Keep whatever the accumulator already holds from earlier reloads rather than dropping it.
        pass
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
        "tags": manifest.scalar_tags,
        "last_scalars": manifest.last_scalars,
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
    include_series: bool = False,
    max_points: int | None = DEFAULT_MAX_POINTS,
    config_fallback: Callable[[str | Path], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Assemble the comparison payload from manifests/timelines — NO event-file parsing by default.

    ``metrics`` is the union of each run's manifest scalar tags, so the UI can list every metric
    and fetch each series on demand (see ``series_for``). Pass ``include_series=True`` to eagerly
    embed downsampled series (parses event files; avoid for large run sets).

    ``config_fallback`` builds a row for a run that has no ``run.json`` (e.g. trained before
    tracking) — typically from its config TOML. Such runs have no cheap tag index, so the curated
    ``DEFAULT_COMPARE_METRICS`` are offered for them (their series still load lazily; absent tags
    just render empty).
    """
    kept: list[tuple[str | Path, dict[str, Any]]] = []
    timelines: dict[str, list[dict[str, Any]]] = {}
    metrics: set[str] = set()
    any_unindexed = False
    for run_dir in run_dirs:
        row = run_row(run_dir)
        if row is None and config_fallback is not None:
            row = config_fallback(run_dir)
            if row is not None and not row.get("tags"):
                any_unindexed = True
        if row is None:
            continue
        kept.append((run_dir, row))
        timelines[row["run_id"]] = read_events(run_dir)
        metrics.update(row.get("tags") or [])
    if any_unindexed:
        metrics.update(DEFAULT_COMPARE_METRICS)
    rows = [row for _, row in kept]
    payload: dict[str, Any] = {
        "runs": rows,
        "columns": _hparam_columns(rows),
        "metrics": sorted(metrics),
        "timelines": timelines,
    }
    if include_series:
        payload["series"] = {
            row["run_id"]: read_scalars(run_dir, tag_prefix, max_points=max_points)
            for run_dir, row in kept
        }
    return payload


def series_for(
    run_dirs: list[str | Path],
    tag: str,
    *,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> dict[str, list[dict[str, Any]]]:
    """On-demand: load ONE metric's downsampled series for each run → {run_id: [points]}.

    This is the lazy path the comparison UI calls per metric. Only the runs being compared are
    touched, and only when their chart is actually viewed; the persistent accumulator makes repeated
    per-metric fetches for the same run cheap (the run is parsed once).
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for run_dir in run_dirs:
        manifest = read_manifest(run_dir)
        rid = manifest.run_id if manifest is not None else Path(run_dir).name
        out[rid] = read_scalars(run_dir, max_points=max_points).get(tag, [])
    return out


def _run_id_for(run_dir: str | Path) -> str:
    manifest = read_manifest(run_dir)
    return manifest.run_id if manifest is not None else Path(run_dir).name


def series_for_tag(
    output_dir: str | Path,
    run_names: list[str],
    tag: str,
    *,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> dict[str, list[dict[str, Any]]]:
    """One metric's series per run → {run_id: [points]}, served by the Rust data server.

    Delegates to TensorBoard's ``tensorboard-data-server`` (Rust: parses in parallel, downsamples
    and reloads incrementally), the same fast path TensorBoard uses — ~0.8s vs ~4.5s for the Python
    EventAccumulator on an image-heavy run. Falls back to the per-run EventAccumulator path
    (:func:`series_for`) when the data server is unavailable.

    ``output_dir`` is the logdir the server indexes; ``run_names`` are dir names under it (empty =
    all runs). Run dir name == ``run_id`` for tracked runs, so results are keyed by ``run_id``.
    """
    root = Path(output_dir)
    run_dirs = [root / n for n in run_names] if run_names else list_run_dirs(root)
    cap = max_points if max_points and max_points > 0 else DEFAULT_MAX_POINTS
    try:
        from rengu_track import scalar_server

        by_name = scalar_server.read_scalars(
            root, [Path(d).name for d in run_dirs], tags=[tag], max_points=cap
        )
    except Exception:  # noqa: BLE001 - DataServerUnavailable or any glue failure -> EventAccumulator
        return series_for(run_dirs, tag, max_points=max_points)
    return {_run_id_for(d): by_name.get(Path(d).name, {}).get(tag, []) for d in run_dirs}


def scalars_for_run(
    run_dir: str | Path,
    tag_prefix: str = "",
    *,
    max_points: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """All scalar tags for ONE run → {tag: [points]}, served by the Rust data server.

    The single-run analogue of :func:`series_for_tag`: the run-detail metrics view (and its live
    WebSocket snapshot) read every ``train/*`` scalar for one run, so route that through the same
    fast data server instead of the Python EventAccumulator. Falls back to the (cached, incremental)
    EventAccumulator path on any data-server failure. With ``max_points`` unset the full series is
    returned via EventAccumulator, since the data server always samples server-side.
    """
    if not max_points or max_points <= 0:
        return read_scalars(run_dir, tag_prefix, max_points=max_points)
    root = Path(run_dir)
    try:
        from rengu_track import scalar_server

        by_name = scalar_server.read_scalars(root.parent, [root.name], max_points=max_points)
    except Exception:  # noqa: BLE001 - DataServerUnavailable or any glue failure -> EventAccumulator
        return read_scalars(run_dir, tag_prefix, max_points=max_points)
    return _filter_by_prefix(by_name.get(root.name, {}), tag_prefix)


def preview_images(run_dir: str | Path, *, limit: int = 2000) -> list[dict[str, Any]]:
    """List a run's preview frames with parsed ``step`` and ``prompt`` so the UI can show the
    evolution along steps (group by prompt, order by step) instead of a flat dump.

    Returns ``[{name, run_dir, step, prompt}]`` — ``step`` is None for files that don't follow the
    ``step{N}_{prompt}`` convention. Capped to the most recent ``limit`` frames (highest steps).
    """
    root = Path(run_dir).resolve()
    preview = root / "preview"
    if not preview.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in preview.iterdir():
        if not path.is_file() or path.suffix.lower() not in _PREVIEW_EXTS:
            continue
        match = _PREVIEW_RE.match(path.name)
        if match:
            step: int | None = int(match.group(1))
            prompt = match.group(2)
        else:
            step = None
            prompt = path.stem
        items.append({"name": path.name, "run_dir": str(root), "step": step, "prompt": prompt})
    # Keep the newest frames when capping (unparsed sort last); the UI re-sorts ascending per prompt.
    items.sort(key=lambda it: it["step"] if it["step"] is not None else -1, reverse=True)
    return items[:limit]
