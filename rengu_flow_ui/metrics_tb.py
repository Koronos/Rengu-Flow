"""TensorBoard scalar reading — thin shim over rengu_track.reader.

The scalar-reading logic now lives in the tracking core (rengu_track.reader) so both the UI and
any other consumer share one cached implementation. This module preserves the historical
``train/``-prefixed default for the existing per-run metrics endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rengu_track.reader import invalidate_scalars_cache
from rengu_track.reader import scalars_for_run as _scalars_for_run

__all__ = ["read_scalars", "invalidate_scalars_cache"]


def read_scalars(
    run_dir: str | Path,
    tag_prefix: str = "train/",
    *,
    max_points: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {tag: [{step, value, wall_time}, ...]} (defaults to train/* for the metrics view).

    Served by the Rust data server (same fast path as the comparison view) with an EventAccumulator
    fallback. ``max_points`` downsamples each series (the detail view caps to keep long runs snappy).
    """
    return _scalars_for_run(run_dir, tag_prefix, max_points=max_points)
