"""One coordinator for every dataset-caching phase: consistent logs + a monotonic bar.

Caching spans several phases (metadata scan, latent encode per size bucket, one text-
embedding pass per encoder), each of which used to print in its own format and emit
its own ``@@RFPROG@@`` markers with per-bucket current/total — so the UI bar restarted
at every bucket and whole phases were invisible. This module owns both outputs:

  * **Log lines** — one format, greppable and auditable in the captured log file::

        [cache] stage 2/4: latents — 6 buckets
        [cache]   latents 512x512x1 — 240 to encode, 15 cached
        [cache]   building iteration order
        [cache] stage 2/4 done in 3m12s — 240 encoded, 15 reused

    The encoded/reused split is the audit trail for "did it actually reuse my cache?".

  * **Progress markers** — a single global, monotonic percent over the whole plan
    (stage index + intra-stage fraction), with ``stage``/``stages``/``stage_name``/
    ``detail`` fields so the UI can label the bar instead of bouncing it.

The active coordinator is process-global (like tqdm): deep call sites
(``SizeBucketDataset.cache_latents``, ``_map_and_cache``) reach it via
:func:`get_active` without threading it through four layers of signatures. When no
coordinator is active (unit tests, direct calls) :func:`note` falls back to a plain
print so behavior stays observable.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Optional

from rengu_flow.control.progress_stream import ProgressEmitter

_ACTIVE: Optional["CachingProgress"] = None


def get_active() -> Optional["CachingProgress"]:
    return _ACTIVE


def set_active(progress: Optional["CachingProgress"]) -> None:
    """Install the coordinator for the rest of the process (cache worker lifetime)."""
    global _ACTIVE
    _ACTIVE = progress


@contextmanager
def activate(progress: "CachingProgress"):
    global _ACTIVE
    prev = _ACTIVE
    _ACTIVE = progress
    try:
        yield progress
    finally:
        _ACTIVE = prev


def note(message: str) -> None:
    """A sub-step line in the unified format; plain print when no coordinator is active."""
    active = get_active()
    if active is not None:
        active.note(message)
    else:
        print(f"[cache]   {message}", flush=True)


def unit(detail: str):
    """Unit context on the active coordinator; a no-op context when none is active."""
    from contextlib import nullcontext

    active = get_active()
    return active.unit(detail) if active is not None else nullcontext()


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class CachingProgress:
    """Plan-based progress over the caching stages.

    ``plan`` fixes the stage list upfront (names only); each stage learns its unit
    count when it starts (bucket counts are only known after the metadata stage).
    Global percent = (completed stages + intra-stage fraction) / total stages —
    monotonic by construction, so the UI bar only ever moves forward.
    """

    def __init__(self, *, emitter: ProgressEmitter | None = None, quiet: bool = False):
        self._emitter = emitter
        self._quiet = quiet
        self._stages: list[str] = []
        self._stage_idx = -1  # index of the running stage
        self._stage_name = ""
        self._units_total = 0
        self._units_done = 0
        self._unit_fraction = 0.0
        self._detail = ""
        self._stage_started = 0.0
        self._encoded = 0
        self._reused = 0
        self._max_percent = 0.0

    # ---- planning / lifecycle ---------------------------------------------------

    def plan(self, stage_names: list[str]) -> None:
        self._stages = list(stage_names)

    @contextmanager
    def stage(self, name: str, units: int):
        self._stage_idx += 1
        self._stage_name = name
        self._units_total = max(0, int(units))
        self._units_done = 0
        self._unit_fraction = 0.0
        self._detail = ""
        self._encoded = 0
        self._reused = 0
        self._stage_started = time.monotonic()
        unit_word = "buckets" if self._units_total != 1 else "bucket"
        self._log(
            f"[cache] stage {self._stage_idx + 1}/{len(self._stages)}: {name}"
            + (f" — {self._units_total} {unit_word}" if self._units_total else "")
        )
        self._emit(force=True)
        try:
            yield self
        finally:
            elapsed = _fmt_duration(time.monotonic() - self._stage_started)
            counts = ""
            if self._encoded or self._reused:
                counts = f" — {self._encoded} encoded, {self._reused} reused"
            self._log(
                f"[cache] stage {self._stage_idx + 1}/{len(self._stages)}: "
                f"{name} done in {elapsed}{counts}"
            )
            self._units_done = self._units_total
            self._unit_fraction = 0.0
            self._emit(force=True)

    @contextmanager
    def unit(self, detail: str):
        """One bucket/directory inside the current stage."""
        self._detail = detail
        self._unit_fraction = 0.0
        self._emit(force=True)
        try:
            yield self
        finally:
            self._units_done = min(self._units_total, self._units_done + 1)
            self._unit_fraction = 0.0
            self._detail = ""

    # ---- reporting ----------------------------------------------------------------

    def note(self, message: str) -> None:
        self._log(f"[cache]   {message}")

    def unit_progress(self, done: int, total: int) -> None:
        """Intra-unit progress from the encode loop (batches done / total batches)."""
        if total > 0:
            self._unit_fraction = min(1.0, max(0.0, done / total))
        self._emit(force=done >= total)

    def add_encoded(self, n: int) -> None:
        self._encoded += int(n)

    def add_reused(self, n: int) -> None:
        self._reused += int(n)

    # ---- output -------------------------------------------------------------------

    @property
    def percent(self) -> float:
        stages = max(1, len(self._stages))
        if self._stage_idx < 0:
            return 0.0
        frac_in_stage = 0.0
        if self._units_total > 0:
            frac_in_stage = (self._units_done + self._unit_fraction) / self._units_total
        pct = 100.0 * (self._stage_idx + min(1.0, frac_in_stage)) / stages
        # Monotonic guarantee even if a caller reports out of order.
        self._max_percent = max(self._max_percent, min(100.0, pct))
        return self._max_percent

    def _payload(self) -> dict[str, Any]:
        return {
            "phase": "caching",
            "stage": self._stage_idx + 1,
            "stages": len(self._stages),
            "stage_name": self._stage_name,
            "detail": self._detail or None,
            "current": self._units_done,
            "total": self._units_total,
            "percent": round(self.percent, 1),
        }

    def _emit(self, *, force: bool = False) -> None:
        if self._emitter is not None:
            self._emitter.emit(self._payload(), force=force)

    def _log(self, line: str) -> None:
        if not self._quiet:
            print(line, flush=True)
