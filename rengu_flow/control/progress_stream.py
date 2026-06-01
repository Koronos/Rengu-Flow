"""Throttled stdout progress markers shared by the trainer and the web UI backend.

The trainer emits one compact single-line JSON marker to stdout, prefixed with a
distinctive token unlikely to appear in normal logs. The web UI backend parses the
last complete marker from the captured log to drive its live progress bar and strips
all marker lines from the log text it shows the user.

This replaces per-iteration ``status.json`` writes: nothing is written to disk per
training step (only tensors/checkpoints). TensorBoard scalar logging is unchanged.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

# Distinctive line prefix. Marker lines look like:
#   @@RFPROG@@ {"phase":"training","step":123,"max_steps":5000,...}
PROGRESS_MARKER_PREFIX = "@@RFPROG@@"


class ProgressEmitter:
    """Time-throttle progress markers to stdout (rank 0 only).

    ``min_interval_sec`` caps the emit rate (default ~1/sec). ``force=True`` always
    emits regardless of the throttle window — use it on the final step and on
    save/epoch/phase boundaries so the UI never misses a meaningful transition.
    """

    def __init__(
        self,
        *,
        min_interval_sec: float = 1.0,
        clock: Callable[[], float] | None = None,
        write: Callable[[str], None] | None = None,
    ) -> None:
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._clock = clock or time.monotonic
        self._write = write or print
        self._last_emit: float | None = None

    def emit(self, payload: dict[str, Any], *, force: bool = False) -> bool:
        """Emit a marker line if the throttle window has elapsed (or ``force``).

        Returns True when a line was written. ``payload`` is the JSON body (the
        prefix is added here).
        """
        now = self._clock()
        if not force and self._last_emit is not None:
            if now - self._last_emit < self.min_interval_sec:
                return False
        self._last_emit = now
        self._write(format_progress_marker(payload))
        return True


def format_progress_marker(payload: dict[str, Any]) -> str:
    """Render one marker line: ``@@RFPROG@@ {json}`` (no trailing newline)."""
    return f"{PROGRESS_MARKER_PREFIX} {json.dumps(payload, separators=(',', ':'))}"


def is_progress_marker(line: str) -> bool:
    """True if ``line`` (with or without surrounding whitespace) is a marker line."""
    return line.lstrip().startswith(PROGRESS_MARKER_PREFIX)


def parse_progress_marker(line: str) -> dict[str, Any] | None:
    """Parse one marker line into its payload dict, or None if not a valid marker."""
    stripped = line.strip()
    if not stripped.startswith(PROGRESS_MARKER_PREFIX):
        return None
    body = stripped[len(PROGRESS_MARKER_PREFIX) :].strip()
    if not body:
        return None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def parse_last_progress_marker(text: str) -> dict[str, Any] | None:
    """Return the payload of the LAST complete marker line in ``text``.

    Only complete lines (terminated by ``\\n``) are considered, so a partial marker
    still being written is ignored until it completes. Returns None when there is no
    complete marker.
    """
    newline = text.rfind("\n")
    if newline == -1:
        # No complete line yet; never parse a trailing partial.
        return None
    complete = text[: newline + 1]
    for line in reversed(complete.splitlines()):
        if is_progress_marker(line):
            parsed = parse_progress_marker(line)
            if parsed is not None:
                return parsed
    return None


def strip_progress_markers(text: str) -> str:
    """Remove all marker lines from ``text`` for display.

    Complete marker lines (including their trailing newline) are dropped. A trailing
    partial marker line (no newline yet) is also suppressed so a half-written marker
    never flashes in the log; it will be re-emitted complete on the next tick.
    """
    if PROGRESS_MARKER_PREFIX not in text:
        return text
    # splitlines(keepends=True) preserves each line's own terminator, so dropping a
    # marker line never disturbs the newline that belongs to a preceding kept line.
    # A trailing partial marker (last element with no terminator) is dropped too.
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if is_progress_marker(line):
            continue
        out.append(line)
    return "".join(out)
