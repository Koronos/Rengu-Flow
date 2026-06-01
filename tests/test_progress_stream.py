"""Tests for the throttled stdout progress-marker protocol."""

from __future__ import annotations

from rengu_flow.control.progress_stream import (
    PROGRESS_MARKER_PREFIX,
    ProgressEmitter,
    format_progress_marker,
    is_progress_marker,
    parse_last_progress_marker,
    parse_progress_marker,
    strip_progress_markers,
)


def test_format_and_parse_round_trip() -> None:
    payload = {"phase": "training", "step": 5, "loss": 0.25, "percent": 50.0}
    line = format_progress_marker(payload)
    assert line.startswith(PROGRESS_MARKER_PREFIX)
    assert is_progress_marker(line)
    assert parse_progress_marker(line) == payload


def test_parse_rejects_non_marker_lines() -> None:
    assert parse_progress_marker("normal log line") is None
    assert parse_progress_marker(f"{PROGRESS_MARKER_PREFIX} not-json") is None
    assert parse_progress_marker(f"{PROGRESS_MARKER_PREFIX} [1,2,3]") is None
    assert parse_progress_marker(PROGRESS_MARKER_PREFIX) is None


def test_parse_last_marker_ignores_trailing_partial() -> None:
    a = format_progress_marker({"step": 1})
    b = format_progress_marker({"step": 2})
    # Only complete (newline-terminated) lines are parsed; the partial b is ignored.
    text = f"hello\n{a}\n{b}"
    assert parse_last_progress_marker(text) == {"step": 1}
    # Once b completes, it becomes the last marker.
    assert parse_last_progress_marker(text + "\n") == {"step": 2}


def test_parse_last_marker_none_without_complete_line() -> None:
    assert parse_last_progress_marker("") is None
    assert parse_last_progress_marker("no newline yet") is None


def test_strip_progress_markers_removes_complete_and_trailing_partial() -> None:
    marker = format_progress_marker({"step": 1})
    text = f"line one\n{marker}\nline two\n"
    assert strip_progress_markers(text) == "line one\nline two\n"

    # A trailing partial marker (no newline yet) is suppressed too.
    partial = f"line one\n{marker[:20]}"
    assert strip_progress_markers(partial) == "line one\n"

    # No markers: text is returned unchanged.
    assert strip_progress_markers("a\nb\n") == "a\nb\n"


def test_emitter_throttles_but_forces() -> None:
    now = [0.0]
    written: list[str] = []
    emitter = ProgressEmitter(
        min_interval_sec=1.0,
        clock=lambda: now[0],
        write=written.append,
    )
    assert emitter.emit({"step": 1}) is True  # first always emits
    assert emitter.emit({"step": 2}) is False  # within throttle window
    now[0] = 0.5
    assert emitter.emit({"step": 3}) is False  # still within window
    assert emitter.emit({"step": 4}, force=True) is True  # force bypasses throttle
    now[0] = 1.6
    assert emitter.emit({"step": 5}) is True  # window elapsed
    assert len(written) == 3
    assert all(is_progress_marker(w) for w in written)
