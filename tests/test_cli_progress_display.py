"""Unit tests for the CLI progress-bar renderer (marker stream -> bar / plain lines)."""

from __future__ import annotations

import io

from rengu_flow.cli import progress_display
from rengu_flow.control.progress_stream import format_progress_marker


def test_training_postfix_prefers_smoothed() -> None:
    postfix = progress_display._training_postfix(
        {"loss": 0.42, "loss_avg": 0.40, "step_time_sec": 9.9, "step_time_sec_ema": 2.5, "eta": "5m"}
    )
    assert "loss=0.4000" in postfix
    assert "2.50s/it" in postfix
    assert "eta 5m" in postfix


def test_training_postfix_falls_back_to_instant() -> None:
    postfix = progress_display._training_postfix(
        {"loss": 0.42, "step_time_sec": 3.0}
    )
    assert "loss=0.4200" in postfix
    assert "3.00s/it" in postfix


def test_render_plain_summarizes_markers_and_passes_logs(capsys) -> None:
    lines = [
        "loading model...\n",
        format_progress_marker(
            {
                "phase": "training",
                "step": 5,
                "loss": 0.3,
                "epoch": 1,
                "max_steps": 100,
                "percent": 5.0,
                "loss_avg": 0.31,
                "step_time_sec_ema": 2.0,
                "eta": "3m",
            }
        )
        + "\n",
        "deepspeed note\n",
        format_progress_marker({"phase": "caching", "current": 4, "total": 8, "percent": 50.0})
        + "\n",
    ]
    progress_display._render_plain(io.StringIO("".join(lines)))
    out = capsys.readouterr().out
    # Real log lines pass through verbatim.
    assert "loading model..." in out
    assert "deepspeed note" in out
    # Training marker becomes one compact summary line (smoothed values).
    assert "step=5/100" in out
    assert "avr_loss=0.310000" in out
    assert "speed=2.00 s/it" in out
    # Caching marker becomes a compact caching line.
    assert "caching 4/8 (50.0%)" in out
    # No raw marker prefix leaks into the output.
    assert "@@RFPROG@@" not in out
