"""Render trainer ``@@RFPROG@@`` progress markers as an in-place tqdm bar for the CLI.

The trainer (and the dataset cache pass) emit throttled JSON progress markers to
stdout. When ``rengu train`` runs from a terminal we don't want those raw marker lines
(nor a per-step log chain) scrolling past — we want a single Kohya-style bar that
updates in place. This wraps the training subprocess: marker lines drive a tqdm bar and
every other line is printed above the bar via ``tqdm.write``. When stdout is not a TTY
(logs redirected to a file or captured by the web UI) we fall back to a compact one-line
summary per marker so log files stay greppable.

The trainer's own per-step tqdm bars are disabled when their stream is not a TTY; since
we pipe the subprocess output, only the marker stream reaches us and we render the bar
ourselves, so the two never fight over the terminal.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from rengu_flow.control.progress_stream import is_progress_marker, parse_progress_marker
from rengu_flow.training_progress import format_training_log_line

# Bar layout mirrors Kohya: description, percentage, bar, count, then our own stable
# postfix (smoothed loss / s-per-it / ETA). We deliberately omit tqdm's own rate so the
# numbers come from the trainer's EMA instead of jumping with each manual refresh.
_TRAIN_BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]"


def _training_postfix(payload: dict[str, Any]) -> str:
    """Stable loss / speed / ETA hint for the training bar's postfix."""
    parts: list[str] = []
    loss = payload.get("loss_avg")
    if loss is None:
        loss = payload.get("loss")
    if loss is not None:
        try:
            parts.append(f"loss={float(loss):.4f}")
        except (TypeError, ValueError):
            pass
    sit = payload.get("step_time_sec_ema") or payload.get("step_time_sec")
    if sit:
        try:
            parts.append(f"{float(sit):.2f}s/it")
        except (TypeError, ValueError):
            pass
    eta = payload.get("eta")
    if eta:
        parts.append(f"eta {eta}")
    return ", ".join(parts)


def _render_tty(stream) -> None:
    """Drive an in-place tqdm bar from the marker stream; pass other lines through."""
    from tqdm import tqdm

    bar = None
    phase: str | None = None

    def close() -> None:
        nonlocal bar
        if bar is not None:
            bar.close()
            bar = None

    def passthrough(text: str) -> None:
        if bar is not None:
            bar.write(text, file=sys.stdout)
        else:
            print(text, flush=True)

    for raw in stream:
        line = raw.rstrip("\n")
        if not is_progress_marker(line):
            passthrough(line)
            continue
        payload = parse_progress_marker(line)
        if payload is None:
            continue

        ph = payload.get("phase") or "training"
        if ph == "caching":
            total = payload.get("total")
            if phase != "caching":
                close()
                bar = tqdm(
                    total=total,
                    desc="Caching",
                    unit="batch",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    leave=True,
                )
                phase = "caching"
            if total and bar.total != total:
                bar.total = total
            bar.n = int(payload.get("current") or 0)
            bar.refresh()
        elif "step" in payload:
            total = payload.get("max_steps")
            if phase != "training":
                close()
                bar = tqdm(
                    total=total,
                    desc="Training",
                    unit="step",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    leave=True,
                    bar_format=_TRAIN_BAR_FORMAT if total else None,
                )
                phase = "training"
            if total and bar.total != total:
                bar.total = total
            bar.n = int(payload.get("step") or 0)
            bar.set_postfix_str(_training_postfix(payload))
            bar.refresh()
        else:
            # Boundary markers without progress (e.g. waiting_disk_export): surface as text.
            passthrough(f"[{ph}]")
    close()


def _render_plain(stream) -> None:
    """Non-TTY fallback: one compact line per marker, other lines verbatim."""
    for raw in stream:
        line = raw.rstrip("\n")
        if not is_progress_marker(line):
            print(line, flush=True)
            continue
        payload = parse_progress_marker(line)
        if payload is None:
            continue
        ph = payload.get("phase") or "training"
        if ph == "caching":
            cur = payload.get("current")
            total = payload.get("total")
            pct = payload.get("percent")
            pct_s = f" ({pct}%)" if pct is not None else ""
            print(f"caching {cur}/{total}{pct_s}", flush=True)
        elif "step" in payload:
            print(
                format_training_log_line(
                    step=int(payload.get("step") or 0),
                    loss=float(payload.get("loss") or 0.0),
                    epoch=int(payload.get("epoch") or 0),
                    metrics=payload,
                ),
                flush=True,
            )
        else:
            print(f"[{ph}]", flush=True)


def run_training_with_progress(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> int:
    """Run a training/cache subprocess, rendering its progress markers as a live bar.

    stderr is merged into stdout so deepspeed/torch logs interleave above the bar in
    order. Returns the subprocess exit code. SIGINT (Ctrl-C) reaches the child via the
    shared process group; we keep draining its output so its shutdown/checkpoint log and
    final marker are still shown.
    """
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    render = _render_tty if sys.stdout.isatty() else _render_plain
    assert proc.stdout is not None
    try:
        render(proc.stdout)
    except KeyboardInterrupt:
        # The child got the same SIGINT (shared foreground process group). Keep draining
        # so its checkpoint-on-quit output and final marker still reach the terminal.
        try:
            render(proc.stdout)
        except KeyboardInterrupt:
            pass
    return proc.wait()
