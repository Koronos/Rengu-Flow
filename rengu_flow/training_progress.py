"""Training step timing, ETA, and progress metrics (shared by trainer and UI)."""

from __future__ import annotations

import math
import time
from typing import Any, Callable


def resolve_target_steps(
    max_steps: int | None,
    total_steps: int | None,
) -> int | None:
    """Global step budget: explicit max_steps, else epochs-derived total_steps."""
    if max_steps is not None:
        try:
            ms = int(max_steps)
            if ms > 0:
                return ms
        except (TypeError, ValueError):
            pass
    if total_steps is not None:
        try:
            ts = int(total_steps)
            if ts > 0:
                return ts
        except (TypeError, ValueError):
            pass
    return None


def format_eta(seconds: float | int | None) -> str | None:
    """Human-readable ETA like Kohya/tqdm: ``1h 23m``, ``45s``, ``<1s``."""
    if seconds is None:
        return None
    try:
        total = int(math.ceil(float(seconds)))
    except (TypeError, ValueError):
        return None
    if total < 0:
        return None
    if total < 1:
        return "<1s"
    parts: list[str] = []
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs and not hours:
        parts.append(f"{secs}s")
    elif secs and hours and not minutes:
        parts.append(f"{secs}s")
    return " ".join(parts) if parts else "<1s"


def format_steps_per_second(rate: float | None, *, digits: int = 2) -> str | None:
    if rate is None or rate <= 0 or not math.isfinite(rate):
        return None
    return f"{rate:.{digits}f}"


class TrainingProgressTracker:
    """Track per-step duration, EMA step time, speed, and ETA.

    Kohya sd-scripts uses ``tqdm(..., smoothing=0)`` for the step bar: rate and ETA
    come from the most recent interval between updates (no EMA). We expose both an
    instant rate (``steps_per_second``) and an EMA-smoothed rate (``steps_per_second_ema``)
    for steadier UI display.
    """

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        total_steps: int | None = None,
        ema_alpha: float = 0.1,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.target_steps = resolve_target_steps(max_steps, total_steps)
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._clock = clock or time.perf_counter
        self._ema_step_sec: float | None = None
        self._last_step_sec: float | None = None

    def record_step_duration(self, duration_sec: float) -> None:
        """Update timing EMA from one completed training step (seconds)."""
        if duration_sec <= 0 or not math.isfinite(duration_sec):
            return
        self._last_step_sec = duration_sec
        if self._ema_step_sec is None:
            self._ema_step_sec = duration_sec
            return
        a = self.ema_alpha
        self._ema_step_sec = a * duration_sec + (1.0 - a) * self._ema_step_sec

    def metrics(self, *, step: int) -> dict[str, Any]:
        """Progress fields to merge into status.json and log lines."""
        out: dict[str, Any] = {}
        target = self.target_steps
        if target is not None:
            out["max_steps"] = target

        remaining: int | None = None
        if target is not None and target > 0:
            remaining = max(0, target - int(step))
            out["steps_remaining"] = remaining
            out["percent"] = round(min(100.0, 100.0 * float(step) / float(target)), 1)

        if self._last_step_sec and self._last_step_sec > 0:
            out["step_time_sec"] = round(self._last_step_sec, 4)
            out["steps_per_second"] = round(1.0 / self._last_step_sec, 4)

        if self._ema_step_sec and self._ema_step_sec > 0:
            out["steps_per_second_ema"] = round(1.0 / self._ema_step_sec, 4)
            if remaining is not None and remaining > 0:
                eta_sec = remaining * self._ema_step_sec
                out["eta_sec"] = int(math.ceil(eta_sec))
                eta_hr = format_eta(out["eta_sec"])
                if eta_hr:
                    out["eta"] = eta_hr

        return out


def format_training_log_line(
    *,
    step: int,
    loss: float,
    epoch: int,
    metrics: dict[str, Any],
) -> str:
    """Single-line training progress for stdout (rank 0, every logging_steps)."""
    parts: list[str] = []
    target = metrics.get("max_steps")
    if target:
        pct = metrics.get("percent")
        pct_s = f" ({pct}%)" if pct is not None else ""
        parts.append(f"step={step}/{target}{pct_s}")
    else:
        parts.append(f"step={step}")

    parts.append(f"loss={loss:.6f}")

    sps = metrics.get("steps_per_second")
    sps_ema = metrics.get("steps_per_second_ema")
    if sps is not None:
        speed = f"speed={sps} step/s"
        if sps_ema is not None and sps_ema != sps:
            speed += f" (ema {sps_ema})"
        parts.append(speed)

    remaining = metrics.get("steps_remaining")
    if remaining is not None:
        parts.append(f"remaining={remaining}")

    eta = metrics.get("eta")
    if eta:
        parts.append(f"eta={eta}")

    parts.append(f"epoch={epoch}")
    return " ".join(parts)
