"""Training step timing, ETA, and progress metrics (shared by trainer and UI)."""

from __future__ import annotations

import math
import time
from collections import deque
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
    """Track per-step duration, EMA step time, speed, smoothed loss, and ETA.

    Kohya sd-scripts uses ``tqdm(..., smoothing=0)`` for the step bar but displays a
    smoothed ``avr_loss`` (a moving average over one epoch's worth of steps via its
    ``LossRecorder``). Raw per-step speed and loss both jump around a lot, which makes
    the ETA hard to read, so we expose smoothed values for display:

    - ``steps_per_second`` / ``step_time_sec``: instant (last interval).
    - ``steps_per_second_ema`` / ``step_time_sec_ema``: EMA-smoothed; ETA is derived
      from the EMA so it drifts instead of jumping.
    - ``loss`` (added by the payload builder) is the instant loss; ``loss_avg`` is a
      Kohya-style moving average over the last ``loss_window`` steps.
    """

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        total_steps: int | None = None,
        ema_alpha: float = 0.1,
        loss_window: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.target_steps = resolve_target_steps(max_steps, total_steps)
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._clock = clock or time.perf_counter
        self._ema_step_sec: float | None = None
        self._last_step_sec: float | None = None
        # Kohya averages loss over one epoch of steps; clamp to a sane range so the
        # window stays responsive (small epochs) without unbounded memory (huge epochs).
        window = 50 if loss_window is None else int(loss_window)
        self._loss_window = max(1, min(window, 1000))
        self._loss_buf: deque[float] = deque(maxlen=self._loss_window)
        self._loss_sum: float = 0.0

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

    def record_loss(self, loss: float) -> None:
        """Push one step's loss into the moving-average window (Kohya ``avr_loss``)."""
        try:
            value = float(loss)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        if len(self._loss_buf) == self._loss_buf.maxlen:
            self._loss_sum -= self._loss_buf[0]
        self._loss_buf.append(value)
        self._loss_sum += value

    @property
    def loss_avg(self) -> float | None:
        if not self._loss_buf:
            return None
        return self._loss_sum / len(self._loss_buf)

    def metrics(self, *, step: int) -> dict[str, Any]:
        """Progress fields to merge into the stdout progress marker payload."""
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
            out["step_time_sec_ema"] = round(self._ema_step_sec, 4)
            out["steps_per_second_ema"] = round(1.0 / self._ema_step_sec, 4)
            if remaining is not None and remaining > 0:
                eta_sec = remaining * self._ema_step_sec
                out["eta_sec"] = int(math.ceil(eta_sec))
                eta_hr = format_eta(out["eta_sec"])
                if eta_hr:
                    out["eta"] = eta_hr

        avg = self.loss_avg
        if avg is not None and math.isfinite(avg):
            out["loss_avg"] = round(avg, 6)

        return out


def budget_display_epoch(step: int, steps_per_epoch: int, epochs: int) -> int:
    """Budget-relative epoch (1..epochs) for the progress display.

    With a resolution schedule, dataloader epochs are short (each stage trains a subset
    of resolutions), so the raw epoch counter overshoots the configured ``epochs``. The
    total step budget is still ``epochs * steps_per_epoch``, so the meaningful epoch
    number is derived from the current step within that budget and capped at ``epochs``.
    """
    if steps_per_epoch <= 0:
        return int(step)
    return max(1, min(int(epochs), (int(step) - 1) // int(steps_per_epoch) + 1))


def build_progress_payload(
    *,
    step: int,
    loss: float,
    epoch: int,
    metrics: dict[str, Any],
    phase: str = "training",
    val_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compact progress payload for the throttled stdout marker.

    Reuses the numeric fields from ``TrainingProgressTracker.metrics()`` and adds the
    per-step loss, current epoch, and phase. Mirrors the fields the UI bar consumes.

    ``val_metrics`` (when present) carries the latest deterministic generalization probe —
    ``val_loss``, optional ``train_probe`` and ``val_gap`` (the overfitting signal) — so the
    UI can show the held-out loss and the train-val gap next to the train loss.
    """
    payload: dict[str, Any] = {
        "phase": phase,
        "step": int(step),
        "loss": round(float(loss), 6),
        "epoch": int(epoch),
    }
    payload.update(metrics)
    if val_metrics:
        if "val_loss" in val_metrics:
            payload["val_loss"] = round(float(val_metrics["val_loss"]), 6)
        if "val_gap" in val_metrics:
            payload["val_gap"] = round(float(val_metrics["val_gap"]), 6)
        if "train_probe" in val_metrics:
            payload["train_probe"] = round(float(val_metrics["train_probe"]), 6)
    return payload


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

    loss_avg = metrics.get("loss_avg")
    if loss_avg is not None:
        parts.append(f"avr_loss={loss_avg:.6f}")
    else:
        parts.append(f"loss={loss:.6f}")

    # Kohya-style display: prefer the EMA-smoothed s/it so the speed reads steadily.
    sit_ema = metrics.get("step_time_sec_ema")
    sps = metrics.get("steps_per_second")
    if sit_ema is not None and sit_ema > 0:
        parts.append(f"speed={sit_ema:.2f} s/it")
    elif sps is not None:
        parts.append(f"speed={sps} step/s")

    remaining = metrics.get("steps_remaining")
    if remaining is not None:
        parts.append(f"remaining={remaining}")

    eta = metrics.get("eta")
    if eta:
        parts.append(f"eta={eta}")

    parts.append(f"epoch={epoch}")
    return " ".join(parts)
