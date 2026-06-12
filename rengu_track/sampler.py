"""Background system-metrics sampler: pushes ``system/*`` scalars to the sink over the run.

A daemon thread samples ``rengu_track.system_stats`` at a fixed interval and logs the primary
GPU's util / VRAM / temp / power plus host CPU/RAM as ``system/*`` scalars (so they show up as
curves in TB alongside loss). It tracks running aggregates (peak VRAM, mean util) and flushes
them to the manifest summary on ``stop()``. Sampling is best-effort — a failed sample is
skipped, never fatal — and the whole thing is inert when tracking is disabled (the caller simply
never starts it).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from rengu_track.system_stats import collect_system_stats

logger = logging.getLogger("rengu_track")


class SystemSampler:
    def __init__(
        self,
        sink: Any,
        *,
        interval_sec: float = 10.0,
        step_fn: Callable[[], int] | None = None,
    ) -> None:
        self._sink = sink
        self._interval = max(1.0, float(interval_sec))
        self._step_fn = step_fn or (lambda: 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_vram_gb = 0.0
        self._util_sum = 0.0
        self._util_n = 0
        self._samples = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="rengu-track-sampler", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        # Sample immediately, then every interval; the stop event makes the wait interruptible.
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._interval)

    def _current_step(self) -> int:
        try:
            return int(self._step_fn())
        except Exception:
            return 0

    def _sample_once(self) -> None:
        try:
            stats = collect_system_stats(sample_cpu=True)
        except Exception as exc:
            logger.debug("system sampler: collect failed: %s", exc)
            return
        step = self._current_step()
        summary = stats.get("summary", {}) or {}

        if summary.get("cpu_percent") is not None:
            self._sink.scalar("system/cpu_percent", summary["cpu_percent"], step)
        if summary.get("ram_used_gb") is not None:
            self._sink.scalar("system/ram_used_gb", summary["ram_used_gb"], step)

        gpus = summary.get("gpus") or []
        if gpus:
            primary = gpus[0]
            util = primary.get("util_percent")
            if util is not None:
                self._sink.scalar("system/gpu_util_percent", util, step)
                self._util_sum += util
                self._util_n += 1
            vram = primary.get("vram_used_gb")
            if vram is not None:
                self._sink.scalar("system/vram_used_gb", vram, step)
                self._peak_vram_gb = max(self._peak_vram_gb, vram)
            temp = primary.get("temp_c")
            if temp is not None:
                self._sink.scalar("system/gpu_temp_c", temp, step)

        # power_w lives in the detailed device record, not the compact summary.
        devices = ((stats.get("detail", {}) or {}).get("gpus", {}) or {}).get("devices") or []
        if devices and devices[0].get("power_w") is not None:
            self._sink.scalar("system/gpu_power_w", devices[0]["power_w"], step)

        self._samples += 1

    def _aggregates(self) -> dict[str, Any]:
        if not self._samples:
            return {}
        out: dict[str, Any] = {"system/peak_vram_gb": round(self._peak_vram_gb, 2)}
        if self._util_n:
            out["system/mean_gpu_util_percent"] = round(self._util_sum / self._util_n, 1)
        return out

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self._interval + 2.0)
            self._thread = None
        aggregates = self._aggregates()
        if aggregates:
            try:
                self._sink.summary(aggregates)
            except Exception:
                pass
