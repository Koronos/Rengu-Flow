"""The producer client: one ``MetricsSink`` object that replaces the ``tb_writer`` +
``wandb_enable`` pair threaded through the training loop.

The sink fans every call out to its configured backends and records lifecycle events to the
per-run timeline. Backend failures NEVER propagate — tracking is auxiliary and must not crash a
run — but they are logged (once per backend+method) rather than swallowed silently.

``build_sink`` reads the ``[tracking]`` config and assembles backends; ``NullSink`` is the
"disconnected" state returned when tracking is disabled (the loop runs untouched). The producer
is responsible for only building a live sink on rank 0 (the core stays free of any distributed
dependency, and uses stdlib logging so it never imports rengu_flow).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rengu_track.backends.base import Backend
from rengu_track.events import append_event

logger = logging.getLogger("rengu_track")

_DEFAULT_BACKENDS = ["manifest", "tensorboard"]


class MetricsSink:
    """Fan-out metric sink over a list of backends + the event timeline."""

    def __init__(self, backends: list[Backend], run_dir: str | Path) -> None:
        self._backends = list(backends)
        self.run_dir = Path(run_dir)
        self._warned: set[tuple[int, str]] = set()

    def _safe(self, backend: Backend, method: str, *args: Any, **kwargs: Any) -> None:
        try:
            getattr(backend, method)(*args, **kwargs)
        except Exception as exc:  # tracking must never crash the run
            key = (id(backend), method)
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(
                    "tracking: %s.%s failed (further failures silenced): %s",
                    type(backend).__name__,
                    method,
                    exc,
                )

    def _fan(self, method: str, *args: Any, **kwargs: Any) -> None:
        for backend in self._backends:
            self._safe(backend, method, *args, **kwargs)

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._fan("scalar", tag, float(value), step)

    def histogram(self, tag: str, values: Any, step: int) -> None:
        self._fan("histogram", tag, values, step)

    def image(self, tag: str, image: Any, step: int) -> None:
        self._fan("image", tag, image, step)

    def summary(self, metrics: dict[str, Any]) -> None:
        self._fan("summary", metrics)

    def set_hparams(self, config: dict[str, Any]) -> None:
        self._fan("set_metadata", config=config)

    def set_lineage(self, lineage: dict[str, Any]) -> None:
        self._fan("set_metadata", lineage=lineage)

    def set_hardware(self, hardware: dict[str, Any]) -> None:
        self._fan("set_metadata", hardware=hardware)

    def event(
        self,
        event_type: str,
        *,
        step: int | None = None,
        payload: dict[str, Any] | None = None,
        source: str = "trainer",
    ) -> None:
        try:
            append_event(self.run_dir, event_type, step=step, payload=payload, source=source)
        except OSError as exc:
            logger.warning("tracking: failed to append event %s: %s", event_type, exc)

    def close(self, *, status: str | None = None) -> None:
        self._fan("close", status=status)


class NullSink:
    """No-op sink (tracking disabled). Every call is a no-op, including events."""

    run_dir: Path | None = None

    def scalar(self, *args: Any, **kwargs: Any) -> None: ...

    def histogram(self, *args: Any, **kwargs: Any) -> None: ...

    def image(self, *args: Any, **kwargs: Any) -> None: ...

    def summary(self, *args: Any, **kwargs: Any) -> None: ...

    def set_hparams(self, *args: Any, **kwargs: Any) -> None: ...

    def set_lineage(self, *args: Any, **kwargs: Any) -> None: ...

    def set_hardware(self, *args: Any, **kwargs: Any) -> None: ...

    def event(self, *args: Any, **kwargs: Any) -> None: ...

    def close(self, *args: Any, **kwargs: Any) -> None: ...


def build_sink(config: dict[str, Any], run_dir: str | Path) -> MetricsSink | NullSink:
    """Assemble a sink from the ``[tracking]`` config. Returns ``NullSink`` when disabled.

    Call this on rank 0 only (use ``NullSink()`` on other ranks) — the core deliberately does
    not import the distributed facade.
    """
    tracking = config.get("tracking", {}) or {}
    if not tracking.get("enabled", True):
        return NullSink()

    selected = tracking.get("backends", _DEFAULT_BACKENDS)
    run_dir = Path(run_dir)
    run_id = run_dir.name
    name = config.get("run_name") or run_id
    backends: list[Backend] = []

    if "manifest" in selected:
        from rengu_track.backends.manifest import ManifestBackend

        backends.append(ManifestBackend(run_dir, run_id=run_id, name=name, config=config))

    if "tensorboard" in selected:
        from rengu_track.backends.tensorboard import TensorBoardBackend

        backends.append(TensorBoardBackend(run_dir))

    if "wandb" in selected:
        wb = tracking.get("wandb", {}) or {}
        try:
            from rengu_track.backends.wandb import WandbBackend

            backends.append(
                WandbBackend(
                    project=wb.get("project", "rengu-flow"),
                    name=wb.get("run_name") or name,
                    config=config,
                    dir=run_dir,
                    api_key=wb.get("api_key"),
                )
            )
        except ImportError:
            logger.warning("tracking: wandb backend requested but wandb is not installed; skipping")

    return MetricsSink(backends, run_dir)
