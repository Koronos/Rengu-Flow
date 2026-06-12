"""Manifest backend: persists structured metadata to ``run.json``.

This is the "Local" backend (default ON). It ignores per-step scalars/images (those live in
TB) and instead owns config/hparams, lineage, hardware, and the rolling summary — rewriting
``run.json`` atomically whenever any of them changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rengu_track.backends.base import Backend
from rengu_track.run import RunManifest, flatten_hparams, write_manifest


class ManifestBackend(Backend):
    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        name: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        config = config or {}
        self._last_scalars: dict[str, float] = {}
        self._last_step: int | None = None
        self.manifest = RunManifest(
            run_id=run_id,
            name=name or run_id,
            status="running",
            config=config,
            hparams_flat=flatten_hparams(config),
        )
        write_manifest(self.run_dir, self.manifest)

    def _apply_scalar_index(self) -> None:
        """Fold the in-memory scalar index into the manifest before it is written."""
        if self._last_scalars:
            self.manifest.last_scalars = dict(self._last_scalars)
            self.manifest.scalar_tags = sorted(self._last_scalars)
            self.manifest.last_step = self._last_step

    def scalar(self, tag: str, value: float, step: int) -> None:
        # Record the last value per tag (and the latest step) in memory only — no per-step disk
        # write. The index is folded into run.json on the next metadata/summary/close write.
        self._last_scalars[tag] = float(value)
        if self._last_step is None or step > self._last_step:
            self._last_step = step

    def set_metadata(
        self,
        *,
        config: dict[str, Any] | None = None,
        lineage: dict[str, Any] | None = None,
        hardware: dict[str, Any] | None = None,
    ) -> None:
        if config is not None:
            self.manifest.config = config
            self.manifest.hparams_flat = flatten_hparams(config)
        if lineage is not None:
            self.manifest.lineage = lineage
        if hardware is not None:
            self.manifest.hardware = hardware
        self._apply_scalar_index()
        write_manifest(self.run_dir, self.manifest)

    def summary(self, metrics: dict[str, Any]) -> None:
        # ``system/*`` aggregates land under system_summary; everything else under summary.
        for key, value in metrics.items():
            if key.startswith("system/"):
                self.manifest.system_summary[key[len("system/") :]] = value
            else:
                self.manifest.summary[key] = value
        self._apply_scalar_index()
        write_manifest(self.run_dir, self.manifest)

    def close(self, *, status: str | None = None) -> None:
        if status is not None:
            self.manifest.status = status
        self._apply_scalar_index()
        write_manifest(self.run_dir, self.manifest)
