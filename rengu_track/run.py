"""Per-run manifest (``run.json``): the structured metadata TensorBoard can't hold well.

Scalars / time-series stay in TB event files (and are read back via the event accumulator).
This file holds what TB is poor at — the full nested config, derived flat hparams, lineage,
hardware, and a rolling summary — so a viewer can build a cross-run comparison table by reading
one small JSON per run (no event-file parse, no DB). Written atomically (tempfile + ``os.replace``),
mirroring ``rengu_flow.control.status_file``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "run.json"


def now_iso() -> str:
    """UTC timestamp in ISO-8601 (shared by the manifest and the event log)."""
    return datetime.now(timezone.utc).isoformat()


def flatten_dict(data: Any, parent: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dicts/lists to dotted/indexed keys.

    Recurses into dicts and into lists that contain dicts/lists (indexed ``key[i]``), so a
    list-of-dicts like ``stage=[{...},{...}]`` becomes ``stage[0].x`` rows instead of one
    unreadable repr blob. A list of plain scalars is kept as a leaf so the caller can render
    it in a single cell (e.g. ``betas=[0.9, 0.999]``).
    """
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        pairs = [(f"{parent}{sep}{k}" if parent else str(k), v) for k, v in data.items()]
    else:  # list/tuple
        pairs = [(f"{parent}[{i}]", v) for i, v in enumerate(data)]
    for dotted, value in pairs:
        if isinstance(value, dict) or (
            isinstance(value, (list, tuple)) and any(isinstance(i, (dict, list, tuple)) for i in value)
        ):
            out.update(flatten_dict(value, dotted, sep))
        else:
            out[dotted] = value
    return out


def flatten_hparams(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten config to dotted scalar columns for the comparison table.

    Scalars (and None) pass through; list/tuple values become a comma-joined string so they
    still render as one column cell; anything else is stringified.
    """
    flat: dict[str, Any] = {}
    for key, value in flatten_dict(config).items():
        if value is None or isinstance(value, (bool, int, float, str)):
            flat[key] = value
        elif isinstance(value, (list, tuple)):
            flat[key] = ", ".join(str(item) for item in value)
        else:
            flat[key] = str(value)
    return flat


@dataclass
class RunManifest:
    """Structured per-run metadata persisted to ``run.json``."""

    run_id: str
    name: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = ""
    status: str = "running"
    config: dict[str, Any] = field(default_factory=dict)
    hparams_flat: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    system_summary: dict[str, Any] = field(default_factory=dict)
    # Cheap scalar index: the tag names logged and their last value. Lets the UI list/compare
    # views show metrics without parsing TensorBoard event files (recorded by the manifest backend).
    scalar_tags: list[str] = field(default_factory=list)
    last_scalars: dict[str, float] = field(default_factory=dict)
    last_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # Drop unknown keys gracefully (forward-compat: a newer writer added a field).
        return cls(**{k: v for k, v in data.items() if k in known})


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> None:
    """Atomically write ``run_dir/run.json`` (stamps ``updated_at``)."""
    root = Path(run_dir)
    manifest.updated_at = now_iso()
    target = root / MANIFEST_NAME
    fd, tmp = tempfile.mkstemp(dir=root, prefix=".run_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # The embedded config arrives post-defaults, where values like
            # model.dtype are live torch.dtype objects — stringify anything JSON
            # can't hold instead of failing the run at sink construction.
            json.dump(manifest.to_dict(), f, indent=2, default=str)
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_manifest(run_dir: str | Path) -> RunManifest | None:
    """Read ``run_dir/run.json`` if present and valid, else None."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError):
        return None
