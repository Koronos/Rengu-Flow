"""Optional status.json in run_dir for passive UI polling (rank 0 only)."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_status_file(run_dir: str | Path) -> dict[str, Any] | None:
    """Read run_dir/status.json if present and valid."""
    path = Path(run_dir) / "status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_status_file(
    run_dir: str | Path,
    *,
    step: int,
    examples: int,
    epoch: int,
    loss: float,
    phase: str = "training",
) -> None:
    """Atomically write run_dir/status.json (caller gates on monitoring.enable_status_file)."""
    root = Path(run_dir)
    payload: dict[str, Any] = {
        "step": step,
        "examples": examples,
        "epoch": epoch,
        "loss": loss,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    target = root / "status.json"
    fd, tmp = tempfile.mkstemp(dir=root, prefix=".status_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
