"""Scan output_dir for training run directories (filesystem history)."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any


def scan_output_runs(output_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    entries = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        entries.append(describe_run_dir(child))
    return entries


def pick_main_config_path(run_dir: Path) -> Path | None:
    """Training config TOML inside a run folder (excludes dataset-only files)."""
    config_files = sorted(run_dir.glob("*.toml"))
    for c in config_files:
        if c.name == "dataset.toml" or "dataset" in c.name.lower():
            continue
        return c
    if config_files:
        return config_files[0]
    return None


def describe_run_dir(run_dir: Path) -> dict[str, Any]:
    name = run_dir.name
    main = pick_main_config_path(run_dir)
    main_config = str(main) if main else None
    from rengu_flow.control.status_file import read_status_file

    status = read_status_file(run_dir)
    artifacts = _list_artifacts(run_dir)
    return {
        "id": name,
        "path": str(run_dir.resolve()),
        "name": name,
        "config_path": main_config,
        "status": status,
        "artifacts": artifacts,
        "has_tensorboard": any(run_dir.glob("events.out.tfevents.*")),
    }


def _list_artifacts(run_dir: Path) -> list[dict[str, str]]:
    out = []
    patterns = ("global_step*", "epoch*", "step*", "signal_step*")
    for pat in patterns:
        for p in sorted(glob.glob(str(run_dir / pat))):
            if os.path.isdir(p):
                out.append({"type": pat.rstrip("*"), "path": p})
    return out


def find_config_in_run(run_dir: str | Path) -> str | None:
    p = Path(run_dir)
    main = pick_main_config_path(p)
    return str(main) if main else None
