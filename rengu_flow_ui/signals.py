"""Send signal files to a running training run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rengu_flow.utils import signal_files as sf
from rengu_flow_ui.training_hub import ACTIVE_STATES

SIGNAL_MAP = {
    "save": sf.SIGNAL_SAVE,
    "save_quit": sf.SIGNAL_SAVE_QUIT,
    "export_model": sf.SIGNAL_EXPORT_MODEL,
    "export_model_quit": sf.SIGNAL_EXPORT_MODEL_QUIT,
    "preview": sf.SIGNAL_PREVIEW,
    "reload_config": sf.SIGNAL_RELOAD_CONFIG,
    "continue": sf.SIGNAL_CONTINUE,
    "quit": sf.SIGNAL_QUIT,
}

# UI metadata for GET /api/v1/signals (aligned with docs/user/signal-files.md)
SIGNAL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "save",
        "label": "Checkpoint",
        "group": "Resume checkpoint",
        "hint": "Write a DeepSpeed resume checkpoint on the next step (not inference weights).",
    },
    {
        "id": "save_quit",
        "label": "Checkpoint & quit",
        "group": "Resume checkpoint",
        "hint": "Resume checkpoint, then exit training.",
    },
    {
        "id": "export_model",
        "label": "Export model",
        "group": "Model export",
        "hint": "Export adapter or full weights to signal_step<N>/ on the next step.",
    },
    {
        "id": "export_model_quit",
        "label": "Export & quit",
        "group": "Model export",
        "hint": "Export inference weights, then exit.",
    },
    {
        "id": "preview",
        "label": "Preview",
        "group": "Preview",
        "hint": "Run configured preview sampling and log images to TensorBoard.",
    },
    {
        "id": "reload_config",
        "label": "Apply preview changes",
        "group": "Preview",
        "hint": "Reload the [preview] section from the run's config and apply it live "
        "(prompts, cadence, enabled, sampling). Other sections are not hot-reloaded.",
    },
    {
        "id": "continue",
        "label": "Continue export",
        "group": "Disk recovery",
        "hint": "While paused after disk-full export: retry export, then resume training.",
        "disk_wait_only": True,
        "variant": "primary",
    },
    {
        "id": "quit",
        "label": "Quit without save",
        "group": "Disk recovery",
        "hint": "While paused after disk-full export: exit without checkpoint or export.",
        "disk_wait_only": True,
        "variant": "danger",
    },
]

assert set(SIGNAL_MAP) == {d["id"] for d in SIGNAL_DEFINITIONS}

ACTIVE_JOB_STATES = sorted(ACTIVE_STATES)


def list_signal_definitions() -> dict[str, Any]:
    """Payload for GET /api/v1/signals."""
    return {
        "signals": SIGNAL_DEFINITIONS,
        "active_job_states": list(ACTIVE_JOB_STATES),
    }


def send_signal(run_dir: str | Path, signal_type: str) -> str:
    if signal_type not in SIGNAL_MAP:
        raise ValueError(f"Unknown signal: {signal_type}. Valid: {list(SIGNAL_MAP)}")
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")
    path = root / SIGNAL_MAP[signal_type]
    path.touch()
    return str(path)


def run_dir_accepts_signals(run_dir: str | Path) -> bool:
    """True when a UI job in running/stopping state owns this run directory."""
    from rengu_flow_ui import db

    job = db.find_job_by_run_dir(str(run_dir))
    if job is None:
        return False
    return job.state in ACTIVE_JOB_STATES
