"""Send signal files to a running training run directory."""

from __future__ import annotations

from pathlib import Path

from renga_flow.utils import signal_files as sf

SIGNAL_MAP = {
    "save": sf.SIGNAL_SAVE,
    "save_quit": sf.SIGNAL_SAVE_QUIT,
    "export_model": sf.SIGNAL_EXPORT_MODEL,
    "export_model_quit": sf.SIGNAL_EXPORT_MODEL_QUIT,
    "preview": sf.SIGNAL_PREVIEW,
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
