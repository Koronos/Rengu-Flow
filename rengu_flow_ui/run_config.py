"""Run-folder TOML as source of truth for resume / continue training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from rengu_flow_ui import runs_scanner
from rengu_flow_ui.job_import import resolve_run_path, unstage_config_dataset_refs
from rengu_flow_ui.paths import resolve_repo_path


class RunConfigError(ValueError):
    """Invalid or missing run configuration."""


def read_run_config_text(run_path: str | Path) -> str:
    """Read the training TOML snapshot from a run directory."""
    run_dir = resolve_run_path(str(run_path))
    config_path = runs_scanner.pick_main_config_path(run_dir)
    if config_path is None:
        raise RunConfigError(f"No training config .toml in run folder: {run_dir}")
    return config_path.read_text(encoding="utf-8")


def read_run_config_dict(run_path: str | Path) -> dict[str, Any]:
    try:
        return toml.loads(read_run_config_text(run_path))
    except Exception as e:
        raise RunConfigError(f"Could not parse run config TOML: {e}") from e


def resolve_output_dir(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config.get("output_dir") or "output")


def resume_checkpoint_arg(run_path: str | Path, config: dict[str, Any] | None = None) -> str:
    """CLI value for ``--resume_from_checkpoint`` (folder name or absolute path)."""
    run_dir = resolve_run_path(str(run_path))
    cfg = config if config is not None else read_run_config_dict(run_dir)
    out_path = resolve_output_dir(cfg)
    if run_dir.resolve().parent == out_path:
        return run_dir.name
    return str(run_dir.resolve())


def list_checkpoints(run_path: str | Path) -> list[dict[str, Any]]:
    """Checkpoints available to resume from, newest first.

    Each entry: ``{name, step, is_latest, suspect}``. ``is_latest`` marks the folder
    recorded in the run's ``latest`` pointer (the last known-good save). ``suspect`` is
    true for any checkpoint saved *after* the latest pointer — these can be truncated or
    corrupt (e.g. the disk filled up mid-save), so the UI flags them with a warning.
    """
    from rengu_flow.utils.save_io import global_step_sort_key

    run_dir = resolve_run_path(str(run_path))
    if not run_dir.is_dir():
        return []
    dirs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("global_step")),
        key=lambda p: global_step_sort_key(p.name),
    )  # ascending by step
    if not dirs:
        return []
    # DeepSpeed records the last known-good tag (e.g. "global_step5") in a `latest` file.
    latest_file = run_dir / "latest"
    latest_name: str | None = None
    if latest_file.is_file():
        latest_name = latest_file.read_text(encoding="utf-8").strip() or None
    if latest_name is None or not any(p.name == latest_name for p in dirs):
        # No pointer recorded (or it points nowhere): treat the highest-step checkpoint as latest.
        latest_name = dirs[-1].name
    latest_step = global_step_sort_key(latest_name)
    out: list[dict[str, Any]] = []
    for path in dirs:
        step = global_step_sort_key(path.name)
        out.append(
            {
                "name": path.name,
                "step": step,
                "is_latest": path.name == latest_name,
                "suspect": step > latest_step,
            }
        )
    out.sort(key=lambda c: c["step"], reverse=True)
    return out


def describe_run_config(run_path: str | Path) -> dict[str, Any]:
    run_dir = resolve_run_path(str(run_path))
    config_path = runs_scanner.pick_main_config_path(run_dir)
    cfg = read_run_config_dict(run_dir)
    return {
        "run_dir": str(run_dir),
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(resolve_output_dir(cfg)),
        "resume_from": resume_checkpoint_arg(run_dir, cfg),
        # Reverse any per-job staging dataset path so the editor shows the original
        # reference (library ref / the run's own dataset copy), not a staging copy.
        "content": unstage_config_dataset_refs(read_run_config_text(run_dir), run_dir=run_dir),
        "model_type": (cfg.get("model") or {}).get("type") if isinstance(cfg.get("model"), dict) else None,
        "epochs": cfg.get("epochs"),
        "max_steps": cfg.get("max_steps"),
    }
