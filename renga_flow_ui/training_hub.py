"""Unified Train hub: jobs + filesystem runs, progress, previews."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import toml

from renga_flow.control.status_file import read_status_file
from renga_flow_ui import db, metrics_tb, runs_scanner
from renga_flow_ui.job_queue import list_jobs_sorted
from renga_flow_ui.paths import resolve_repo_path
from renga_flow_ui.run_config import RunConfigError, read_run_config_dict

ACTIVE_STATES = frozenset({"running", "stopping"})
QUEUED_STATES = frozenset({"pending"})
TERMINAL_STATES = frozenset({"finished", "failed", "stopped"})


def resolve_job_run_dir(job: db.JobRecord) -> Path | None:
    """Best-effort run directory for a job."""
    if job.run_dir:
        p = Path(job.run_dir)
        if p.is_dir():
            return p.resolve()
    if job.output_dir:
        runs = runs_scanner.scan_output_runs(job.output_dir)
        if runs:
            return Path(runs[-1]["path"]).resolve()
    return None


def _run_folder_name(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    return run_dir.name


def _read_run_limits(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None or not run_dir.is_dir():
        return {}
    try:
        cfg = read_run_config_dict(run_dir)
    except RunConfigError:
        try:
            cfg_path = runs_scanner.pick_main_config_path(run_dir)
            if cfg_path and cfg_path.is_file():
                cfg = toml.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                return {}
        except Exception:
            return {}
    if not isinstance(cfg, dict):
        return {}
    out: dict[str, Any] = {}
    if cfg.get("max_steps") is not None:
        try:
            out["max_steps"] = int(cfg["max_steps"])
        except (TypeError, ValueError):
            pass
    if cfg.get("epochs") is not None:
        try:
            out["epochs"] = int(cfg["epochs"])
        except (TypeError, ValueError):
            pass
    model = cfg.get("model")
    if isinstance(model, dict) and model.get("type"):
        out["model_type"] = str(model["type"])
    if isinstance(cfg.get("run_name"), str) and cfg["run_name"].strip():
        out["run_name_label"] = cfg["run_name"].strip()
    return out


def compute_run_progress(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None or not run_dir.is_dir():
        return None

    status = read_status_file(run_dir)
    limits = _read_run_limits(run_dir)
    step = status.get("step") if status else None
    loss = status.get("loss") if status else None
    epoch = status.get("epoch") if status else None

    if step is None or loss is None:
        scalars = metrics_tb.read_scalars(run_dir)
        series = scalars.get("train/loss") or []
        if series:
            last = series[-1]
            step = step if step is not None else last.get("step")
            loss = loss if loss is not None else last.get("value")

    max_steps = limits.get("max_steps")
    percent: float | None = None
    if step is not None and max_steps and max_steps > 0:
        percent = round(min(100.0, 100.0 * float(step) / float(max_steps)), 1)

    return {
        "step": step,
        "max_steps": max_steps,
        "epoch": epoch,
        "epochs": limits.get("epochs"),
        "loss": loss,
        "percent": percent,
        "phase": status.get("phase") if status else None,
        "updated_at": status.get("updated_at") if status else None,
        "status_available": status is not None,
        "model_type": limits.get("model_type"),
        "run_name_label": limits.get("run_name_label"),
    }


def _job_to_training_run(job: db.JobRecord) -> dict[str, Any]:
    run_dir = resolve_job_run_dir(job)
    progress = compute_run_progress(run_dir)
    run_name = _run_folder_name(run_dir)
    label = (progress or {}).get("run_name_label") or run_name
    return {
        "key": f"job:{job.id}",
        "kind": "job",
        "job_id": job.id,
        "config_id": job.config_id,
        "state": job.state,
        "run_dir": str(run_dir) if run_dir else job.run_dir,
        "run_name": run_name,
        "label": label,
        "output_dir": job.output_dir,
        "num_gpus": job.num_gpus,
        "resume_from": job.resume_from,
        "queue_position": job.queue_position,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "exit_code": job.exit_code,
        "progress": progress,
        "has_tensorboard": bool(run_dir and list(run_dir.glob("events.out.tfevents.*"))),
    }


def _disk_to_training_run(desc: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(desc["path"])
    progress = compute_run_progress(run_dir)
    run_name = desc.get("name") or run_dir.name
    label = (progress or {}).get("run_name_label") or run_name
    return {
        "key": f"disk:{run_name}",
        "kind": "disk",
        "job_id": None,
        "config_id": None,
        "state": "on_disk",
        "run_dir": str(run_dir),
        "run_name": run_name,
        "label": label,
        "output_dir": str(run_dir.parent),
        "num_gpus": None,
        "resume_from": None,
        "queue_position": None,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "progress": progress,
        "has_tensorboard": bool(desc.get("has_tensorboard")),
    }


def _sort_runs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple:
        state = row.get("state") or ""
        if state in ACTIVE_STATES:
            return (0, 0, row.get("started_at") or "")
        if state in QUEUED_STATES:
            return (1, row.get("queue_position") if row.get("queue_position") is not None else 999, "")
        if state == "on_disk":
            return (3, 0, row.get("run_name") or "")
        return (2, 0, row.get("finished_at") or row.get("started_at") or "")

    return sorted(items, key=sort_key)


def _matches_query(row: dict[str, Any], term: str) -> bool:
    if not term:
        return True
    hay = " ".join(
        str(x or "")
        for x in (
            row.get("key"),
            row.get("config_id"),
            row.get("run_name"),
            row.get("label"),
            row.get("state"),
            (row.get("progress") or {}).get("model_type"),
        )
    ).lower()
    return term in hay


def list_training_runs(
    *,
    q: str = "",
    page: int = 1,
    page_size: int = 20,
    include_disk: bool = True,
    output_dir: str = "output",
    state_filter: str | None = None,
) -> dict[str, Any]:
    """Unified Train hub list: UI jobs plus optional filesystem-only runs."""
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    term = (q or "").strip().lower()

    items: list[dict[str, Any]] = []
    known_run_dirs: set[str] = set()

    for job in list_jobs_sorted():
        row = _job_to_training_run(job)
        if row.get("run_dir"):
            known_run_dirs.add(str(Path(row["run_dir"]).resolve()))
        items.append(row)

    if include_disk:
        root = resolve_repo_path(output_dir)
        for desc in runs_scanner.scan_output_runs(root):
            resolved = str(Path(desc["path"]).resolve())
            if resolved in known_run_dirs:
                continue
            items.append(_disk_to_training_run(desc))

    if state_filter:
        sf = state_filter.strip().lower()
        if sf == "active":
            items = [r for r in items if r["state"] in ACTIVE_STATES]
        elif sf == "queued":
            items = [r for r in items if r["state"] in QUEUED_STATES]
        elif sf == "finished":
            items = [r for r in items if r["state"] in TERMINAL_STATES]
        elif sf in ("disk", "on_disk"):
            items = [r for r in items if r["state"] == "on_disk"]

    if term:
        items = [r for r in items if _matches_query(r, term)]

    items = _sort_runs(items)
    total = len(items)
    offset = (page - 1) * page_size
    page_items = items[offset : offset + page_size]

    running = sum(1 for r in items if r["state"] in ACTIVE_STATES)
    pending = sum(1 for r in items if r["state"] in QUEUED_STATES)

    return {
        "items": page_items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "stats": {"running": running, "pending": pending},
    }


def get_active_training_run() -> dict[str, Any] | None:
    for job in list_jobs_sorted():
        if job.state not in ACTIVE_STATES:
            continue
        run = _job_to_training_run(job)
        run_dir = run.get("run_dir")
        if run_dir:
            path = Path(run_dir)
            run["scalars"] = metrics_tb.read_scalars(path)
            run["preview_images"] = list_run_preview_images(path)
        else:
            run["scalars"] = {}
            run["preview_images"] = []
        return run
    return None


def list_run_preview_images(run_dir: str | Path, *, limit: int = 12) -> list[dict[str, str]]:
    root = Path(run_dir).resolve()
    preview = root / "preview"
    if not preview.is_dir():
        return []
    images = sorted(
        (
            p
            for p in preview.iterdir()
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[dict[str, str]] = []
    for path in images[:limit]:
        out.append({"name": path.name, "run_dir": str(root)})
    return out


def resolve_preview_image(run_dir: str, name: str) -> Path:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(run_dir)
    if not re.fullmatch(r"[\w.\-]+", name or ""):
        raise ValueError("Invalid preview file name")
    path = (root / "preview" / name).resolve()
    if not str(path).startswith(str(root)):
        raise ValueError("Invalid preview path")
    if not path.is_file():
        raise FileNotFoundError(name)
    return path
