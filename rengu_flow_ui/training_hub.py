"""Unified Train hub: jobs + filesystem runs, progress, previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from rengu_flow_ui import db, metrics_tb, runs_scanner
from rengu_flow_ui.job_queue import list_jobs_sorted
from rengu_flow_ui.paths import resolve_repo_path
from rengu_flow_ui.run_config import RunConfigError, read_run_config_dict

ACTIVE_STATES = frozenset({"running", "stopping"})
QUEUED_STATES = frozenset({"pending"})
NEW_STATES = frozenset({"new"})
TERMINAL_STATES = frozenset({"finished", "failed", "stopped"})


def _job_started_epoch(job: db.JobRecord) -> float | None:
    try:
        from datetime import datetime

        return datetime.fromisoformat(job.started_at).timestamp()
    except Exception:
        return None


def _fallback_run_dir_for_job(job: db.JobRecord) -> Path | None:
    """Best guess at an active job's folder before the trainer's ``Run dir:`` line is parsed.

    Never borrow an unrelated or older run's folder (the bug behind a fresh "new run
    from this config" showing the SOURCE run's stats): only consider folders created
    at/after this job started, preferring one whose name matches the job's run_name.
    Returns None when nothing clearly belongs to this job yet — the next poll fills in
    the authoritative run_dir from the log.
    """
    runs = runs_scanner.scan_output_runs(job.output_dir)
    if not runs:
        return None
    started = _job_started_epoch(job)
    run_name = _config_run_name(job.config_content)
    match_name: str | None = None
    if run_name:
        from rengu_flow.run_naming import sanitize_run_name

        match_name = sanitize_run_name(run_name) or None

    def name_matches(folder: str) -> bool:
        if not match_name:
            return False
        # Folders are "{timestamp}_{name}" (date first).
        return folder == match_name or folder.endswith(f"_{match_name}")

    best: tuple[bool, float, Path] | None = None
    for r in runs:
        path = Path(r["path"])
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        folder = r.get("name") or path.name
        named = name_matches(folder)
        # A folder that predates this job cannot be its output.
        if started is not None and mtime < started:
            continue
        # No start time and no name match: don't guess (would borrow an unrelated run).
        if started is None and not named:
            continue
        cand = (named, mtime, path)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    return best[2].resolve() if best else None


def resolve_job_run_dir(job: db.JobRecord) -> Path | None:
    """Best-effort run directory for a job."""
    if job.run_dir:
        p = Path(job.run_dir)
        if p.is_dir():
            return p.resolve()
    # Fallback only for active jobs without a recorded run_dir yet. A terminal job must
    # not borrow another run's folder, and an active job must not borrow the previous
    # (source) run's folder while its own is still being created.
    if job.output_dir and job.state in ACTIVE_STATES:
        return _fallback_run_dir_for_job(job)
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


def compute_run_progress(
    run_dir: Path | None,
    *,
    marker: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the UI ``progress`` payload for a run.

    For live runs, ``marker`` is the latest ``@@RFPROG@@`` payload parsed from the log
    (see ``live_stream``); it provides step/loss/epoch/speed/ETA. status.json is no
    longer written, so for finished/imported runs (no marker) we fall back to the last
    TensorBoard ``train/loss`` scalar so History rows still show final step/loss.
    """
    m = marker or {}
    has_dir = run_dir is not None and run_dir.is_dir()
    # The live marker (e.g. the caching phase) arrives from the log before the run
    # folder exists, so surface it even without a run_dir. A run folder is only needed
    # to read config limits and the TensorBoard-scalar fallback for finished runs.
    if not has_dir and not m:
        return None

    limits = _read_run_limits(run_dir) if has_dir else {}
    step = m.get("step")
    loss = m.get("loss")
    epoch = m.get("epoch")

    if (step is None or loss is None) and has_dir:
        scalars = metrics_tb.read_scalars(run_dir)
        series = scalars.get("train/loss") or []
        if series:
            last = series[-1]
            step = step if step is not None else last.get("step")
            loss = loss if loss is not None else last.get("value")

    # Generalization probe (held-out val loss + train-val gap). Prefer the live marker; fall
    # back to the last TensorBoard scalars for finished/imported runs.
    val_loss = m.get("val_loss")
    val_gap = m.get("val_gap")
    if (val_loss is None or val_gap is None) and has_dir:
        scalars = metrics_tb.read_scalars(run_dir, tag_prefix="")
        if val_loss is None:
            vseries = scalars.get("val/loss") or []
            if vseries:
                val_loss = vseries[-1].get("value")
        if val_gap is None:
            gseries = scalars.get("val/gap") or []
            if gseries:
                val_gap = gseries[-1].get("value")

    # Prefer the marker's own max_steps/percent (it knows epoch-derived totals too),
    # falling back to the config-derived budget.
    max_steps = m.get("max_steps") or limits.get("max_steps")
    percent = m.get("percent")
    if percent is None and step is not None and max_steps and max_steps > 0:
        percent = round(min(100.0, 100.0 * float(step) / float(max_steps)), 1)

    return {
        "step": step,
        "max_steps": max_steps,
        "epoch": epoch,
        "epochs": limits.get("epochs"),
        "loss": loss,
        "loss_avg": m.get("loss_avg"),
        "val_loss": val_loss,
        "val_gap": val_gap,
        "percent": percent,
        "phase": m.get("phase"),
        "updated_at": None,
        "status_available": bool(marker),
        "model_type": limits.get("model_type"),
        "run_name_label": limits.get("run_name_label"),
        "step_time_sec": m.get("step_time_sec"),
        "step_time_sec_ema": m.get("step_time_sec_ema"),
        "steps_per_second": m.get("steps_per_second"),
        "steps_per_second_ema": m.get("steps_per_second_ema"),
        "eta_sec": m.get("eta_sec"),
        "eta": m.get("eta"),
        "steps_remaining": m.get("steps_remaining"),
        # Caching-phase fields (present only while caching).
        "current": m.get("current"),
        "total": m.get("total"),
    }


def _config_run_name(config_content: str | None) -> str | None:
    """The run_name from a job's config snapshot — the display name before a folder exists."""
    if not config_content:
        return None
    try:
        name = toml.loads(config_content).get("run_name")
    except Exception:
        return None
    return name.strip() if isinstance(name, str) and name.strip() else None


def _job_to_training_run(job: db.JobRecord) -> dict[str, Any]:
    run_dir = resolve_job_run_dir(job)
    progress = compute_run_progress(run_dir)
    # Prefer the run folder name; fall back to the config's run_name so queued/draft runs
    # (which have no folder yet) still show a meaningful name instead of "—".
    run_name = _run_folder_name(run_dir) or _config_run_name(job.config_content)
    label = (progress or {}).get("run_name_label") or run_name
    return {
        "key": f"job:{job.id}",
        "kind": "job",
        "job_id": job.id,
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
        "cache_only": job.cache_only,
        "trust_cache": job.trust_cache,
        "regenerate_cache": job.regenerate_cache,
        "progress": progress,
        "has_tensorboard": bool(run_dir and list(run_dir.glob("events.out.tfevents.*"))),
    }


def _sort_runs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple:
        state = row.get("state") or ""
        if state in ACTIVE_STATES:
            return (0, 0, row.get("started_at") or "")
        if state in QUEUED_STATES:
            return (1, row.get("queue_position") if row.get("queue_position") is not None else 999, "")
        if state in NEW_STATES:
            return (2, 0, row.get("started_at") or "")
        return (3, 0, row.get("finished_at") or row.get("started_at") or "")

    return sorted(items, key=sort_key)


def _matches_query(row: dict[str, Any], term: str) -> bool:
    if not term:
        return True
    hay = " ".join(
        str(x or "")
        for x in (
            row.get("key"),
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
    state_filter: str | None = None,
) -> dict[str, Any]:
    """Train hub list: UI database jobs only (no filesystem scanning)."""
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    term = (q or "").strip().lower()

    items: list[dict[str, Any]] = [_job_to_training_run(job) for job in list_jobs_sorted()]

    if state_filter:
        sf = state_filter.strip().lower()
        if sf == "active":
            items = [r for r in items if r["state"] in ACTIVE_STATES]
        elif sf == "queued":
            items = [r for r in items if r["state"] in QUEUED_STATES]
        elif sf == "new":
            items = [r for r in items if r["state"] in NEW_STATES]
        elif sf == "finished":
            items = [r for r in items if r["state"] in TERMINAL_STATES]

    if term:
        items = [r for r in items if _matches_query(r, term)]

    items = _sort_runs(items)
    total = len(items)
    offset = (page - 1) * page_size
    page_items = items[offset : offset + page_size]

    running = sum(1 for r in items if r["state"] in ACTIVE_STATES)
    pending = sum(1 for r in items if r["state"] in QUEUED_STATES)
    saved = sum(1 for r in items if r["state"] in NEW_STATES)

    return {
        "items": page_items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "stats": {"running": running, "pending": pending, "saved": saved},
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


def list_run_preview_images(run_dir: str | Path, *, limit: int = 2000) -> list[dict[str, str]]:
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
    # Allow any filename (prompts may contain spaces, parentheses, etc.) but
    # reject anything that could escape the preview directory.
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("Invalid preview file name")
    preview_dir = (root / "preview").resolve()
    path = (preview_dir / name).resolve()
    if path.parent != preview_dir:
        raise ValueError("Invalid preview path")
    if not path.is_file():
        raise FileNotFoundError(name)
    return path
