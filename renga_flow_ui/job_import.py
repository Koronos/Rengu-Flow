"""Import script-mode training runs into the UI job registry."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import toml

from renga_flow_ui import configs_store, datasets_store, db, library_db
from renga_flow_ui import runs_scanner
from renga_flow_ui.settings import logs_dir, repo_root


class JobImportError(ValueError):
    """User-facing import failure."""


def resolve_run_path(run_path: str) -> Path:
    raw = (run_path or "").strip()
    if not raw:
        raise JobImportError("Run folder path is required")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (repo_root() / p).resolve()
    else:
        p = p.resolve()
    if not p.is_dir():
        raise JobImportError(f"Not a directory: {p}")
    return p


def is_training_run_dir(run_dir: Path) -> bool:
    """Heuristic: folder looks like a renga-flow output run."""
    if not run_dir.is_dir():
        return False
    if runs_scanner.pick_main_config_path(run_dir) is not None:
        return True
    if (run_dir / "status.json").is_file():
        return True
    if any(run_dir.glob("events.out.tfevents.*")):
        return True
    for pat in ("global_step*", "epoch*", "step*", "signal_step*"):
        if any(run_dir.glob(pat)):
            return True
    return False


def preview_import(run_path: str) -> dict[str, Any]:
    run_dir = resolve_run_path(run_path)
    if not is_training_run_dir(run_dir):
        raise JobImportError(
            "Folder does not look like a training run "
            "(expected a config .toml, checkpoints, status.json, or TensorBoard events)."
        )
    desc = runs_scanner.describe_run_dir(run_dir)
    existing = db.find_job_by_run_dir(str(run_dir))
    config_path = runs_scanner.pick_main_config_path(run_dir)
    dataset_path = _dataset_file_in_run(run_dir, config_path)
    suggested_config_id = library_db._safe_id(run_dir.name)
    suggested_dataset_id = library_db._safe_id(f"{run_dir.name}_dataset")
    return {
        "ok": True,
        "run": desc,
        "run_dir": str(run_dir),
        "output_dir": str(run_dir.parent),
        "config_path": str(config_path) if config_path else None,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "already_imported": existing is not None,
        "existing_job_id": existing.id if existing else None,
        "suggested_config_id": suggested_config_id,
        "suggested_dataset_id": suggested_dataset_id,
    }


def import_run(
    run_path: str,
    *,
    import_config: bool = True,
    config_id: str | None = None,
    import_dataset: bool = True,
    dataset_id: str | None = None,
    allow_duplicate: bool = False,
) -> db.JobRecord:
    run_dir = resolve_run_path(run_path)
    if not is_training_run_dir(run_dir):
        raise JobImportError("Folder does not look like a training run")

    run_dir_s = str(run_dir)
    existing = db.find_job_by_run_dir(run_dir_s)
    if existing and not allow_duplicate:
        raise JobImportError(
            f"This run is already in the job list (job {existing.id}). "
            "Delete that entry first or pass allow_duplicate."
        )

    desc = runs_scanner.describe_run_dir(run_dir)
    config_path = runs_scanner.pick_main_config_path(run_dir)
    if config_path is None:
        raise JobImportError("No training config .toml found in the run folder")

    lib_config_id: str | None = None
    if import_config:
        lib_config_id = _import_config_from_run(
            run_dir,
            config_path,
            config_id=config_id,
            import_dataset=import_dataset,
            dataset_id=dataset_id,
        )
    elif import_dataset:
        ds_path = _dataset_file_in_run(run_dir, config_path)
        if ds_path is not None:
            did = library_db._safe_id(dataset_id or f"{run_dir.name}_dataset")
            library_db.write_dataset_text(did, ds_path.read_text(encoding="utf-8"))

    started_at, finished_at = _infer_timestamps(run_dir, desc)
    log_path = _write_import_log(run_dir, config_path, lib_config_id)

    return db.create_imported_job(
        run_dir=run_dir_s,
        config_path=str(config_path.resolve()),
        config_id=lib_config_id or library_db._safe_id(run_dir.name),
        output_dir=str(run_dir.parent),
        log_path=str(log_path),
        started_at=started_at,
        finished_at=finished_at,
        exit_code=0,
        source_run_dir=run_dir_s,
    )


def _import_config_from_run(
    run_dir: Path,
    config_path: Path,
    *,
    config_id: str | None,
    import_dataset: bool,
    dataset_id: str | None,
) -> str:
    content = config_path.read_text(encoding="utf-8")
    try:
        cfg = toml.loads(content)
    except Exception as e:
        raise JobImportError(f"Could not parse config TOML: {e}") from e

    cid = library_db._safe_id(config_id or run_dir.name)

    if import_dataset:
        ds_path = _dataset_file_in_run(run_dir, config_path, cfg)
        if ds_path is not None:
            did = library_db._safe_id(dataset_id or f"{cid}_dataset")
            datasets_store.write_dataset_text(did, ds_path.read_text(encoding="utf-8"))
            cfg["dataset"] = datasets_store.dataset_library_ref(did)

    configs_store.write_config_text(cid, toml.dumps(cfg))
    return cid


def _dataset_file_in_run(
    run_dir: Path,
    config_path: Path | None,
    cfg: dict[str, Any] | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if cfg and isinstance(cfg.get("dataset"), str):
        ds_val = cfg["dataset"].strip()
        if ds_val and not library_db.is_library_dataset_ref(ds_val):
            p = Path(ds_val)
            if not p.is_absolute():
                p = run_dir / p
            candidates.append(p)
    for name in ("dataset.toml",):
        candidates.append(run_dir / name)
    if config_path:
        for f in run_dir.glob("*.toml"):
            if f.resolve() == config_path.resolve():
                continue
            if "dataset" in f.name.lower():
                candidates.append(f)
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p.resolve()
    return None


def _infer_timestamps(run_dir: Path, desc: dict[str, Any]) -> tuple[str, str | None]:
    finished_at: str | None = None
    status = desc.get("status")
    if isinstance(status, dict) and status.get("updated_at"):
        finished_at = str(status["updated_at"])

    try:
        mtime = run_dir.stat().st_mtime
        started_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        started_at = datetime.now(timezone.utc).isoformat()

    if finished_at is None:
        finished_at = started_at
    return started_at, finished_at


def _write_import_log(run_dir: Path, config_path: Path, config_id: str | None) -> Path:
    logs_dir().mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = logs_dir() / f"imported-{run_dir.name}-{stamp}.log"
    lines = [
        f"--- imported script run: {run_dir.name} ---",
        f"Run dir: {run_dir}",
        f"Config: {config_path}",
    ]
    if config_id:
        lines.append(f"Library config id: {config_id}")
    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path
