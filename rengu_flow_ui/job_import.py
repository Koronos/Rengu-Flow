"""Import script-mode training runs into the UI job registry."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import toml

from rengu_flow_ui import datasets_store, db, library_db
from rengu_flow_ui import runs_scanner
from rengu_flow_ui.paths import resolve_repo_path
from rengu_flow_ui.settings import logs_dir


class JobImportError(ValueError):
    """User-facing import failure."""


def _resolve_path_entry(entry: Any) -> Any:
    """Resolve a single relative, non-library path entry to an absolute string.

    Library ``rengu-flow-dataset:`` refs and non-string values pass through
    unchanged. Already-absolute paths come back unchanged (idempotent).
    """
    if not isinstance(entry, str) or not entry.strip():
        return entry
    if library_db.is_library_dataset_ref(entry):
        return entry
    return str(resolve_repo_path(entry))


def resolve_config_dataset_paths(config_toml_text: str) -> str:
    """Rewrite a training config's ``dataset`` field to absolute repo-root paths.

    The ``dataset`` value may be a single string or a list. Each relative,
    non-library entry is resolved against the repo root; library refs and
    absolute paths are left untouched. The original str-vs-list shape is
    preserved. ``output_dir`` and other fields are not touched.
    """
    cfg = toml.loads(config_toml_text)
    if not isinstance(cfg, dict):
        return config_toml_text
    dataset_val = cfg.get("dataset")
    if isinstance(dataset_val, str):
        cfg["dataset"] = _resolve_path_entry(dataset_val)
    elif isinstance(dataset_val, list):
        cfg["dataset"] = [_resolve_path_entry(x) for x in dataset_val]
    return toml.dumps(cfg)


def resolve_dataset_toml_paths(dataset_toml_text: str) -> str:
    """Rewrite a dataset TOML's directory ``path``/``cache_dir`` to absolute paths.

    Handles the common ``[[directory]]`` array-of-tables and a single
    ``[directory]`` table. Resolves a relative top-level ``cache_dir`` and each
    directory's relative ``cache_dir`` if present. Absolute paths and library
    refs are left unchanged (idempotent).
    """
    cfg = toml.loads(dataset_toml_text)
    if not isinstance(cfg, dict):
        return dataset_toml_text

    top_cache = cfg.get("cache_dir")
    if isinstance(top_cache, str) and top_cache.strip():
        cfg["cache_dir"] = _resolve_path_entry(top_cache)

    def _fix_dir(entry: Any) -> Any:
        if not isinstance(entry, dict):
            return entry
        row = dict(entry)
        for key in ("path", "cache_dir"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                row[key] = _resolve_path_entry(val)
        return row

    directories = cfg.get("directory")
    if isinstance(directories, list):
        cfg["directory"] = [_fix_dir(d) for d in directories]
    elif isinstance(directories, dict):
        cfg["directory"] = _fix_dir(directories)
    return toml.dumps(cfg)


def resolve_run_path(run_path: str) -> Path:
    raw = (run_path or "").strip()
    if not raw:
        raise JobImportError("Run folder path is required")
    p = resolve_repo_path(raw)
    if not p.is_dir():
        raise JobImportError(f"Not a directory: {p}")
    return p


def is_training_run_dir(run_dir: Path) -> bool:
    """Heuristic: folder looks like a rengu-flow output run."""
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

    # The imported run is self-contained via its config_content snapshot; there is no
    # separate config library. Optionally add the run's dataset TOML to the dataset library.
    config_content = resolve_config_dataset_paths(
        config_path.read_text(encoding="utf-8")
    )
    if import_dataset:
        ds_path = _dataset_file_in_run(run_dir, config_path)
        if ds_path is not None:
            datasets_store.insert_dataset(
                resolve_dataset_toml_paths(ds_path.read_text(encoding="utf-8")),
                name=f"{run_dir.name} dataset",
            )

    started_at, finished_at = _infer_timestamps(run_dir, desc)
    log_path = _write_import_log(run_dir, config_path)

    return db.create_imported_job(
        run_dir=run_dir_s,
        config_path=str(config_path.resolve()),
        output_dir=str(run_dir.parent),
        log_path=str(log_path),
        started_at=started_at,
        finished_at=finished_at,
        exit_code=0,
        source_run_dir=run_dir_s,
        config_content=config_content,
    )


def _dataset_file_in_run(
    run_dir: Path,
    config_path: Path | None,
    cfg: dict[str, Any] | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if cfg:
        from rengu_flow.config.loader import normalize_dataset_paths

        for ds_val in normalize_dataset_paths(cfg.get("dataset")):
            if not library_db.is_library_dataset_ref(ds_val):
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


def _write_import_log(run_dir: Path, config_path: Path) -> Path:
    logs_dir().mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = logs_dir() / f"imported-{run_dir.name}-{stamp}.log"
    lines = [
        f"--- imported script run: {run_dir.name} ---",
        f"Run dir: {run_dir}",
        f"Config: {config_path}",
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path
