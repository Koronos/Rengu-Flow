"""Training config library: SQLite storage, validation, job staging."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import toml

from renga_flow.config import set_config_defaults
from renga_flow.config.dataset_merge import merge_dataset_configs
from renga_flow.config.validation import (
    ConfigValidationError,
    collect_validation_errors,
    format_validation_issues,
    section_hints_for_empty_config,
)
from renga_flow_ui import library_db
from renga_flow_ui.config_form import _dtype_to_str
from renga_flow_ui.registry_probe import probe_resolution, resolution_errors
from renga_flow_ui.settings import ensure_data_dirs, staging_dir

# Re-export for callers
_safe_id = library_db._safe_id
list_config_ids = library_db.list_config_ids
read_config_text = library_db.read_config_text
insert_config = library_db.insert_config
update_config_text = library_db.update_config_text
delete_config = library_db.delete_config
duplicate_config = library_db.duplicate_config


def create_config(content: str) -> int:
    return insert_config(content)


def config_exists(config_id: str | int) -> bool:
    return library_db.config_exists(config_id)


def list_configs_summary(
    *,
    sort: str | None = None,
    order: str | None = None,
) -> list[dict[str, Any]]:
    return library_db.list_configs_summary(sort=sort, order=order)


def search_configs_page(
    q: str,
    *,
    page: int,
    page_size: int,
    sort: str | None = None,
    order: str | None = None,
) -> dict[str, Any]:
    return library_db.search_configs(
        q, page=page, page_size=page_size, sort=sort, order=order
    )


def _config_json_safe(config: dict[str, Any]) -> dict[str, Any]:
    return _dtype_to_str(config)


def _validation_failure(issues: list[str], **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "errors": issues, "error": format_validation_issues(issues)}
    out.update(extra)
    return out


def validate_toml_text(content: str) -> dict[str, Any]:
    """Parse, validate, apply defaults; return result dict."""
    text = (content or "").strip()
    if not text:
        issues = [
            "Config is empty — add content in the Form or TOML tab, or click Import example.",
            *section_hints_for_empty_config(),
        ]
        return _validation_failure(issues)

    try:
        config = toml.loads(content)
    except Exception as e:
        return _validation_failure([f"Could not parse TOML: {e}"])

    issues = collect_validation_errors(config)
    if issues:
        return _validation_failure(issues)

    try:
        set_config_defaults(config)
    except KeyError as e:
        return _validation_failure(
            [f"Missing key {e!s} while applying defaults — check required [model] fields."]
        )

    resolution = probe_resolution(config)
    probe_issues = resolution_errors(resolution)
    if probe_issues:
        return _validation_failure(
            probe_issues,
            config=_config_json_safe(config),
            resolution=resolution,
        )
    return {"ok": True, "config": _config_json_safe(config), "resolution": resolution}


def _copy_dataset_file_if_outside_staging(ds_file: Path, job_staging: Path) -> None:
    if ds_file.is_file() and ds_file.parent.resolve() != job_staging.resolve():
        shutil.copy(ds_file, job_staging / ds_file.name)


def _materialize_dataset_for_job(config: dict[str, Any], job_staging: Path) -> None:
    """Resolve library refs; merge multiple dataset paths into one staged TOML when needed."""
    ds = config.get("dataset")
    if isinstance(ds, str):
        ds_path = _resolve_dataset_value(ds, job_staging)
        config["dataset"] = ds_path
        _copy_dataset_file_if_outside_staging(Path(ds_path), job_staging)
        return
    if not isinstance(ds, list):
        return
    paths = [p.strip() for p in ds if isinstance(p, str) and p.strip()]
    if not paths:
        return
    resolved = [_resolve_dataset_value(p, job_staging) for p in paths]
    for src in resolved:
        _copy_dataset_file_if_outside_staging(Path(src), job_staging)
    if len(resolved) == 1:
        config["dataset"] = resolved[0]
        return
    loaded: list[dict[str, Any]] = []
    for src in resolved:
        from renga_flow_ui.dataset_form import loads_for_training

        with open(src, encoding="utf-8") as f:
            loaded.append(loads_for_training(f.read()))
    merged = merge_dataset_configs(loaded)
    merged_path = job_staging / "training_dataset_merged.toml"
    merged_path.write_text(toml.dumps(merged), encoding="utf-8")
    config["dataset"] = str(merged_path.resolve())


def _resolve_dataset_value(value: str, job_staging: Path) -> str:
    """Turn library dataset refs and relative paths into absolute TOML paths for training."""
    if library_db.is_library_dataset_ref(value):
        did = library_db.library_dataset_id_from_ref(value)
        from renga_flow_ui.dataset_form import strip_display_name_from_toml

        content = strip_display_name_from_toml(library_db.read_dataset_text(did))
        out = job_staging / f"{did}.dataset.toml"
        out.write_text(content, encoding="utf-8")
        return str(out.resolve())
    p = Path(value)
    if p.is_absolute():
        return str(p.resolve())
    return str((job_staging / p).resolve())


def materialize_staging(
    content: str,
    job_id: str | int,
    *,
    source_path: Path | None = None,
) -> Path:
    """Write train.toml for a job; resolve dataset library refs to files in staging."""
    del source_path
    ensure_data_dirs()
    job_staging = staging_dir() / str(job_id)
    job_staging.mkdir(parents=True, exist_ok=True)
    config = toml.loads(content)
    set_config_defaults(config)
    if "dataset" in config:
        _materialize_dataset_for_job(config, job_staging)
    out = job_staging / "train.toml"
    out.write_text(toml.dumps(config), encoding="utf-8")
    return out


def import_example(src: Path) -> int:
    return insert_config(src.read_text(encoding="utf-8"))


def write_config_temp_for_validate(config_id: str | int) -> Path:
    return library_db.write_config_temp_file(config_id, staging_dir=staging_dir())
