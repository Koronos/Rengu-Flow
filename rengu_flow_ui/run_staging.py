"""Training config validation, run-name helpers, and job staging.

The standalone config library was removed: a run carries its own TOML snapshot
(``JobRecord.config_content``). This module keeps the pieces still needed —
TOML validation, dataset-ref resolution into staging, and ``next_run_name``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import toml

from rengu_flow.platform_compat import PLATFORM
from rengu_flow.config import set_config_defaults
from rengu_flow.config.dataset_merge import merge_dataset_configs
from rengu_flow.config.validation import (
    collect_validation_errors,
    format_validation_issues,
    section_hints_for_empty_config,
)
from rengu_flow_ui import library_db
from rengu_flow_ui.config_form import _dtype_to_str
from rengu_flow_ui.optimizer_form import collect_optimizer_betas_validation_errors
from rengu_flow_ui.registry_probe import probe_resolution, resolution_errors
from rengu_flow_ui.settings import ensure_data_dirs, staging_dir


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
    issues.extend(collect_optimizer_betas_validation_errors(config))
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
    # Host-side preflight (path existence, writability, impossible combos): the same
    # first barrier the CLI trainer runs, so a bad path costs a click, not a caching pass.
    from rengu_flow.config.preflight import collect_preflight_issues

    probe_issues.extend(collect_preflight_issues(config))
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
        from rengu_flow_ui.dataset_form import loads_for_training

        with open(src, encoding="utf-8") as f:
            loaded.append(loads_for_training(f.read()))
    merged = merge_dataset_configs(loaded)
    merged_path = job_staging / "training_dataset_merged.toml"
    merged_path.write_text(toml.dumps(merged), encoding="utf-8")
    config["dataset"] = str(merged_path.resolve())


def _resolve_dataset_value(value: str, job_staging: Path) -> str:
    """Turn library dataset refs and relative paths into absolute TOML paths for training.

    Returns a ``PLATFORM.config_path`` (forward-slash on Windows): the resolved path is written
    back into the config and re-parsed as TOML, and a raw Windows ``C:\\…`` path is invalid TOML
    (``toml`` does not escape backslashes on dump). Forward slashes are valid TOML and work on
    Windows, so the staged config round-trips and stays cross-platform portable.
    """
    if library_db.is_library_dataset_ref(value):
        did = library_db.library_dataset_id_from_ref(value)
        from rengu_flow_ui.dataset_form import strip_display_name_from_toml

        content = strip_display_name_from_toml(library_db.read_dataset_text(did))
        out = job_staging / f"{did}.dataset.toml"
        out.write_text(content, encoding="utf-8")
        return PLATFORM.config_path(out.resolve())
    p = Path(value)
    if p.is_absolute():
        return PLATFORM.config_path(p.resolve())
    # Relative dataset paths resolve from the repo root — the training subprocess runs there
    # (popen_repo_subprocess uses cwd=repo_root), matching how the `rengu train` CLI loads them.
    # Resolving against job_staging produced data/staging/<id>/<path> which never exists.
    from rengu_flow_ui.settings import repo_root

    return PLATFORM.config_path((repo_root() / p).resolve())


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
    # Validate on a copy: set_config_defaults() is not idempotent (injects alpha, turns dtype
    # strings into torch.dtype objects) and the trainer re-applies it to the staged file.
    # Persist only the dataset-ref resolution below.
    import copy

    set_config_defaults(copy.deepcopy(config))
    if "dataset" in config:
        _materialize_dataset_for_job(config, job_staging)
    out = job_staging / "train.toml"
    out.write_text(toml.dumps(config), encoding="utf-8")
    return out


def _run_name_in_use(name: str) -> bool:
    """True if any existing job (snapshot) or output run folder already uses ``name``."""
    from rengu_flow_ui import db, runs_scanner
    from rengu_flow_ui.paths import resolve_repo_path

    target = name.strip()
    if not target:
        return False
    for job in db.list_jobs(limit=1000):
        content = job.config_content or ""
        if not content:
            continue
        try:
            cfg = toml.loads(content)
        except Exception:
            continue
        if isinstance(cfg.get("run_name"), str) and cfg["run_name"].strip() == target:
            return True
    try:
        root = resolve_repo_path("output")
        for desc in runs_scanner.scan_output_runs(root):
            if desc.get("name") == target:
                return True
    except Exception:
        pass
    return False


def next_run_name(base: str) -> str:
    """Return ``base`` with an incremental ``_N`` suffix that is not already in use.

    Strips a trailing ``_<digits>`` from ``base`` first so cloning a clone keeps a
    single counter (``run_2`` -> ``run_3`` rather than ``run_2_2``).
    """
    stem = re.sub(r"_\d+$", "", (base or "").strip()) or "run"
    n = 2
    while True:
        candidate = f"{stem}_{n}"
        if not _run_name_in_use(candidate):
            return candidate
        n += 1
