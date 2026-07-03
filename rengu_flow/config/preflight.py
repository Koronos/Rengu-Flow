"""Fast pre-training checks: catch what WILL fail before any model loads.

The structural validator (``validation.py``) is pure — it never touches the
filesystem, so configs validate in tests and on machines that don't hold the
weights. This module is the second half of the first barrier: cheap host-side
probes (path existence, writability, registry membership, pattern sanity) plus
the incompatible combinations that used to surface as mid-run raises or silent
runtime fallbacks. Both the CLI trainer and the web UI's validate endpoint run
it, so a bad path or an impossible combo costs seconds, not a caching pass.

Model knowledge comes from the capability registry (``model_fields`` declares
which keys are paths; ``adapter_module_roots`` declares the DiT's top-level
module names) — the same single sources the UI schema is built from, so no
check is duplicated per model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rengu_flow.registry.model_capabilities import get_capability

# Presence-keyed loss switches (loss_utils.compute_diffusion_loss_per_element):
# setting more than one silently trains with the highest-priority — make the
# conflict explicit instead.
_LOSS_KEYS = ("huber_delta", "smooth_l1_beta", "pseudo_huber_c")


def _path_value(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def _nearest_existing_ancestor(path: Path) -> Path:
    cur = path
    while not cur.exists():
        parent = cur.parent
        if parent == cur:
            return cur
        cur = parent
    return cur


def _check_writable_dir(issues: list[str], key: str, raw: Any) -> None:
    path = _path_value(raw)
    if path is None:
        return
    anchor = _nearest_existing_ancestor(path)
    if not os.access(anchor, os.W_OK):
        issues.append(f"{key} is not writable: {path} (nearest existing dir {anchor}).")


def _first_segment(pattern: str) -> str:
    return pattern.split(".", 1)[0]


def _has_glob(segment: str) -> bool:
    return any(ch in segment for ch in "*?[")


def collect_preflight_issues(config: dict[str, Any]) -> list[str]:
    """Host-side checks on a config that already passed structural validation.

    Returns issues (empty = ready). Never raises on malformed sections — the
    structural validator owns those messages.

    ``RENGU_PREFLIGHT=0`` disables all host probes — for validating a config on a
    machine that doesn't hold the weights/dataset (CI, another box). Default: on.
    """
    if os.environ.get("RENGU_PREFLIGHT", "1") == "0":
        return []
    issues: list[str] = []
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    cap = get_capability(model_cfg.get("type"))

    # --- files the run will read -------------------------------------------------
    for spec in getattr(cap, "model_fields", []) or []:
        if spec.get("type") != "path":
            continue
        key = spec["path"].split(".", 1)[1]
        path = _path_value(model_cfg.get(key))
        if path is not None and not path.exists():
            issues.append(f"model.{key} does not exist: {path}")

    adapter = config.get("adapter") if isinstance(config.get("adapter"), dict) else {}
    init_from = _path_value(adapter.get("init_from_existing"))
    if init_from is not None and not init_from.exists():
        issues.append(f"adapter.init_from_existing does not exist: {init_from}")

    resume = config.get("resume_from_checkpoint")
    resume_path = _path_value(resume) if not isinstance(resume, bool) else None
    if resume_path is not None and not resume_path.exists():
        issues.append(f"resume_from_checkpoint does not exist: {resume_path}")

    dataset = config.get("dataset")
    dataset_paths = dataset if isinstance(dataset, list) else [dataset]
    for entry in dataset_paths:
        ds_path = _path_value(entry)
        if ds_path is None or str(entry).startswith("rengu-flow-dataset:"):
            continue  # library refs / non-str handled by the structural validator
        if not ds_path.exists():
            issues.append(f"dataset TOML does not exist: {ds_path}")
            continue
        issues.extend(_dataset_directory_issues(ds_path))

    # --- places the run will write -------------------------------------------------
    _check_writable_dir(issues, "output_dir", config.get("output_dir"))
    _check_writable_dir(issues, "cache_root", config.get("cache_root"))

    # --- adapter layer selection: catch typos without loading the model -------------
    roots = list(getattr(cap, "adapter_module_roots", []) or [])
    if roots:
        for key in ("target_include", "target_exclude"):
            for pattern in adapter.get(key) or []:
                first = _first_segment(str(pattern))
                if not _has_glob(first) and first not in roots:
                    issues.append(
                        f"adapter.{key} pattern {pattern!r} matches nothing: no module "
                        f"root named {first!r}. Roots: {', '.join(sorted(roots))}."
                    )

    # --- incompatible combinations (no silent runtime fallback) ---------------------
    set_loss_keys = [k for k in _LOSS_KEYS if config.get(k) is not None]
    if len(set_loss_keys) > 1:
        issues.append(
            f"Only one loss switch may be set; found {', '.join(set_loss_keys)}. "
            "They are presence-keyed alternatives to MSE — remove all but one."
        )

    engine = str(config.get("engine") or "").strip().lower()
    if engine and engine not in ("deepspeed", "accelerate"):
        issues.append(f"engine {engine!r} is not one of: deepspeed, accelerate (or empty for the host default).")

    optimizer = config.get("optimizer") if isinstance(config.get("optimizer"), dict) else {}
    if optimizer.get("gradient_release"):
        from rengu_flow.engine import resolve_backend

        resolved = resolve_backend(config)
        if resolved != "deepspeed":
            issues.append(
                "optimizer.gradient_release requires engine='deepspeed'; this run resolves "
                f"to engine={resolved!r}. Remove gradient_release or set engine explicitly."
            )

    if config.get("async_model_export") and int(config.get("pipeline_stages") or 1) > 1:
        issues.append(
            "async_model_export does not work with pipeline_stages > 1 (the export snapshot "
            "spans pipeline ranks); it would silently fall back to synchronous export. "
            "Disable one of the two."
        )

    preview_cfg = config.get("preview") if isinstance(config.get("preview"), dict) else {}
    if preview_cfg.get("enabled") and not preview_cfg.get("prompts"):
        issues.append("preview.enabled is true but preview.prompts is empty — no previews would render.")

    scheduler = config.get("lr_scheduler")
    if isinstance(scheduler, str) and scheduler.strip() and "." not in scheduler:
        from rengu_flow.optim.resolver import scheduler_registry

        if scheduler.strip().lower() not in scheduler_registry:
            known = ", ".join(sorted(scheduler_registry))
            issues.append(f"lr_scheduler {scheduler!r} is not registered. Available: {known}.")

    return issues


def _dataset_directory_issues(ds_path: Path) -> list[str]:
    """Existence of the data folders a dataset TOML points at (cheap stat calls)."""
    import toml

    issues: list[str] = []
    try:
        data = toml.load(ds_path)
    except Exception:
        return []  # unparseable TOML is the dataset validator's message, not ours
    for i, directory in enumerate(data.get("directory") or []):
        if not isinstance(directory, dict):
            continue
        for key in ("path", "mask_path", "control_path"):
            folder = _path_value(directory.get(key))
            if folder is not None and not folder.is_dir():
                issues.append(
                    f"dataset [[directory]] #{i + 1} {key} is not a directory: {folder}"
                )
    return issues
