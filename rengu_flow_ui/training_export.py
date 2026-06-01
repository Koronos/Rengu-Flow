"""Export training configs for CLI use: ZIP bundle with resolved dataset TOML files."""

from __future__ import annotations

import copy
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import toml

from rengu_flow.config.dataset_library_ref import (
    canonical_dataset_ref,
    is_library_dataset_ref,
    iter_dataset_path_strings,
    library_dataset_id_from_ref,
)
from rengu_flow.config.loader import normalize_dataset_paths
from rengu_flow_ui import library_db
from rengu_flow_ui.dataset_form import loads_for_training, strip_display_name_from_toml
from rengu_flow_ui.paths import resolve_repo_path

_DIRECTORY_PATH_KEYS = ("path", "mask_path", "control_path")


def resolve_media_path(path_str: str, *, dataset_toml_path: Path | None) -> str:
    """Turn a relative media path into an absolute path for portable export."""
    raw = path_str.strip()
    if not raw:
        return path_str
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())
    repo_candidate = resolve_repo_path(raw)
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    if dataset_toml_path is not None:
        via_toml = (dataset_toml_path.parent / raw).resolve()
        if via_toml.exists():
            return str(via_toml)
    return str(repo_candidate.resolve())


def absolutize_dataset_config(
    config: dict[str, Any],
    *,
    dataset_toml_path: Path | None,
) -> dict[str, Any]:
    """Rewrite ``[[directory]]`` path fields to absolute paths."""
    out = copy.deepcopy(config)
    top_cache = out.get("cache_dir")
    if isinstance(top_cache, str) and top_cache.strip():
        out["cache_dir"] = resolve_media_path(top_cache, dataset_toml_path=dataset_toml_path)
    directories = out.get("directory")
    if not isinstance(directories, list):
        return out
    new_dirs: list[Any] = []
    for entry in directories:
        if not isinstance(entry, dict):
            new_dirs.append(entry)
            continue
        row = dict(entry)
        for key in (*_DIRECTORY_PATH_KEYS, "cache_dir"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                row[key] = resolve_media_path(val, dataset_toml_path=dataset_toml_path)
        new_dirs.append(row)
    out["directory"] = new_dirs
    return out


def export_dataset_toml_text(
    content: str,
    *,
    dataset_toml_path: Path | None = None,
) -> str:
    """Dataset TOML suitable for download (UI metadata stripped, paths absolute)."""
    cfg = loads_for_training(content)
    cfg = absolutize_dataset_config(cfg, dataset_toml_path=dataset_toml_path)
    return toml.dumps(cfg)


def _safe_bundle_stem(name: str) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", (name or "training_export").strip())
    return stem.strip("._") or "training_export"


def _unique_zip_name(used: set[str], stem: str) -> str:
    base = f"datasets/{stem}.toml"
    if base not in used:
        used.add(base)
        return base
    n = 2
    while True:
        candidate = f"datasets/{stem}_{n}.toml"
        if candidate not in used:
            used.add(candidate)
            return candidate
        n += 1


def build_training_export_zip(
    content: str,
    *,
    bundle_stem: str = "training_export",
) -> tuple[bytes, str]:
    """Build a ZIP with the train config and resolved dataset TOML file(s).

    Returns:
        ``(zip_bytes, download_filename)``
    """
    try:
        config = toml.loads(content)
    except Exception as e:
        raise ValueError(f"Could not parse training TOML: {e}") from e
    if not isinstance(config, dict):
        raise ValueError("Training config root must be a TOML table.")

    export_config = json.loads(json.dumps(config))
    used_zip_names: set[str] = set()
    zip_datasets: list[tuple[str, str]] = []
    ref_to_zip: dict[str, str] = {}

    def register_mapping(source: str, zip_rel: str) -> None:
        ref_to_zip[source] = zip_rel
        if is_library_dataset_ref(source):
            ref_to_zip[canonical_dataset_ref(source)] = zip_rel

    def add_dataset_toml(
        toml_content: str,
        *,
        filename_stem: str,
        source_path: Path | None,
        map_keys: list[str],
    ) -> str:
        cfg = loads_for_training(toml_content)
        cfg = absolutize_dataset_config(cfg, dataset_toml_path=source_path)
        text = toml.dumps(cfg)
        zip_rel = _unique_zip_name(used_zip_names, filename_stem)
        zip_datasets.append((zip_rel, text))
        for key in map_keys:
            register_mapping(key, zip_rel)
        return zip_rel

    seen_sources: set[str] = set()
    for _field, entry in iter_dataset_path_strings(config):
        if entry in seen_sources:
            continue
        seen_sources.add(entry)

        if is_library_dataset_ref(entry):
            try:
                did = library_dataset_id_from_ref(entry)
            except ValueError as e:
                raise ValueError(str(e)) from e
            raw = strip_display_name_from_toml(library_db.read_dataset_text(did))
            add_dataset_toml(
                raw,
                filename_stem=f"dataset_{did}",
                source_path=None,
                map_keys=[entry, canonical_dataset_ref(entry), f"rengu-flow-dataset:{did}"],
            )
            continue

        src = resolve_repo_path(entry)
        if not src.is_file():
            raise FileNotFoundError(f"Dataset file not found: {entry}")
        raw = src.read_text(encoding="utf-8")
        add_dataset_toml(
            raw,
            filename_stem=src.stem,
            source_path=src,
            map_keys=[entry, str(src.resolve())],
        )

    def map_dataset_ref(value: str) -> str:
        if value in ref_to_zip:
            return ref_to_zip[value]
        if is_library_dataset_ref(value):
            canon = canonical_dataset_ref(value)
            if canon in ref_to_zip:
                return ref_to_zip[canon]
        resolved = str(resolve_repo_path(value).resolve())
        if resolved in ref_to_zip:
            return ref_to_zip[resolved]
        if value in ref_to_zip:
            return ref_to_zip[value]
        raise FileNotFoundError(f"Dataset was not exported: {value!r}")

    dataset_val = export_config.get("dataset")
    if isinstance(dataset_val, str) and dataset_val.strip():
        export_config["dataset"] = map_dataset_ref(dataset_val.strip())
    elif isinstance(dataset_val, list):
        export_config["dataset"] = [
            map_dataset_ref(x.strip()) for x in dataset_val if isinstance(x, str) and x.strip()
        ]

    eval_datasets = export_config.get("eval_datasets")
    if isinstance(eval_datasets, list):
        new_eval: list[Any] = []
        for entry in eval_datasets:
            if isinstance(entry, str) and entry.strip():
                new_eval.append(map_dataset_ref(entry.strip()))
            elif isinstance(entry, dict) and isinstance(entry.get("config"), str):
                row = dict(entry)
                row["config"] = map_dataset_ref(row["config"].strip())
                new_eval.append(row)
            else:
                new_eval.append(entry)
        export_config["eval_datasets"] = new_eval

    stem = _safe_bundle_stem(bundle_stem)
    main_name = f"{stem}.toml"
    main_toml = toml.dumps(export_config)
    readme = (
        "Rengu Flow training export\n"
        "========================\n\n"
        f"1. Extract this ZIP to a folder.\n"
        f"2. Run: python -m rengu_flow.main --config {main_name}\n\n"
        "Dataset image paths inside datasets/*.toml are absolute paths on the machine "
        "where you exported. Move image folders or edit paths if you train on another machine.\n"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(main_name, main_toml)
        for zip_path, dataset_text in zip_datasets:
            zf.writestr(zip_path, dataset_text)
        zf.writestr("README.txt", readme)

    if not zip_datasets and normalize_dataset_paths(config.get("dataset")):
        raise ValueError("Training config lists datasets but none could be exported.")

    return buf.getvalue(), f"{stem}.zip"
