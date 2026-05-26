"""Dataset TOML library: SQLite storage and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from renga_flow.data.dataset_config import (
    DatasetConfigError,
    validate_dataset_config_for_real_data,
)
from renga_flow_ui import library_db
from renga_flow_ui.dataset_scan import preview_dataset_config
from renga_flow_ui.settings import ensure_data_dirs, repo_root

_safe_id = library_db._safe_id
list_dataset_ids = library_db.list_dataset_ids
read_dataset_text = library_db.read_dataset_text
write_dataset_text = library_db.write_dataset_text
delete_dataset = library_db.delete_dataset
duplicate_dataset = library_db.duplicate_dataset


def dataset_exists(dataset_id: str) -> bool:
    return library_db.dataset_exists(dataset_id)


def list_datasets_summary() -> list[dict[str, Any]]:
    return library_db.list_datasets_summary()


def search_datasets_page(q: str, *, page: int, page_size: int) -> dict[str, Any]:
    return library_db.search_datasets(q, page=page, page_size=page_size)


def parse_dataset_dict(content: str) -> dict[str, Any]:
    return toml.loads(content)


def validate_dataset_text(content: str) -> dict[str, Any]:
    try:
        config = toml.loads(content)
    except Exception as e:
        return {"ok": False, "error": f"TOML parse error: {e}"}
    try:
        validate_dataset_config_for_real_data(config)
    except DatasetConfigError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    preview = preview_dataset_config(config)
    return {"ok": True, "config": config, "preview": preview}


def dataset_library_ref(dataset_id: str) -> str:
    return library_db.dataset_library_ref(dataset_id)


def compose_datasets(
    target_id: str,
    source_ids: list[str],
    *,
    merge_globals: str = "first",
) -> str:
    """Merge [[directory]] tables from library datasets into one record."""
    del merge_globals
    if not source_ids:
        raise ValueError("Select at least one source dataset")
    merged: dict[str, Any] = {}
    directories: list[dict[str, Any]] = []
    for sid in source_ids:
        cfg = parse_dataset_dict(read_dataset_text(sid))
        if not merged:
            for key, val in cfg.items():
                if key != "directory":
                    merged[key] = val
        directories.extend(cfg.get("directory") or [])
    merged["directory"] = directories
    if "resolutions" not in merged:
        merged["resolutions"] = [1024]
    if "frame_buckets" not in merged:
        merged["frame_buckets"] = [1]
    content = toml.dumps(merged)
    write_dataset_text(target_id, content)
    return target_id


def import_example(src: Path, dataset_id: str | None = None) -> str:
    cid = library_db._safe_id(dataset_id or src.stem)
    library_db.write_dataset_text(cid, src.read_text(encoding="utf-8"))
    return cid


def list_for_training_picker() -> list[dict[str, str]]:
    """Entries for main config ``dataset`` field (library ref + repo examples)."""
    ensure_data_dirs()
    out: list[dict[str, str]] = []
    for did in list_dataset_ids():
        ref = library_db.dataset_library_ref(did)
        out.append({"id": did, "path": ref, "label": did})
    examples = repo_root() / "examples"
    if examples.is_dir():
        for p in sorted(examples.glob("*dataset*.toml")):
            out.append(
                {
                    "id": p.stem,
                    "path": str(p.resolve()),
                    "label": f"{p.name} (example file)",
                }
            )
    return out
