"""Dataset TOML library: SQLite storage and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from renga_flow.config.dataset_merge import merge_dataset_configs
from renga_flow.data.dataset_config import (
    DatasetConfigError,
    validate_dataset_config_for_real_data,
)
from renga_flow_ui import library_db
from renga_flow_ui.dataset_form import (
    embed_display_name,
    loads_for_training,
    parse_toml_to_form,
    strip_display_name_from_toml,
)
from renga_flow_ui.dataset_scan import preview_dataset_config
from renga_flow_ui.settings import ensure_data_dirs

list_dataset_ids = library_db.list_dataset_ids
read_dataset_text = library_db.read_dataset_text
insert_dataset = library_db.insert_dataset
update_dataset_text = library_db.update_dataset_text
delete_dataset = library_db.delete_dataset
duplicate_dataset = library_db.duplicate_dataset


def read_dataset_for_ui(dataset_id: str | int) -> dict[str, Any]:
    """Return stored name and TOML unchanged. UI builders parse separately."""
    did = library_db._coerce_record_id(dataset_id)
    try:
        library_db.refresh_dataset_index(did)
    except FileNotFoundError:
        raise
    except Exception:
        pass
    row = library_db.read_dataset_row(did)
    return {
        "id": did,
        "name": row["name"],
        "content": embed_display_name(row["content"], row["name"]),
    }


def parse_dataset_for_ui(content: str) -> dict[str, Any]:
    """Parse TOML into a form model for dataset UI; does not rewrite stored content."""
    form, notes = parse_toml_to_form(content)
    return {"form": form, "ui_notes": notes}


def dataset_exists(dataset_id: str | int) -> bool:
    return library_db.dataset_exists(dataset_id)


def list_datasets_summary(
    *,
    sort: str | None = None,
    order: str | None = None,
) -> list[dict[str, Any]]:
    return library_db.list_datasets_summary(sort=sort, order=order)


def search_datasets_page(
    q: str,
    *,
    page: int,
    page_size: int,
    sort: str | None = None,
    order: str | None = None,
) -> dict[str, Any]:
    return library_db.search_datasets(
        q, page=page, page_size=page_size, sort=sort, order=order
    )


def parse_dataset_dict(content: str) -> dict[str, Any]:
    return loads_for_training(content)


def prepare_dataset_content_for_storage(content: str, name: str | None) -> str:
    """Normalize TOML on save: strip display name from body, re-embed from library name."""
    return embed_display_name(strip_display_name_from_toml(content), name)


def validate_dataset_text(content: str) -> dict[str, Any]:
    try:
        config = loads_for_training(content)
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


def dataset_library_ref(dataset_id: str | int, display_name: str | None = None) -> str:
    return library_db.dataset_library_ref(dataset_id, display_name)


def create_dataset(content: str, name: str | None = None) -> int:
    return insert_dataset(content, name=name)


def compose_datasets(source_ids: list[str | int]) -> int:
    """Merge [[directory]] tables from library datasets into one record."""
    if not source_ids:
        raise ValueError("Select at least one source dataset")
    configs = [parse_dataset_dict(read_dataset_text(sid)) for sid in source_ids]
    merged = merge_dataset_configs(configs)
    content = toml.dumps(merged)
    return insert_dataset(content)


def import_example(src: Path) -> int:
    return insert_dataset(src.read_text(encoding="utf-8"))


def list_for_training_picker() -> list[dict[str, str]]:
    """Entries for training/eval dataset pickers (library only).

    Repo ``examples/`` files are not listed; users import them into the library first.
    """
    ensure_data_dirs()
    out: list[dict[str, str]] = []
    for summary in list_datasets_summary():
        did = summary["id"]
        display = summary.get("name") or str(did)
        ref = library_db.dataset_library_ref(did, display)
        out.append({"id": str(did), "path": ref, "label": f"{display} (#{did})"})
    return out
