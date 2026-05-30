"""Export/import the config & dataset library to/from TOML files (migration mode).

The SQLite library is mostly a TOML blob per row plus derived index columns. This module
makes those rows portable and schema-independent:

- **Export** writes ``configs/<id>.toml`` and ``datasets/<id>.toml``. Each file is the row's
  stored TOML content with a trailing ``[__rengu_index]`` table (id / kind / name /
  timestamps) appended. The trainer ignores unknown top-level tables, so an exported file is
  still a valid training/dataset TOML.
- **Import** reads those files back, strips the index table, and restores each row under its
  original id (re-deriving the index columns from content). Files without a valid index, or
  keys it does not recognize, are skipped — so the format tolerates older/newer exports.

See ``docs/developer/run-model-redesign.md`` §4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import toml

from rengu_flow_ui import library_db

INDEX_SECTION = "__rengu_index"


def _strip_index(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k != INDEX_SECTION}


def _record_file_text(content: str, index: dict[str, Any]) -> str:
    """Original content (verbatim) + an appended ``[__rengu_index]`` table."""
    body = (content or "").rstrip()
    index_toml = toml.dumps({INDEX_SECTION: index})
    return f"{body}\n\n{index_toml}" if body else index_toml


def export_library(dest_dir: str | Path) -> dict[str, int]:
    """Write every config and dataset row to ``<dest>/configs`` and ``<dest>/datasets``."""
    dest = Path(dest_dir)
    (dest / "configs").mkdir(parents=True, exist_ok=True)
    (dest / "datasets").mkdir(parents=True, exist_ok=True)
    counts = {"configs": 0, "datasets": 0}
    with library_db._connect() as conn:
        library_db.init_library_tables(conn)
        for row in conn.execute(
            "SELECT id, content, created_at, updated_at FROM training_configs"
        ).fetchall():
            (dest / "configs" / f"{row['id']}.toml").write_text(
                _record_file_text(
                    row["content"],
                    {
                        "id": row["id"],
                        "kind": "config",
                        "name": "",
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    },
                ),
                encoding="utf-8",
            )
            counts["configs"] += 1
        for row in conn.execute(
            "SELECT id, name, content, created_at, updated_at FROM datasets"
        ).fetchall():
            (dest / "datasets" / f"{row['id']}.toml").write_text(
                _record_file_text(
                    row["content"],
                    {
                        "id": row["id"],
                        "kind": "dataset",
                        "name": row["name"] or "",
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    },
                ),
                encoding="utf-8",
            )
            counts["datasets"] += 1
    return counts


def _read_record(path: Path) -> dict[str, Any] | None:
    """Parse one exported file; return its fields, or ``None`` if it has no valid index."""
    try:
        cfg = toml.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    index = cfg.get(INDEX_SECTION)
    if not isinstance(index, dict) or index.get("id") in (None, ""):
        return None  # not one of our records — ignore
    return {
        "id": index["id"],
        "kind": str(index.get("kind") or ""),
        "name": str(index.get("name") or ""),
        "created_at": str(index.get("created_at") or library_db._now()),
        "updated_at": str(index.get("updated_at") or library_db._now()),
        "content": toml.dumps(_strip_index(cfg)),
    }


def import_library(src_dir: str | Path, *, overwrite: bool = False) -> dict[str, int]:
    """Restore rows from an export directory.

    Existing ids are skipped unless ``overwrite=True``. Returns counts of imported,
    skipped (existing or unrecognized), per kind.
    """
    src = Path(src_dir)
    counts = {"configs": 0, "datasets": 0, "skipped": 0}
    verb = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
    with library_db._cursor() as cur:
        for kind, subdir, table in (
            ("config", "configs", "training_configs"),
            ("dataset", "datasets", "datasets"),
        ):
            folder = src / subdir
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.toml")):
                rec = _read_record(path)
                if rec is None or (rec["kind"] and rec["kind"] != kind):
                    counts["skipped"] += 1
                    continue
                if kind == "config":
                    model_type, dataset_ref, meta = library_db._extract_config_index(
                        rec["content"]
                    )
                    cur.execute(
                        f"""
                        {verb} INTO training_configs (
                            id, content, model_type, dataset_ref, meta_json,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec["id"],
                            rec["content"],
                            model_type,
                            dataset_ref,
                            json.dumps(meta),
                            rec["created_at"],
                            rec["updated_at"],
                        ),
                    )
                else:
                    directory_count, meta = library_db._extract_dataset_index(rec["content"])
                    cur.execute(
                        f"""
                        {verb} INTO datasets (
                            id, content, name, directory_count, meta_json,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec["id"],
                            rec["content"],
                            rec["name"],
                            directory_count,
                            json.dumps(meta),
                            rec["created_at"],
                            rec["updated_at"],
                        ),
                    )
                counts["configs" if kind == "config" else "datasets"] += (
                    1 if cur.rowcount else 0
                )
                if not cur.rowcount:
                    counts["skipped"] += 1
    return counts
