"""SQLite library for training configs and dataset TOML (content + index columns)."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4

import toml

from renga_flow_ui.settings import db_path, ensure_data_dirs

DATASET_REF_PREFIX = "renga-flow-dataset:"


@dataclass
class LibraryRecord:
    id: str
    content: str
    created_at: str
    updated_at: str
    model_type: str | None = None
    dataset_ref: str | None = None
    directory_count: int | None = None
    meta_json: str = "{}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._")
    return base or uuid4().hex[:8]


def _connect() -> sqlite3.Connection:
    ensure_data_dirs()
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_library_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_configs (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            model_type TEXT,
            dataset_ref TEXT,
            meta_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            directory_count INTEGER NOT NULL DEFAULT 0,
            meta_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_training_configs_model ON training_configs(model_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_training_configs_updated ON training_configs(updated_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_datasets_updated ON datasets(updated_at)"
    )


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    conn = _connect()
    try:
        init_library_tables(conn)
        yield conn.cursor()
        conn.commit()
    finally:
        conn.close()


def dataset_library_ref(dataset_id: str) -> str:
    return f"{DATASET_REF_PREFIX}{_safe_id(dataset_id)}"


def is_library_dataset_ref(value: str) -> bool:
    return isinstance(value, str) and value.startswith(DATASET_REF_PREFIX)


def library_dataset_id_from_ref(value: str) -> str:
    return _safe_id(value[len(DATASET_REF_PREFIX) :])


def _extract_config_index(content: str) -> tuple[str | None, str | None, dict[str, Any]]:
    try:
        cfg = toml.loads(content)
    except Exception:
        return None, None, {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    model_type = model.get("type") if isinstance(model, dict) else None
    dataset_ref = cfg.get("dataset") if isinstance(cfg.get("dataset"), str) else None
    meta: dict[str, Any] = {}
    if isinstance(cfg.get("run_name"), str):
        meta["run_name"] = cfg["run_name"]
    if isinstance(cfg.get("output_dir"), str):
        meta["output_dir"] = cfg["output_dir"]
    return (
        str(model_type) if model_type is not None else None,
        dataset_ref,
        meta,
    )


def _extract_dataset_index(content: str) -> tuple[int, dict[str, Any]]:
    try:
        cfg = toml.loads(content)
    except Exception:
        return 0, {}
    directories = cfg.get("directory") or []
    if not isinstance(directories, list):
        directories = []
    meta: dict[str, Any] = {
        "resolutions": cfg.get("resolutions"),
        "frame_buckets": cfg.get("frame_buckets"),
    }
    paths = []
    for entry in directories:
        if isinstance(entry, dict) and entry.get("path"):
            paths.append(str(entry["path"]))
    if paths:
        meta["directory_paths"] = paths[:32]
    return len(directories), meta


# --- Training configs ---


def list_config_ids() -> list[str]:
    with _connect() as conn:
        init_library_tables(conn)
        rows = conn.execute(
            "SELECT id FROM training_configs ORDER BY updated_at DESC"
        ).fetchall()
    return [r["id"] for r in rows]


def _config_summary_row(r: sqlite3.Row) -> dict[str, Any]:
    meta = json.loads(r["meta_json"] or "{}")
    return {
        "id": r["id"],
        "model_type": r["model_type"],
        "dataset_ref": r["dataset_ref"],
        "updated_at": r["updated_at"],
        "run_name": meta.get("run_name"),
    }


def list_configs_summary() -> list[dict[str, Any]]:
    with _connect() as conn:
        init_library_tables(conn)
        rows = conn.execute(
            """
            SELECT id, model_type, dataset_ref, updated_at, meta_json
            FROM training_configs ORDER BY updated_at DESC
            """
        ).fetchall()
    return [_config_summary_row(r) for r in rows]


def search_configs(
    q: str = "",
    *,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Paginated config library search (id, model_type, dataset_ref)."""
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size
    term = (q or "").strip()
    like = f"%{term}%"
    where = ""
    params: list[Any] = []
    if term:
        where = (
            "WHERE id LIKE ? OR COALESCE(model_type, '') LIKE ? "
            "OR COALESCE(dataset_ref, '') LIKE ?"
        )
        params = [like, like, like]
    with _connect() as conn:
        init_library_tables(conn)
        total_row = conn.execute(
            f"SELECT COUNT(*) AS n FROM training_configs {where}",
            params,
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0
        rows = conn.execute(
            f"""
            SELECT id, model_type, dataset_ref, updated_at, meta_json
            FROM training_configs {where}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, page_size, offset),
        ).fetchall()
    return {
        "items": [_config_summary_row(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def config_exists(config_id: str) -> bool:
    cid = _safe_id(config_id)
    with _connect() as conn:
        init_library_tables(conn)
        row = conn.execute(
            "SELECT 1 FROM training_configs WHERE id = ?", (cid,)
        ).fetchone()
    return row is not None


def read_config_text(config_id: str) -> str:
    cid = _safe_id(config_id)
    with _connect() as conn:
        init_library_tables(conn)
        row = conn.execute(
            "SELECT content FROM training_configs WHERE id = ?", (cid,)
        ).fetchone()
    if row is None:
        raise FileNotFoundError(config_id)
    return row["content"]


def write_config_text(config_id: str, content: str) -> str:
    cid = _safe_id(config_id)
    now = _now()
    model_type, dataset_ref, meta = _extract_config_index(content)
    meta_json = json.dumps(meta)
    with _cursor() as cur:
        exists = cur.execute(
            "SELECT id FROM training_configs WHERE id = ?", (cid,)
        ).fetchone()
        if exists:
            cur.execute(
                """
                UPDATE training_configs
                SET content = ?, model_type = ?, dataset_ref = ?, meta_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (content, model_type, dataset_ref, meta_json, now, cid),
            )
        else:
            cur.execute(
                """
                INSERT INTO training_configs (
                    id, content, model_type, dataset_ref, meta_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cid, content, model_type, dataset_ref, meta_json, now, now),
            )
    return cid


def delete_config(config_id: str) -> None:
    cid = _safe_id(config_id)
    with _cursor() as cur:
        cur.execute("DELETE FROM training_configs WHERE id = ?", (cid,))
        if cur.rowcount == 0:
            raise FileNotFoundError(config_id)


def duplicate_config(config_id: str, new_id: str | None = None) -> str:
    text = read_config_text(config_id)
    target = _safe_id(new_id or f"{config_id}_copy")
    while config_exists(target):
        target = _safe_id(f"{target}_{uuid4().hex[:4]}")
    write_config_text(target, text)
    return target


def write_config_temp_file(config_id: str, *, staging_dir) -> "Path":
    from pathlib import Path

    path = Path(staging_dir) / f"_validate_{_safe_id(config_id)}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(read_config_text(config_id), encoding="utf-8")
    return path


# --- Datasets ---


def list_dataset_ids() -> list[str]:
    with _connect() as conn:
        init_library_tables(conn)
        rows = conn.execute("SELECT id FROM datasets ORDER BY updated_at DESC").fetchall()
    return [r["id"] for r in rows]


def _dataset_summary_row(r: sqlite3.Row) -> dict[str, Any]:
    meta = json.loads(r["meta_json"] or "{}")
    return {
        "id": r["id"],
        "directory_count": r["directory_count"],
        "updated_at": r["updated_at"],
        "directory_paths": meta.get("directory_paths") or [],
    }


def list_datasets_summary() -> list[dict[str, Any]]:
    with _connect() as conn:
        init_library_tables(conn)
        rows = conn.execute(
            """
            SELECT id, directory_count, updated_at, meta_json
            FROM datasets ORDER BY updated_at DESC
            """
        ).fetchall()
    return [_dataset_summary_row(r) for r in rows]


def search_datasets(
    q: str = "",
    *,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Paginated dataset library search (id, paths in meta_json)."""
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size
    term = (q or "").strip()
    like = f"%{term}%"
    where = ""
    params: list[Any] = []
    if term:
        where = "WHERE id LIKE ? OR COALESCE(meta_json, '') LIKE ?"
        params = [like, like]
    with _connect() as conn:
        init_library_tables(conn)
        total_row = conn.execute(
            f"SELECT COUNT(*) AS n FROM datasets {where}",
            params,
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0
        rows = conn.execute(
            f"""
            SELECT id, directory_count, updated_at, meta_json
            FROM datasets {where}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, page_size, offset),
        ).fetchall()
    return {
        "items": [_dataset_summary_row(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def dataset_exists(dataset_id: str) -> bool:
    did = _safe_id(dataset_id)
    with _connect() as conn:
        init_library_tables(conn)
        row = conn.execute("SELECT 1 FROM datasets WHERE id = ?", (did,)).fetchone()
    return row is not None


def read_dataset_text(dataset_id: str) -> str:
    did = _safe_id(dataset_id)
    with _connect() as conn:
        init_library_tables(conn)
        row = conn.execute("SELECT content FROM datasets WHERE id = ?", (did,)).fetchone()
    if row is None:
        raise FileNotFoundError(dataset_id)
    return row["content"]


def write_dataset_text(dataset_id: str, content: str) -> str:
    did = _safe_id(dataset_id)
    now = _now()
    directory_count, meta = _extract_dataset_index(content)
    meta_json = json.dumps(meta)
    with _cursor() as cur:
        exists = cur.execute("SELECT id FROM datasets WHERE id = ?", (did,)).fetchone()
        if exists:
            cur.execute(
                """
                UPDATE datasets
                SET content = ?, directory_count = ?, meta_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (content, directory_count, meta_json, now, did),
            )
        else:
            cur.execute(
                """
                INSERT INTO datasets (
                    id, content, directory_count, meta_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (did, content, directory_count, meta_json, now, now),
            )
    return did


def delete_dataset(dataset_id: str) -> None:
    did = _safe_id(dataset_id)
    with _cursor() as cur:
        cur.execute("DELETE FROM datasets WHERE id = ?", (did,))
        if cur.rowcount == 0:
            raise FileNotFoundError(dataset_id)


def duplicate_dataset(dataset_id: str, new_id: str | None = None) -> str:
    text = read_dataset_text(dataset_id)
    target = _safe_id(new_id or f"{dataset_id}_copy")
    while dataset_exists(target):
        target = _safe_id(f"{target}_{uuid4().hex[:4]}")
    write_dataset_text(target, text)
    return target
