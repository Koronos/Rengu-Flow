"""SQLite library for training configs and dataset TOML (integer autoincrement ids)."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import toml

from rengu_flow.config.dataset_library_ref import (
    DATASET_REF_PREFIX,
    dataset_library_ref,
    is_library_dataset_ref,
    library_dataset_id_from_ref,
)

from rengu_flow_ui._time import now_utc_iso as _now
from rengu_flow_ui.settings import db_path, ensure_data_dirs

__all__ = [
    "DATASET_REF_PREFIX",
    "dataset_library_ref",
    "is_library_dataset_ref",
    "library_dataset_id_from_ref",
]


@dataclass
class LibraryRecord:
    id: int
    content: str
    created_at: str
    updated_at: str
    model_type: str | None = None
    dataset_ref: str | None = None
    directory_count: int | None = None
    meta_json: str = "{}"



def _safe_id(name: str) -> str:
    """Sanitize a label for staging filenames (not used as DB primary keys)."""
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._")
    return base or "unnamed"


def _coerce_record_id(record_id: str | int) -> int:
    if isinstance(record_id, bool):
        raise FileNotFoundError(record_id)
    if isinstance(record_id, int):
        return record_id
    s = str(record_id).strip()
    if not s.isdigit():
        raise FileNotFoundError(record_id)
    return int(s)


def _safe_meta_json(raw: str | None) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _safe_int_column(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


_DATASET_SORT_KEYS = frozenset({"id", "name", "created_at", "updated_at"})
DEFAULT_LIBRARY_SORT = "id"
DEFAULT_LIBRARY_ORDER = "desc"


def normalize_library_sort(
    sort: str | None = None,
    order: str | None = None,
) -> tuple[str, str]:
    """Return validated ``(sort_key, asc|desc)`` for dataset library list queries."""
    key = (sort or DEFAULT_LIBRARY_SORT).strip().lower()
    if key not in _DATASET_SORT_KEYS:
        key = DEFAULT_LIBRARY_SORT
    direction = (order or DEFAULT_LIBRARY_ORDER).strip().lower()
    if direction not in ("asc", "desc"):
        direction = DEFAULT_LIBRARY_ORDER
    return key, direction


def _library_order_clause(
    sort: str | None = None,
    order: str | None = None,
) -> str:
    key, direction = normalize_library_sort(sort, order)
    dir_sql = direction.upper()
    id_tie = f", id {dir_sql}"
    if key == "name":
        return f"ORDER BY name COLLATE NOCASE {dir_sql}{id_tie}"
    if key == "id":
        return f"ORDER BY id {dir_sql}"
    return f"ORDER BY {key} {dir_sql}{id_tie}"


def _connect() -> sqlite3.Connection:
    ensure_data_dirs()
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    # Same WAL + busy_timeout + synchronous tuning as db._connect (same DB file): keeps the
    # runs/datasets reads concurrent with job writes instead of blocking on the journal lock.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_library_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            directory_count INTEGER NOT NULL DEFAULT 0,
            meta_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_datasets_updated ON datasets(updated_at)"
    )
    # GPU arbitration (see rengu_flow_ui/gpu_lease.py). `PRIMARY KEY (device)` IS the mutex:
    # inserting over an occupied device raises IntegrityError, an atomic compare-and-swap that
    # holds across processes. Additive table, healed by CREATE TABLE IF NOT EXISTS exactly as
    # `datasets` was, so SCHEMA_VERSION does not move.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_leases (
            device INTEGER PRIMARY KEY,
            holder_kind TEXT NOT NULL,
            holder_id TEXT NOT NULL,
            pid INTEGER,
            pid_create_time REAL,
            acquired_at TEXT NOT NULL
        )
        """
    )
    _migrate_datasets_name(conn)


def _migrate_datasets_name(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(datasets)").fetchall()}
    if "name" not in cols:
        conn.execute("ALTER TABLE datasets ADD COLUMN name TEXT NOT NULL DEFAULT ''")
        rows = conn.execute("SELECT id FROM datasets").fetchall()
        for row in rows:
            did = int(row["id"])
            conn.execute(
                "UPDATE datasets SET name = ? WHERE id = ? AND (name IS NULL OR name = '')",
                (f"Dataset {did}", did),
            )


def _normalize_dataset_name(name: str | None, dataset_id: int | None = None) -> str:
    s = (name or "").strip()
    if s:
        return s[:200]
    if dataset_id is not None:
        return f"Dataset {dataset_id}"
    return ""


def _dataset_display_name(raw_name: str | None, dataset_id: int) -> str:
    s = (raw_name or "").strip()
    return s if s else f"Dataset {dataset_id}"


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    conn = _connect()
    try:
        init_library_tables(conn)
        yield conn.cursor()
        conn.commit()
    finally:
        conn.close()


def _extract_dataset_index(content: str) -> tuple[int, dict[str, Any]]:
    from rengu_flow_ui.dataset_form import parse_toml_to_form

    stripped = (content or "").strip()
    if not stripped:
        return 0, {}

    try:
        form, _warnings = parse_toml_to_form(stripped)
    except Exception:
        return 0, {}

    directories = form.get("_directories") or []
    paths: list[str] = []
    for entry in directories:
        if not isinstance(entry, dict):
            continue
        path_val = entry.get("path")
        if isinstance(path_val, str) and path_val.strip():
            paths.append(path_val.strip())

    meta: dict[str, Any] = {}
    resolutions = form.get("resolutions")
    if resolutions is not None:
        meta["resolutions"] = resolutions
    frame_buckets = form.get("frame_buckets")
    if frame_buckets is not None:
        meta["frame_buckets"] = frame_buckets
    if paths:
        meta["directory_paths"] = paths[:32]
    return len(paths), meta


# --- Datasets ---


def list_dataset_ids() -> list[int]:
    with _connect() as conn:
        init_library_tables(conn)
        rows = conn.execute(
            f"SELECT id FROM datasets {_library_order_clause()}"
        ).fetchall()
    return [int(r["id"]) for r in rows]


def _dataset_summary_row(r: sqlite3.Row) -> dict[str, Any]:
    meta = _safe_meta_json(r["meta_json"])
    did = int(r["id"])
    return {
        "id": did,
        "name": _dataset_display_name(r["name"] if "name" in r.keys() else "", did),
        "directory_count": _safe_int_column(r["directory_count"], 0),
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "directory_paths": meta.get("directory_paths") or [],
    }


def list_datasets_summary(
    *,
    sort: str | None = None,
    order: str | None = None,
) -> list[dict[str, Any]]:
    clause = _library_order_clause(sort, order)
    with _connect() as conn:
        init_library_tables(conn)
        rows = conn.execute(
            f"""
            SELECT id, name, directory_count, created_at, updated_at, meta_json
            FROM datasets {clause}
            """
        ).fetchall()
    return [_dataset_summary_row(r) for r in rows]


def search_datasets(
    q: str = "",
    *,
    page: int = 1,
    page_size: int = 20,
    sort: str | None = None,
    order: str | None = None,
) -> dict[str, Any]:
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size
    term = (q or "").strip()
    like = f"%{term}%"
    where = ""
    params: list[Any] = []
    if term:
        where = (
            "WHERE CAST(id AS TEXT) LIKE ? OR COALESCE(name, '') LIKE ? "
            "OR COALESCE(meta_json, '') LIKE ?"
        )
        params = [like, like, like]
    with _connect() as conn:
        init_library_tables(conn)
        total_row = conn.execute(
            f"SELECT COUNT(*) AS n FROM datasets {where}",
            params,
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0
        clause = _library_order_clause(sort, order)
        rows = conn.execute(
            f"""
            SELECT id, name, directory_count, created_at, updated_at, meta_json
            FROM datasets {where}
            {clause}
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


def dataset_exists(dataset_id: str | int) -> bool:
    did = _coerce_record_id(dataset_id)
    with _connect() as conn:
        init_library_tables(conn)
        row = conn.execute("SELECT 1 FROM datasets WHERE id = ?", (did,)).fetchone()
    return row is not None


def read_dataset_row(dataset_id: str | int) -> dict[str, Any]:
    did = _coerce_record_id(dataset_id)
    with _connect() as conn:
        init_library_tables(conn)
        row = conn.execute(
            "SELECT id, name, content FROM datasets WHERE id = ?", (did,)
        ).fetchone()
    if row is None:
        raise FileNotFoundError(dataset_id)
    raw = row["content"]
    content = "" if raw is None else (raw if isinstance(raw, str) else str(raw))
    return {
        "id": did,
        "name": _dataset_display_name(row["name"], did),
        "content": content,
    }


def read_dataset_text(dataset_id: str | int) -> str:
    return read_dataset_row(dataset_id)["content"]


def refresh_dataset_index(dataset_id: str | int) -> None:
    did = _coerce_record_id(dataset_id)
    with _cursor() as cur:
        row = cur.execute("SELECT content FROM datasets WHERE id = ?", (did,)).fetchone()
        if row is None:
            raise FileNotFoundError(dataset_id)
        content = row["content"]
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        directory_count, meta = _extract_dataset_index(content)
        cur.execute(
            """
            UPDATE datasets
            SET directory_count = ?, meta_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (directory_count, json.dumps(meta), _now(), did),
        )


def insert_dataset(content: str, name: str | None = None) -> int:
    from rengu_flow_ui.datasets_store import prepare_dataset_content_for_storage

    now = _now()
    content = prepare_dataset_content_for_storage(content, name)
    directory_count, meta = _extract_dataset_index(content)
    meta_json = json.dumps(meta)
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO datasets (
                content, name, directory_count, meta_json, created_at, updated_at
            ) VALUES (?, '', ?, ?, ?, ?)
            """,
            (content, directory_count, meta_json, now, now),
        )
        did = int(cur.lastrowid)
        final_name = _normalize_dataset_name(name, did)
        cur.execute("UPDATE datasets SET name = ? WHERE id = ?", (final_name, did))
        return did


def update_dataset_text(
    dataset_id: str | int,
    content: str,
    *,
    name: str | None = None,
) -> int:
    did = _coerce_record_id(dataset_id)
    now = _now()
    from rengu_flow_ui.datasets_store import prepare_dataset_content_for_storage

    row = None
    if name is None:
        try:
            row = read_dataset_row(did)
        except FileNotFoundError:
            pass
    storage_name = name if name is not None else (row["name"] if row else None)
    content = prepare_dataset_content_for_storage(content, storage_name)
    directory_count, meta = _extract_dataset_index(content)
    meta_json = json.dumps(meta)
    with _cursor() as cur:
        if name is not None:
            final_name = _normalize_dataset_name(name, did)
            cur.execute(
                """
                UPDATE datasets
                SET content = ?, name = ?, directory_count = ?, meta_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (content, final_name, directory_count, meta_json, now, did),
            )
        else:
            cur.execute(
                """
                UPDATE datasets
                SET content = ?, directory_count = ?, meta_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (content, directory_count, meta_json, now, did),
            )
        if cur.rowcount == 0:
            raise FileNotFoundError(dataset_id)
    return did


def delete_dataset(dataset_id: str | int) -> None:
    did = _coerce_record_id(dataset_id)
    with _cursor() as cur:
        cur.execute("DELETE FROM datasets WHERE id = ?", (did,))
        if cur.rowcount == 0:
            raise FileNotFoundError(dataset_id)


def duplicate_dataset(dataset_id: str | int) -> int:
    row = read_dataset_row(dataset_id)
    base = row["name"]
    if base.startswith("Dataset ") and base[8:].isdigit():
        copy_name = f"Dataset {row['id']} (copy)"
    else:
        copy_name = f"{base} (copy)" if base else None
    return insert_dataset(row["content"], name=copy_name)
