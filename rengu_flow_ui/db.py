"""SQLite job registry for UI-launched training processes."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from rengu_flow_ui.library_db import init_library_tables
from rengu_flow_ui.settings import db_path, ensure_data_dirs

# Bump when the DB schema changes incompatibly. Stored in the file via PRAGMA user_version
# and stamped on init. Startup compares it (see schema_action / cli guard) and, on a real
# mismatch, asks the user to wipe-and-recreate or stay on the previous app version. Until a
# TOML export/import migration exists (see docs/developer/run-model-redesign.md), a bump
# means existing local libraries are discarded.
SCHEMA_VERSION = 3


def _coerce_job_id(job_id: str | int) -> int:
    if isinstance(job_id, bool):
        raise KeyError(job_id)
    if isinstance(job_id, int):
        return job_id
    s = str(job_id).strip()
    if not s.isdigit():
        raise KeyError(job_id)
    return int(s)


@dataclass
class JobRecord:
    id: int
    config_path: str
    state: str
    pid: int | None
    run_dir: str | None
    output_dir: str | None
    num_gpus: int
    resume_from: str | None
    log_path: str
    started_at: str
    finished_at: str | None
    exit_code: int | None
    extra_args: str
    queue_position: int | None = None
    source_run_dir: str | None = None
    # Immutable snapshot of the run's own config TOML (library refs intact, pre-staging).
    # Makes a run self-contained and is the seed for "new run from config".
    # Empty for legacy/imported rows.
    config_content: str = ""
    # Cache toggles, mirrored from extra_args so the queue UI can show/edit them
    # without parsing the CLI string.
    cache_only: bool = False
    trust_cache: bool = False
    regenerate_cache: bool = False


def _connect() -> sqlite3.Connection:
    ensure_data_dirs()
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def reset_ui_database() -> Path:
    """Delete jobs.db and recreate an empty schema (datasets, jobs)."""
    ensure_data_dirs()
    path = db_path()
    if path.exists():
        path.unlink()
    init_db()
    return path


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_path TEXT NOT NULL,
                state TEXT NOT NULL,
                pid INTEGER,
                run_dir TEXT,
                output_dir TEXT,
                num_gpus INTEGER NOT NULL DEFAULT 1,
                resume_from TEXT,
                log_path TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                exit_code INTEGER,
                extra_args TEXT NOT NULL DEFAULT '',
                queue_position INTEGER,
                source_run_dir TEXT,
                config_content TEXT NOT NULL DEFAULT '',
                cache_only INTEGER NOT NULL DEFAULT 0,
                trust_cache INTEGER NOT NULL DEFAULT 0,
                regenerate_cache INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        init_library_tables(conn)
        conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
        conn.commit()


def stored_schema_version() -> int | None:
    """``user_version`` of the existing DB file, or ``None`` when there is no DB yet."""
    if not db_path().exists():
        return None
    with _connect() as conn:
        row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def schema_action(stored: int | None, current: int = SCHEMA_VERSION) -> str:
    """Pure decision for the startup schema guard.

    Returns ``"ok"`` when the DB is absent, legacy-unstamped (``0`` — treated as
    compatible and re-stamped on init), or already at ``current``; ``"incompatible"``
    when a stamped version differs from ``current`` (caller must prompt/abort).
    """
    if stored is None or stored == 0 or stored == current:
        return "ok"
    return "incompatible"


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    conn = _connect()
    try:
        yield conn.cursor()
        conn.commit()
    finally:
        conn.close()


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    keys = row.keys()
    return JobRecord(
        id=int(row["id"]),
        config_path=row["config_path"],
        state=row["state"],
        pid=row["pid"],
        run_dir=row["run_dir"],
        output_dir=row["output_dir"],
        num_gpus=row["num_gpus"],
        resume_from=row["resume_from"],
        log_path=row["log_path"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        exit_code=row["exit_code"],
        extra_args=row["extra_args"] or "",
        queue_position=row["queue_position"] if "queue_position" in keys else None,
        source_run_dir=row["source_run_dir"] if "source_run_dir" in keys else None,
        config_content=(
            row["config_content"]
            if "config_content" in keys and row["config_content"] is not None
            else ""
        ),
        cache_only=bool(row["cache_only"]) if "cache_only" in keys else False,
        trust_cache=bool(row["trust_cache"]) if "trust_cache" in keys else False,
        regenerate_cache=bool(row["regenerate_cache"]) if "regenerate_cache" in keys else False,
    )


def find_job_by_run_dir(run_dir: str) -> JobRecord | None:
    resolved = str(Path(run_dir).expanduser().resolve())
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE run_dir = ? LIMIT 1", (resolved,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_job(row)


def create_imported_job(
    *,
    run_dir: str,
    config_path: str,
    log_path: str,
    output_dir: str,
    started_at: str,
    finished_at: str | None = None,
    exit_code: int | None = 0,
    extra_args: str = "",
    source_run_dir: str | None = None,
    config_content: str = "",
) -> JobRecord:
    """Register a finished script-mode run for UI history and monitoring."""
    src = source_run_dir or run_dir
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                config_path, state, pid, run_dir, output_dir,
                num_gpus, resume_from, log_path, started_at, finished_at,
                exit_code, extra_args, queue_position, source_run_dir, config_content
            ) VALUES (?, 'finished', NULL, ?, ?, 1, NULL, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                config_path,
                run_dir,
                output_dir,
                log_path,
                started_at,
                finished_at,
                exit_code,
                extra_args,
                src,
                config_content,
            ),
        )
        job_id = int(cur.lastrowid)
    return get_job(job_id)


def create_job(
    *,
    config_path: str,
    log_path: str,
    state: str = "pending",
    num_gpus: int = 1,
    resume_from: str | None = None,
    output_dir: str | None = None,
    extra_args: str = "",
    queue_position: int | None = None,
    source_run_dir: str | None = None,
    config_content: str = "",
    cache_only: bool = False,
    trust_cache: bool = False,
    regenerate_cache: bool = False,
) -> JobRecord:
    now = datetime.now(timezone.utc).isoformat()
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                config_path, state, pid, run_dir, output_dir,
                num_gpus, resume_from, log_path, started_at, extra_args, queue_position,
                source_run_dir, config_content, cache_only, trust_cache, regenerate_cache
            ) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config_path,
                state,
                output_dir,
                num_gpus,
                resume_from,
                log_path,
                now,
                extra_args,
                queue_position,
                source_run_dir,
                config_content,
                int(cache_only),
                int(trust_cache),
                int(regenerate_cache),
            ),
        )
        job_id = int(cur.lastrowid)
    return get_job(job_id)


def get_job(job_id: str | int) -> JobRecord:
    jid = _coerce_job_id(job_id)
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    return _row_to_job(row)


def delete_job(job_id: str | int) -> None:
    jid = _coerce_job_id(job_id)
    with _cursor() as cur:
        cur.execute("DELETE FROM jobs WHERE id = ?", (jid,))


def list_jobs(limit: int = 200) -> list[JobRecord]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_job(r) for r in rows]


def update_job(job_id: str | int, **fields: Any) -> JobRecord:
    allowed = {
        "state",
        "pid",
        "run_dir",
        "output_dir",
        "finished_at",
        "exit_code",
        "config_path",
        "num_gpus",
        "resume_from",
        "extra_args",
        "queue_position",
        "source_run_dir",
        "config_content",
        "cache_only",
        "trust_cache",
        "regenerate_cache",
    }
    parts = []
    values: list[Any] = []
    for key, val in fields.items():
        if key not in allowed:
            continue
        parts.append(f"{key} = ?")
        values.append(val)
    if not parts:
        return get_job(job_id)
    jid = _coerce_job_id(job_id)
    values.append(jid)
    with _cursor() as cur:
        cur.execute(f"UPDATE jobs SET {', '.join(parts)} WHERE id = ?", values)
    return get_job(jid)
