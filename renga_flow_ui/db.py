"""SQLite job registry for UI-launched training processes."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from renga_flow_ui.library_db import init_library_tables
from renga_flow_ui.settings import db_path, ensure_data_dirs


@dataclass
class JobRecord:
    id: str
    config_id: str | None
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


def _connect() -> sqlite3.Connection:
    ensure_data_dirs()
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                config_id TEXT,
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
                queue_position INTEGER
            )
            """
        )
        _migrate_jobs_table(conn)
        init_library_tables(conn)
        conn.commit()


def _migrate_jobs_table(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "queue_position" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN queue_position INTEGER")
    if "source_run_dir" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN source_run_dir TEXT")


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    conn = _connect()
    try:
        yield conn.cursor()
        conn.commit()
    finally:
        conn.close()


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        config_id=row["config_id"],
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
        queue_position=row["queue_position"] if "queue_position" in row.keys() else None,
        source_run_dir=row["source_run_dir"] if "source_run_dir" in row.keys() else None,
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
    config_id: str | None,
    log_path: str,
    output_dir: str,
    started_at: str,
    finished_at: str | None = None,
    exit_code: int | None = 0,
    extra_args: str = "",
    source_run_dir: str | None = None,
) -> JobRecord:
    """Register a finished script-mode run for UI history and monitoring."""
    job_id = uuid4().hex[:12]
    src = source_run_dir or run_dir
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                id, config_id, config_path, state, pid, run_dir, output_dir,
                num_gpus, resume_from, log_path, started_at, finished_at,
                exit_code, extra_args, queue_position, source_run_dir
            ) VALUES (?, ?, ?, 'finished', NULL, ?, ?, 1, NULL, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                job_id,
                config_id,
                config_path,
                run_dir,
                output_dir,
                log_path,
                started_at,
                finished_at,
                exit_code,
                extra_args,
                src,
            ),
        )
    return get_job(job_id)


def create_job(
    *,
    config_path: str,
    config_id: str | None,
    log_path: str,
    num_gpus: int = 1,
    resume_from: str | None = None,
    output_dir: str | None = None,
    extra_args: str = "",
    queue_position: int | None = None,
    source_run_dir: str | None = None,
) -> JobRecord:
    job_id = uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                id, config_id, config_path, state, pid, run_dir, output_dir,
                num_gpus, resume_from, log_path, started_at, extra_args, queue_position,
                source_run_dir
            ) VALUES (?, ?, ?, 'pending', NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                config_id,
                config_path,
                output_dir,
                num_gpus,
                resume_from,
                log_path,
                now,
                extra_args,
                queue_position,
                source_run_dir,
            ),
        )
    return get_job(job_id)


def get_job(job_id: str) -> JobRecord:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    return _row_to_job(row)


def delete_job(job_id: str) -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def list_jobs(limit: int = 200) -> list[JobRecord]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_job(r) for r in rows]


def update_job(job_id: str, **fields: Any) -> JobRecord:
    allowed = {
        "state",
        "pid",
        "run_dir",
        "output_dir",
        "finished_at",
        "exit_code",
        "config_id",
        "config_path",
        "num_gpus",
        "resume_from",
        "extra_args",
        "queue_position",
        "source_run_dir",
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
    values.append(job_id)
    with _cursor() as cur:
        cur.execute(f"UPDATE jobs SET {', '.join(parts)} WHERE id = ?", values)
    return get_job(job_id)
