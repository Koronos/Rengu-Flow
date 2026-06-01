"""Tests for the DB schema-version guard."""

from pathlib import Path

import pytest

from rengu_flow_ui import db, schema_guard


def test_schema_action_ok_cases() -> None:
    # Absent DB, legacy-unstamped (0), and matching versions all pass without prompting.
    assert db.schema_action(None, 1) == "ok"
    assert db.schema_action(0, 1) == "ok"
    assert db.schema_action(1, 1) == "ok"


def test_schema_action_incompatible_on_real_mismatch() -> None:
    assert db.schema_action(1, 2) == "incompatible"
    assert db.schema_action(2, 1) == "incompatible"


def test_init_db_stamps_user_version(ui_data_tmp: Path) -> None:
    # The fixture already ran init_db(); re-running is idempotent and the stamp persists.
    db.init_db()
    assert db.stored_schema_version() == db.SCHEMA_VERSION


def test_init_db_heals_drifted_jobs_table(ui_data_tmp: Path) -> None:
    """A `jobs` table stamped at the current version but missing additive columns self-heals.

    Reproduces the real bug: a DB created mid-development sat at the current `user_version`
    yet lacked `cache_only`/`trust_cache`/`regenerate_cache`, so every save crashed with
    `OperationalError: table jobs has no column named cache_only`.
    """
    import sqlite3

    with sqlite3.connect(db.db_path()) as conn:
        conn.execute("DROP TABLE jobs")
        # The real drifted layout: every original column present, only the cache trio missing
        # (and the now-defunct `config_id` still hanging around).
        conn.execute(
            "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, config_id INTEGER, "
            "config_path TEXT NOT NULL, state TEXT NOT NULL, pid INTEGER, run_dir TEXT, "
            "output_dir TEXT, num_gpus INTEGER NOT NULL DEFAULT 1, resume_from TEXT, "
            "log_path TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, "
            "exit_code INTEGER, extra_args TEXT NOT NULL DEFAULT '', queue_position INTEGER, "
            "config_content TEXT NOT NULL DEFAULT '', source_run_dir TEXT)"
        )
        conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION}")
        conn.commit()

    db.init_db()  # additive ADD COLUMN, not a wipe

    with sqlite3.connect(db.db_path()) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert {"cache_only", "trust_cache", "regenerate_cache"} <= cols

    # The insert that used to 500 now succeeds.
    job = db.create_job(config_path="x.toml", log_path="x.log", state="new", cache_only=True)
    assert job.cache_only is True


def test_ensure_schema_compatible_passes_on_current_db(ui_data_tmp: Path) -> None:
    # Fixture DB is freshly stamped at the current version → guard returns without prompting.
    schema_guard.ensure_schema_compatible()


def test_ensure_schema_compatible_blocks_incompatible_non_tty(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An incompatible stamped version with no TTY must abort rather than touch the DB.
    monkeypatch.setattr(db, "stored_schema_version", lambda: db.SCHEMA_VERSION + 1)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        schema_guard.ensure_schema_compatible()


def test_ensure_schema_compatible_recreates_on_yes(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Incompatible + interactive "yes" → wipe-and-recreate, leaving a current-version DB.
    import sqlite3

    monkeypatch.setattr(db, "stored_schema_version", lambda: db.SCHEMA_VERSION + 1)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    schema_guard.ensure_schema_compatible()  # resets via the real reset_ui_database
    # Read the stamp directly (the helper is monkeypatched to report a fake version).
    with sqlite3.connect(db.db_path()) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
