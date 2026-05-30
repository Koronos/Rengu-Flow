"""Tests for the DB schema-version guard."""

from pathlib import Path

from rengu_flow_ui import db


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
