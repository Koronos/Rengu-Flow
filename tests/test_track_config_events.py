"""Tests for UI-side config-edit timeline events (rengu_flow_ui.job_queue._record_config_edit)."""

import pytest

from rengu_flow_ui.job_queue import _record_config_edit
from rengu_track import EVENT_CONFIG_EDITED, read_events

pytestmark = pytest.mark.no_ui_db


def test_record_config_edit_appends_diff(tmp_path):
    old = '[optimizer]\nlr = 0.0001\n[preview]\nevery = 100\n'
    new = '[optimizer]\nlr = 0.0002\n[preview]\nevery = 100\n'
    _record_config_edit(str(tmp_path), old, new)

    events = read_events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == EVENT_CONFIG_EDITED
    assert ev["source"] == "ui"
    assert ev["payload"]["diff"]["changed"] == {"optimizer.lr": [0.0001, 0.0002]}


def test_record_config_edit_noop_when_unchanged(tmp_path):
    same = '[optimizer]\nlr = 0.0001\n'
    _record_config_edit(str(tmp_path), same, same)
    assert read_events(tmp_path) == []


def test_record_config_edit_noop_on_first_sync(tmp_path):
    # No prior config text -> nothing to diff (the trainer records run_started instead).
    _record_config_edit(str(tmp_path), "", '[optimizer]\nlr = 0.0001\n')
    assert read_events(tmp_path) == []
