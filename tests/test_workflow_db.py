"""Tests for rengu_flow_ui.workflow_db (see docs/spec/workflows.md, "Persistence").

The load-bearing property under test is that `content` (editor) and `state_json` (executor) never
collide: `update_graph` must never touch `state_json` and `mutate_state` must never touch
`content`, and `mutate_state`'s compare-and-swap must never lose a concurrent update.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from rengu_flow_ui import db, workflow_db

# ------------------------------------------------------------------------------ CRUD


def test_create_get_workflow(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("My workflow", '{"nodes": []}')
    assert row.id >= 1
    assert row.name == "My workflow"
    assert row.content == '{"nodes": []}'
    assert row.state_json == "{}"
    assert row.version == 0
    assert row.created_at and row.updated_at

    fetched = workflow_db.get_workflow(row.id)
    assert fetched == row


def test_create_workflow_defaults_empty_name(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("", "{}")
    assert row.name == ""


def test_get_workflow_missing_raises_keyerror(ui_data_tmp: Path) -> None:
    with pytest.raises(KeyError):
        workflow_db.get_workflow(404)


def test_list_workflows(ui_data_tmp: Path) -> None:
    assert workflow_db.list_workflows() == []
    a = workflow_db.create_workflow("A", "{}")
    b = workflow_db.create_workflow("B", "{}")
    listed = workflow_db.list_workflows()
    assert [w.id for w in listed] == [a.id, b.id]


def test_delete_workflow(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", "{}")
    workflow_db.delete_workflow(row.id)
    assert workflow_db.list_workflows() == []
    with pytest.raises(KeyError):
        workflow_db.get_workflow(row.id)


def test_delete_missing_workflow_raises_keyerror(ui_data_tmp: Path) -> None:
    with pytest.raises(KeyError):
        workflow_db.delete_workflow(404)


def test_workflows_table_self_heals_on_preexisting_db_without_version_bump(
    ui_data_tmp: Path,
) -> None:
    """A DB created before this table existed must gain it via CREATE TABLE IF NOT EXISTS,
    with no SCHEMA_VERSION bump — the additive-migration policy in db.py:30-38.
    """
    conn = db._connect()
    try:
        conn.execute("DROP TABLE IF EXISTS workflows")
        conn.commit()
    finally:
        conn.close()

    before = db.stored_schema_version()
    db.init_db()  # what happens on every app startup
    after = db.stored_schema_version()
    assert before == after

    # The table is usable again, healed in place.
    row = workflow_db.create_workflow("Healed", "{}")
    assert workflow_db.get_workflow(row.id).name == "Healed"


# ------------------------------------------------------------------------------ update_graph


def test_update_graph_bumps_version_and_updates_content(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", '{"nodes": []}')
    updated = workflow_db.update_graph(
        row.id, '{"nodes": [1]}', expected_version=row.version
    )
    assert updated.content == '{"nodes": [1]}'
    assert updated.version == row.version + 1


def test_update_graph_stale_version_raises_and_changes_nothing(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", '{"nodes": []}')

    with pytest.raises(workflow_db.StaleWorkflowError):
        workflow_db.update_graph(
            row.id, '{"nodes": ["intruder"]}', expected_version=row.version + 1
        )

    unchanged = workflow_db.get_workflow(row.id)
    assert unchanged.content == row.content
    assert unchanged.version == row.version
    assert unchanged.updated_at == row.updated_at


def test_update_graph_stale_error_reports_versions(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", "{}")
    workflow_db.update_graph(row.id, "{}", expected_version=row.version)  # version now 1

    with pytest.raises(workflow_db.StaleWorkflowError) as exc_info:
        workflow_db.update_graph(row.id, "{}", expected_version=row.version)

    err = exc_info.value
    assert err.expected_version == row.version
    assert err.actual_version == row.version + 1


def test_update_graph_missing_workflow_raises_keyerror(ui_data_tmp: Path) -> None:
    with pytest.raises(KeyError):
        workflow_db.update_graph(404, "{}", expected_version=0)


def test_update_graph_never_touches_state_json(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", '{"nodes": []}')
    workflow_db.mutate_state(row.id, lambda s: {**s, "status": "running"})
    state_before = workflow_db.get_state(row.id)

    updated = workflow_db.update_graph(
        row.id, '{"nodes": ["n1"]}', expected_version=row.version
    )

    assert updated.content == '{"nodes": ["n1"]}'
    assert workflow_db.get_state(row.id) == state_before == {"status": "running"}


# ------------------------------------------------------------------------------ mutate_state


def test_mutate_state_never_touches_content(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", '{"nodes": [1, 2, 3]}')

    workflow_db.mutate_state(row.id, lambda s: {**s, "status": "running"})

    after = workflow_db.get_workflow(row.id)
    assert after.content == row.content
    assert after.version == row.version


def test_mutate_state_returns_new_state_and_persists_it(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", "{}")

    def set_status(state: dict) -> dict:
        return {**state, "status": "running"}

    result = workflow_db.mutate_state(row.id, set_status)
    assert result == {"status": "running"}
    assert workflow_db.get_state(row.id) == {"status": "running"}


def test_mutate_state_in_place_mutation_is_also_persisted(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", "{}")

    def bump(state: dict) -> None:
        state["counter"] = state.get("counter", 0) + 1

    workflow_db.mutate_state(row.id, bump)
    workflow_db.mutate_state(row.id, bump)
    assert workflow_db.get_state(row.id) == {"counter": 2}


def test_mutate_state_missing_workflow_raises_keyerror(ui_data_tmp: Path) -> None:
    with pytest.raises(KeyError):
        workflow_db.mutate_state(404, lambda s: s)


def test_get_state_defaults_to_empty_dict(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("A", "{}")
    assert workflow_db.get_state(row.id) == {}


def test_mutate_state_concurrent_updates_do_not_lose_a_write(ui_data_tmp: Path) -> None:
    """The test that matters: force the exact interleaving compare-and-swap exists to fix.

    Two threads both read `state_json` while it still holds `counter: 0` (a shared barrier
    forces both reads to happen before either write), then both attempt to write their own
    `counter: 1`. Without CAS, whichever write lands second unconditionally overwrites the
    first and the final counter is 1 -- one increment silently vanished. With the
    read-modify-write-with-compare-and-swap-and-retry in `mutate_state`, the loser's write is
    rejected (the row no longer matches the value it read), it retries against the fresh
    value, and the final counter is 2: no update is lost.

    Verified by breaking the implementation: temporarily dropping the `AND state_json = ?`
    clause from `mutate_state`'s UPDATE (making it an unconditional read-modify-write) makes
    this test fail with `final == {"counter": 1}` instead of `{"counter": 2}`, exactly the
    lost-update anomaly this design exists to prevent. Restored before committing.
    """
    row = workflow_db.create_workflow("A", "{}")
    workflow_db.mutate_state(row.id, lambda s: {**s, "counter": 0})

    barrier = threading.Barrier(2)
    lock = threading.Lock()
    counter_calls = {"n": 0}

    def bump(state: dict) -> dict:
        new_state = {**state, "counter": state.get("counter", 0) + 1}
        # Only the very first attempt from each thread synchronizes here, guaranteeing both
        # threads have read the same pre-write value before either one writes. A retried
        # attempt (after losing the compare-and-swap race) must not wait again: at that point
        # only one thread is still running and a second party would never arrive.
        with lock:
            counter_calls["n"] += 1
            n = counter_calls["n"]
        if n <= 2:
            barrier.wait(timeout=5)
        return new_state

    def worker() -> None:
        workflow_db.mutate_state(row.id, bump)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive() and not t2.is_alive()

    final = workflow_db.get_state(row.id)
    assert final == {"counter": 2}


# ------------------------------------------------------------------------------ clone_workflow


def test_clone_workflow_copies_content_discards_state_resets_version(
    ui_data_tmp: Path,
) -> None:
    row = workflow_db.create_workflow("Source", '{"nodes": [1, 2]}')
    workflow_db.update_graph(row.id, '{"nodes": [1, 2, 3]}', expected_version=row.version)
    workflow_db.mutate_state(
        row.id, lambda s: {**s, "status": "done", "current_node": "n2"}
    )
    source = workflow_db.get_workflow(row.id)

    clone = workflow_db.clone_workflow(row.id)

    assert clone.id != source.id
    assert clone.content == source.content
    assert clone.state_json == "{}"
    assert clone.version == 0
    assert clone.name == "Source (copy)"
    # The original is untouched.
    assert workflow_db.get_state(row.id) == {"status": "done", "current_node": "n2"}


def test_clone_workflow_with_explicit_name(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("Source", "{}")
    clone = workflow_db.clone_workflow(row.id, name="Renamed clone")
    assert clone.name == "Renamed clone"


def test_clone_workflow_with_empty_base_name(ui_data_tmp: Path) -> None:
    row = workflow_db.create_workflow("", "{}")
    clone = workflow_db.clone_workflow(row.id)
    assert clone.name == ""


def test_clone_missing_workflow_raises_keyerror(ui_data_tmp: Path) -> None:
    with pytest.raises(KeyError):
        workflow_db.clone_workflow(404)


# ------------------------------------------------------------------------------ node_dir


def test_node_dir_layout(ui_data_tmp: Path) -> None:
    p = workflow_db.node_dir(7, "n2")
    assert p == (ui_data_tmp / "workflows" / "7" / "n2").resolve()


def test_node_dir_rejects_parent_traversal(ui_data_tmp: Path) -> None:
    with pytest.raises(KeyError):
        workflow_db.node_dir(7, "..")
    with pytest.raises(KeyError):
        workflow_db.node_dir(7, "../../etc")
    with pytest.raises(KeyError):
        workflow_db.node_dir(7, "n1/../../escape")


def test_node_dir_rejects_absolute_path(ui_data_tmp: Path) -> None:
    import os

    abs_path = "C:\\Windows\\System32" if os.name == "nt" else "/etc/passwd"
    with pytest.raises(KeyError):
        workflow_db.node_dir(7, abs_path)


def test_node_dir_rejects_self(ui_data_tmp: Path) -> None:
    with pytest.raises(KeyError):
        workflow_db.node_dir(7, ".")


# ------------------------------------------------------------------------------ workflows_version


def test_workflows_version_bumps_on_writes(ui_data_tmp: Path) -> None:
    before = workflow_db.workflows_version()
    row = workflow_db.create_workflow("A", "{}")
    after_create = workflow_db.workflows_version()
    assert after_create > before

    workflow_db.update_graph(row.id, "{}", expected_version=row.version)
    after_update = workflow_db.workflows_version()
    assert after_update > after_create

    workflow_db.mutate_state(row.id, lambda s: {**s, "status": "running"})
    after_mutate = workflow_db.workflows_version()
    assert after_mutate > after_update

    workflow_db.delete_workflow(row.id)
    after_delete = workflow_db.workflows_version()
    assert after_delete > after_mutate
