"""Tests for rengu_flow_ui.workflow_routes (see docs/spec/workflows.md, "Persistence" and
"Execution model").

``/start`` and ``/cancel`` are mostly exercised against a stand-in ``workflow_runner``, which pins
the contract the routes must call — ``tick()``, ``start_workflow(id, *, from_node, force)`` raising
a readable ``ValueError``, and ``cancel_workflow(id)`` — without coupling to the runner's
internals. ``test_start_reaches_the_real_runner`` deliberately skips the stand-in: the decoy alone
once hid a wiring break, because the handlers resolve the package attribute rather than
``sys.modules``.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from rengu_flow_ui import workflow_db

# ------------------------------------------------------------------------------ helpers


def _create(client, name: str = "My workflow") -> dict:
    resp = client.post("/api/v1/workflows", json={"name": name})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _node(node_id: str, node_type: str = "folder", **extra) -> dict:
    node = {"id": node_id, "type": node_type, "from": None, "config": {}}
    node.update(extra)
    return node


def _put_graph(client, workflow_id, nodes: list[dict], version: int, name: str = ""):
    return client.put(
        f"/api/v1/workflows/{workflow_id}",
        json={"graph": {"version": 1, "name": name, "variables": [], "nodes": nodes}, "version": version},
    )


@pytest.fixture
def fake_workflow_runner(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Install a stand-in ``rengu_flow_ui.workflow_runner`` and record calls made to it."""
    calls = {"tick": 0, "start": [], "cancel": []}
    fake = types.ModuleType("rengu_flow_ui.workflow_runner")

    def tick() -> None:
        calls["tick"] += 1

    def start_workflow(workflow_id, *, from_node=None, force=False) -> None:
        calls["start"].append((workflow_id, from_node, force))

    def cancel_workflow(workflow_id) -> None:
        calls["cancel"].append(workflow_id)

    fake.tick = tick
    fake.start_workflow = start_workflow
    fake.cancel_workflow = cancel_workflow
    monkeypatch.setitem(sys.modules, "rengu_flow_ui.workflow_runner", fake)
    # `from rengu_flow_ui import workflow_runner` resolves the package ATTRIBUTE, not sys.modules,
    # once the real module has been imported anywhere. Patching only sys.modules silently kept
    # handing the handlers the real runner — which is how these tests passed while the module did
    # not exist yet and broke the moment it landed. Patch both.
    import rengu_flow_ui

    monkeypatch.setattr(rengu_flow_ui, "workflow_runner", fake, raising=False)
    return calls


# ------------------------------------------------------------------------------ CRUD


def test_create_get_workflow(ui_client) -> None:
    created = _create(ui_client, "My workflow")
    assert created["name"] == "My workflow"
    assert created["version"] == 0
    assert created["graph"]["nodes"] == []
    assert created["state"] == {}
    assert created["stale"] == {}

    resp = ui_client.get(f"/api/v1/workflows/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_list_workflows_summary(ui_client) -> None:
    a = _create(ui_client, "A")
    _put_graph(ui_client, a["id"], [_node("n1")], version=0)

    resp = ui_client.get("/api/v1/workflows")
    assert resp.status_code == 200
    rows = resp.json()["workflows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == a["id"]
    assert row["name"] == "A"
    assert row["steps"] == 1
    assert row["status"] == "idle"
    assert "updated_at" in row


def test_delete_workflow(ui_client) -> None:
    created = _create(ui_client)
    resp = ui_client.delete(f"/api/v1/workflows/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    assert ui_client.get(f"/api/v1/workflows/{created['id']}").status_code == 404


def test_get_missing_workflow_404(ui_client) -> None:
    assert ui_client.get("/api/v1/workflows/999").status_code == 404


def test_delete_missing_workflow_404(ui_client) -> None:
    assert ui_client.delete("/api/v1/workflows/999").status_code == 404


@pytest.mark.parametrize("status", ["running", "cancelling"])
def test_delete_while_running_409_and_keeps_the_row(ui_client, status: str) -> None:
    """Deleting a live workflow is strictly worse than editing one, so it gets the same guard.

    The runner drives a workflow by its row: drop the row mid-run and nothing ever finishes the
    node, releases its GPU lease or kills its detached child — and there is no longer anything for
    ``reconcile_on_start`` (which walks ``list_workflows()``) to reconcile against, nor a Stop
    button to press. The row must survive the attempt.
    """
    created = _create(ui_client)

    def _set_status(state: dict) -> None:
        state["status"] = status

    workflow_db.mutate_state(created["id"], _set_status)

    resp = ui_client.delete(f"/api/v1/workflows/{created['id']}")
    assert resp.status_code == 409
    assert ui_client.get(f"/api/v1/workflows/{created['id']}").status_code == 200


# ------------------------------------------------------------------------------ PUT graph


def test_put_graph_saves_and_bumps_version(ui_client) -> None:
    created = _create(ui_client)
    resp = _put_graph(ui_client, created["id"], [_node("n1")], version=0)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["version"] == 1
    assert [n["id"] for n in body["graph"]["nodes"]] == ["n1"]


def test_put_graph_stale_version_409(ui_client) -> None:
    created = _create(ui_client)
    # First save lands (version 0 -> 1).
    ok = _put_graph(ui_client, created["id"], [_node("n1")], version=0)
    assert ok.status_code == 200

    # A second save still carrying the old version is stale.
    stale = _put_graph(ui_client, created["id"], [_node("n1"), _node("n2")], version=0)
    assert stale.status_code == 409


@pytest.mark.parametrize("status", ["running", "cancelling"])
def test_put_graph_while_running_409(ui_client, status: str) -> None:
    created = _create(ui_client)

    def _set_status(state: dict) -> None:
        state["status"] = status

    workflow_db.mutate_state(created["id"], _set_status)

    resp = _put_graph(ui_client, created["id"], [_node("n1")], version=0)
    assert resp.status_code == 409


def test_put_graph_missing_workflow_404(ui_client) -> None:
    resp = _put_graph(ui_client, 999, [_node("n1")], version=0)
    assert resp.status_code == 404


# ------------------------------------------------------------------------------ validate


def test_validate_returns_all_errors_not_just_first(ui_client) -> None:
    created = _create(ui_client)
    nodes = [
        _node("n1", node_type="bogus-type"),  # unknown type
        _node("n2", node_type="folder", **{"from": "n2"}),  # points at itself
        _node("n1", node_type="folder"),  # duplicate id
    ]
    saved = _put_graph(ui_client, created["id"], nodes, version=0)
    assert saved.status_code == 200, saved.text

    resp = ui_client.post(f"/api/v1/workflows/{created['id']}/validate")
    assert resp.status_code == 200
    errors = resp.json()["errors"]
    assert len(errors) == 3
    joined = " | ".join(errors)
    assert "unknown node type" in joined
    assert "points at itself" in joined
    assert "duplicate node id" in joined


def test_validate_missing_workflow_404(ui_client) -> None:
    assert ui_client.post("/api/v1/workflows/999/validate").status_code == 404


# ------------------------------------------------------------------------------ clone


def test_clone_does_not_carry_state(ui_client) -> None:
    created = _create(ui_client, "Original")
    saved = _put_graph(ui_client, created["id"], [_node("n1")], version=0)
    assert saved.status_code == 200

    def _mark_done(state: dict) -> None:
        state["status"] = "done"
        state["nodes"] = {
            "n1": {"status": "done", "output": {"path": "/out"}, "config_hash": "abc"}
        }

    workflow_db.mutate_state(created["id"], _mark_done)

    resp = ui_client.post(f"/api/v1/workflows/{created['id']}/clone")
    assert resp.status_code == 200, resp.text
    cloned = resp.json()
    assert cloned["id"] != created["id"]
    assert cloned["version"] == 0
    assert cloned["state"] == {}
    assert [n["id"] for n in cloned["graph"]["nodes"]] == ["n1"]

    # The clone must not have touched the original's saved state.
    original = ui_client.get(f"/api/v1/workflows/{created['id']}").json()
    assert original["state"]["status"] == "done"


def test_clone_missing_workflow_404(ui_client) -> None:
    assert ui_client.post("/api/v1/workflows/999/clone").status_code == 404


# ------------------------------------------------------------------------------ node log


def test_node_log_tail_by_offset(ui_client, ui_data_tmp: Path) -> None:
    created = _create(ui_client)
    saved = _put_graph(ui_client, created["id"], [_node("n1")], version=0)
    assert saved.status_code == 200

    node_dir = workflow_db.node_dir(created["id"], "n1")
    node_dir.mkdir(parents=True)
    log_path = node_dir / "node.log"
    log_path.write_text("first line\n", encoding="utf-8")

    resp1 = ui_client.get(f"/api/v1/workflows/{created['id']}/nodes/n1/log")
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["chunk"] == "first line\n"
    offset1 = body1["offset"]
    assert offset1 > 0

    with log_path.open("a", encoding="utf-8") as f:
        f.write("second line\n")

    resp2 = ui_client.get(f"/api/v1/workflows/{created['id']}/nodes/n1/log?offset={offset1}")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["chunk"] == "second line\n"
    assert body2["offset"] > offset1


def test_node_log_progress_marker(ui_client) -> None:
    created = _create(ui_client)
    saved = _put_graph(ui_client, created["id"], [_node("n1")], version=0)
    assert saved.status_code == 200

    from rengu_flow.control.progress_stream import format_progress_marker

    node_dir = workflow_db.node_dir(created["id"], "n1")
    node_dir.mkdir(parents=True)
    marker = format_progress_marker({"current": 3, "total": 10})
    (node_dir / "node.log").write_text(marker + "\n", encoding="utf-8")

    resp = ui_client.get(f"/api/v1/workflows/{created['id']}/nodes/n1/log")
    assert resp.status_code == 200
    progress = resp.json()["progress"]
    assert progress == {"current": 3, "total": 10}


def test_node_log_missing_workflow_404(ui_client) -> None:
    resp = ui_client.get("/api/v1/workflows/999/nodes/n1/log")
    assert resp.status_code == 404


def test_node_log_missing_node_404(ui_client) -> None:
    created = _create(ui_client)
    saved = _put_graph(ui_client, created["id"], [_node("n1")], version=0)
    assert saved.status_code == 200

    resp = ui_client.get(f"/api/v1/workflows/{created['id']}/nodes/nope/log")
    assert resp.status_code == 404


# ------------------------------------------------------------------------------ start / cancel


def test_start_calls_runner_and_ticks(ui_client, fake_workflow_runner: dict) -> None:
    created = _create(ui_client)
    resp = ui_client.post(
        f"/api/v1/workflows/{created['id']}/start", json={"from_node": "n1", "force": True}
    )
    assert resp.status_code == 200, resp.text
    assert fake_workflow_runner["start"] == [(str(created["id"]), "n1", True)]
    assert fake_workflow_runner["tick"] == 1


def test_start_without_body_calls_runner_with_defaults(
    ui_client, fake_workflow_runner: dict
) -> None:
    created = _create(ui_client)
    resp = ui_client.post(f"/api/v1/workflows/{created['id']}/start")
    assert resp.status_code == 200, resp.text
    assert fake_workflow_runner["start"] == [(str(created["id"]), None, False)]
    assert fake_workflow_runner["tick"] == 1


def test_start_value_error_maps_to_400_and_skips_tick(
    ui_client, fake_workflow_runner: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*_args, **_kwargs):
        raise ValueError("node n3 . quality.output_dir -> unknown variable ${outdir}")

    monkeypatch.setattr(sys.modules["rengu_flow_ui.workflow_runner"], "start_workflow", _raise)

    created = _create(ui_client)
    resp = ui_client.post(f"/api/v1/workflows/{created['id']}/start")
    assert resp.status_code == 400
    assert "unknown variable" in resp.json()["detail"]
    assert fake_workflow_runner["tick"] == 0


def test_start_missing_workflow_404(ui_client, fake_workflow_runner: dict) -> None:
    resp = ui_client.post("/api/v1/workflows/999/start")
    assert resp.status_code == 404
    assert fake_workflow_runner["start"] == []
    assert fake_workflow_runner["tick"] == 0


def test_cancel_calls_runner_and_ticks(ui_client, fake_workflow_runner: dict) -> None:
    created = _create(ui_client)
    resp = ui_client.post(f"/api/v1/workflows/{created['id']}/cancel")
    assert resp.status_code == 200, resp.text
    assert fake_workflow_runner["cancel"] == [str(created["id"])]
    assert fake_workflow_runner["tick"] == 1


def test_cancel_missing_workflow_404(ui_client, fake_workflow_runner: dict) -> None:
    resp = ui_client.post("/api/v1/workflows/999/cancel")
    assert resp.status_code == 404
    assert fake_workflow_runner["cancel"] == []
    assert fake_workflow_runner["tick"] == 0


def test_start_reaches_the_real_runner(ui_client, tmp_path: Path) -> None:
    """Integration seam: no decoy, so a wiring break between routes and runner shows up here.

    The unit tests above stub ``workflow_runner`` to pin the routes layer's contract. That stub
    hid a real defect once the runner landed — the handlers use ``from rengu_flow_ui import
    workflow_runner``, which reads the package attribute, so a sys.modules-only patch left them
    talking to the real module. This test deliberately keeps the real runner in place.

    The graph is built with ``PUT``, because ``POST /workflows`` takes a name and nothing else: the
    graph only ever enters through the versioned door, which is what makes two open tabs safe. An
    earlier version of this test passed the graph as a ``content`` field on the ``POST`` body,
    which pydantic dropped — so it ran the real runner over an *empty* graph and asserted only
    that some status existed. Assert the verdict of a specific node, or this covers nothing.
    """
    created = _create(ui_client, "integration")
    missing = tmp_path / "does-not-exist"
    saved = _put_graph(
        ui_client, created["id"], [_node("n1", config={"path": str(missing)})], version=0
    )
    assert saved.status_code == 200, saved.text

    resp = ui_client.post(f"/api/v1/workflows/{created['id']}/start", json={})
    assert resp.status_code == 200, resp.text

    # A folder node whose folder is not there is the shortest real verdict the runner can reach:
    # it runs inline, inside the synchronous tick /start performs, so it is decided by the time
    # the response comes back.
    state = ui_client.get(f"/api/v1/workflows/{created['id']}").json()["state"]
    assert state["status"] == "failed"
    node_state = state["nodes"]["n1"]
    assert node_state["status"] == "failed"
    assert "does-not-exist" in node_state["error"]
