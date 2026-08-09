"""API routes for Workflows (see docs/spec/workflows.md).

Thin HTTP/WS layer over ``workflow_db`` (persistence), ``workflow_graph`` (pure graph model) and
``workflow_runner`` (execution). Mirrors ``prep_routes.py``'s shape: a module with
``register_workflow_routes(app)``, pydantic models for the request envelope only, and an
``http_errors``-based mapping of domain errors to HTTP status.

``workflow_runner`` is imported lazily inside the handlers that need it (``/start``, ``/cancel``)
rather than at module scope, so this module — and everything that imports it — stays importable
regardless of that module's own load-time dependencies.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from rengu_flow.control.progress_stream import parse_last_progress_marker
from rengu_flow_ui import jobs, workflow_db, workflow_graph
from rengu_flow_ui._http_util import http_errors
from rengu_flow_ui.settings import ui_token

API_PREFIX = "/api/v1"

# A node's log is tailed while it is in one of these states; anything else is terminal (or not yet
# started), so the log GET/WS give it one last flush and stop rather than tailing forever.
_NODE_ACTIVE_STATUSES = ("running", "stopping")


@contextmanager
def _workflow_http_errors():
    try:
        with http_errors("Workflow not found"):
            yield
    except workflow_db.StaleWorkflowError as e:
        raise HTTPException(409, str(e))


#: While the workflow is in one of these the runner owns it: nodes are spawned, leases are held,
#: and the saved graph is what a tick reads on its next pass.
_RUNNER_OWNED_STATUSES = ("running", "cancelling")


def _reject_while_running(record: workflow_db.WorkflowRecord, action: str) -> None:
    """409 while the runner owns *record* — the guard both ``PUT`` and ``DELETE`` need.

    Editing under a live run would have a tick reading a graph that no longer matches the state it
    is driving. Deleting is strictly worse: the runner never finishes the workflow, so its node
    keeps its GPU lease and its detached child keeps running, with no row left to reconcile against
    and no Stop button to press. Stop first, then delete.
    """
    state = json.loads(record.state_json or "{}")
    if state.get("status") in _RUNNER_OWNED_STATUSES:
        raise HTTPException(409, f"Workflow is running; stop it before {action}")


def _require_node(workflow_id: str, node_id: str) -> workflow_db.WorkflowRecord:
    """Fetch the workflow and confirm ``node_id`` is one of its nodes, else ``KeyError``.

    A single check for both "workflow missing" and "node missing" — the routes that read a node's
    log or drive its progress have no other way to tell a typo'd node id from a real one, and both
    cases are a 404 to the caller.
    """
    record = workflow_db.get_workflow(workflow_id)
    graph = workflow_graph.parse_graph(json.loads(record.content or "{}"))
    if not any(node.id == node_id for node in graph.nodes):
        raise KeyError(node_id)
    return record


def _workflow_detail(record: workflow_db.WorkflowRecord) -> dict[str, Any]:
    """Graph + state + per-node staleness + version — the editor's single load payload."""
    graph = workflow_graph.parse_graph(json.loads(record.content or "{}"))
    state = json.loads(record.state_json or "{}")
    stale = workflow_graph.compute_stale(graph, state.get("nodes"))
    return {
        "id": record.id,
        "name": record.name,
        "graph": workflow_graph.graph_to_dict(graph),
        "state": state,
        "stale": stale,
        "version": record.version,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _workflow_summary(record: workflow_db.WorkflowRecord) -> dict[str, Any]:
    """id/name/step-count/status/updated_at — the list view's row, without the full graph."""
    graph = workflow_graph.parse_graph(json.loads(record.content or "{}"))
    state = json.loads(record.state_json or "{}")
    return {
        "id": record.id,
        "name": record.name,
        "steps": len(graph.nodes),
        "status": state.get("status", "idle"),
        "updated_at": record.updated_at,
    }


class CreateWorkflowBody(BaseModel):
    name: str = ""


class UpdateGraphBody(BaseModel):
    graph: dict
    version: int


class CloneWorkflowBody(BaseModel):
    name: str | None = None


class StartWorkflowBody(BaseModel):
    from_node: str | None = None
    force: bool = False


def register_workflow_routes(app: FastAPI) -> None:
    @app.get(f"{API_PREFIX}/workflows")
    def list_workflows_route() -> dict[str, Any]:
        return {"workflows": [_workflow_summary(w) for w in workflow_db.list_workflows()]}

    @app.post(f"{API_PREFIX}/workflows")
    def create_workflow_route(body: CreateWorkflowBody) -> dict[str, Any]:
        graph = workflow_graph.WorkflowGraph(name=body.name)
        content = json.dumps(workflow_graph.graph_to_dict(graph))
        record = workflow_db.create_workflow(body.name, content)
        return _workflow_detail(record)

    @app.get(f"{API_PREFIX}/workflows/{{workflow_id}}")
    def get_workflow_route(workflow_id: str) -> dict[str, Any]:
        with _workflow_http_errors():
            return _workflow_detail(workflow_db.get_workflow(workflow_id))

    @app.put(f"{API_PREFIX}/workflows/{{workflow_id}}")
    def update_workflow_route(workflow_id: str, body: UpdateGraphBody) -> dict[str, Any]:
        with _workflow_http_errors():
            _reject_while_running(workflow_db.get_workflow(workflow_id), "editing")
            graph = workflow_graph.parse_graph(body.graph)
            content = json.dumps(workflow_graph.graph_to_dict(graph))
            updated = workflow_db.update_graph(
                workflow_id, content, expected_version=body.version
            )
            return _workflow_detail(updated)

    @app.delete(f"{API_PREFIX}/workflows/{{workflow_id}}")
    def delete_workflow_route(workflow_id: str) -> dict[str, bool]:
        with _workflow_http_errors():
            _reject_while_running(workflow_db.get_workflow(workflow_id), "deleting it")
            workflow_db.delete_workflow(workflow_id)
        return {"ok": True}

    @app.post(f"{API_PREFIX}/workflows/{{workflow_id}}/clone")
    def clone_workflow_route(
        workflow_id: str, body: CloneWorkflowBody | None = None
    ) -> dict[str, Any]:
        with _workflow_http_errors():
            cloned = workflow_db.clone_workflow(
                workflow_id, name=(body.name if body else None)
            )
            return _workflow_detail(cloned)

    @app.post(f"{API_PREFIX}/workflows/{{workflow_id}}/validate")
    def validate_workflow_route(workflow_id: str) -> dict[str, list[str]]:
        with _workflow_http_errors():
            record = workflow_db.get_workflow(workflow_id)
        graph = workflow_graph.parse_graph(json.loads(record.content or "{}"))
        return {"errors": workflow_graph.validate(graph)}

    @app.post(f"{API_PREFIX}/workflows/{{workflow_id}}/start")
    def start_workflow_route(
        workflow_id: str, body: StartWorkflowBody | None = None
    ) -> dict[str, Any]:
        from rengu_flow_ui import workflow_runner

        with _workflow_http_errors():
            workflow_db.get_workflow(workflow_id)
            workflow_runner.start_workflow(
                workflow_id,
                from_node=(body.from_node if body else None),
                force=bool(body and body.force),
            )
        # Synchronous tick so the UI sees the effect immediately, without waiting for the
        # poller's interval — same pattern as job_queue.start_job_immediately.
        workflow_runner.tick()
        return _workflow_detail(workflow_db.get_workflow(workflow_id))

    @app.post(f"{API_PREFIX}/workflows/{{workflow_id}}/cancel")
    def cancel_workflow_route(workflow_id: str) -> dict[str, Any]:
        from rengu_flow_ui import workflow_runner

        with _workflow_http_errors():
            workflow_db.get_workflow(workflow_id)
            workflow_runner.cancel_workflow(workflow_id)
        workflow_runner.tick()
        return _workflow_detail(workflow_db.get_workflow(workflow_id))

    @app.get(f"{API_PREFIX}/workflows/{{workflow_id}}/nodes/{{node_id}}/log")
    def node_log_route(workflow_id: str, node_id: str, offset: int = 0) -> dict[str, Any]:
        with _workflow_http_errors():
            _require_node(workflow_id, node_id)
        path = workflow_db.node_dir(workflow_id, node_id) / "node.log"
        chunk, new_offset = jobs.tail_log_path(path, offset)
        raw_tail = jobs.read_raw_log_tail_path(path)
        progress = parse_last_progress_marker(raw_tail) if raw_tail else None
        return {"chunk": chunk, "offset": new_offset, "progress": progress}

    @app.websocket(f"{API_PREFIX}/workflows/{{workflow_id}}/nodes/{{node_id}}/log/ws")
    async def workflow_node_log_ws(websocket: WebSocket, workflow_id: str, node_id: str) -> None:
        token = ui_token()
        if token:
            # Check token from query param (?token=...) since WS headers are unreliable across
            # browsers — same convention as every other WS in this app.
            qs_token = websocket.query_params.get("token", "")
            if qs_token != token:
                await websocket.close(code=4401, reason="Invalid token")
                return
        await websocket.accept()

        async def _send_log(text: str) -> None:
            # Split into sub-frames so a multi-MB tail never trips the 1 MB WS frame limit.
            for frame in jobs.iter_log_frames(text):
                await websocket.send_text(frame)

        try:
            _require_node(workflow_id, node_id)
        except KeyError:
            await websocket.send_text("[error] workflow or node not found\n")
            await websocket.close()
            return

        path = workflow_db.node_dir(workflow_id, node_id) / "node.log"
        # Start near the end: new output is appended from this offset onward, mirroring
        # /jobs/{id}/logs/ws.
        offset = await asyncio.to_thread(jobs.log_tail_start_offset_path, path)
        try:
            while True:
                chunk, offset = await asyncio.to_thread(jobs.tail_log_path, path, offset)
                if chunk:
                    await _send_log(chunk)
                try:
                    state = await asyncio.to_thread(workflow_db.get_state, workflow_id)
                except KeyError:
                    await websocket.send_text("[error] workflow not found\n")
                    break
                node_state = (state.get("nodes") or {}).get(node_id) or {}
                if node_state.get("status") not in _NODE_ACTIVE_STATUSES:
                    # Terminal (or not yet started) — flush any trailing output and close,
                    # rather than tailing forever until the client disconnects.
                    chunk, offset = await asyncio.to_thread(jobs.tail_log_path, path, offset)
                    if chunk:
                        await _send_log(chunk)
                    break
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass

    @app.websocket(f"{API_PREFIX}/workflows/events/ws")
    async def workflow_events_ws(websocket: WebSocket) -> None:
        # Push a "workflows-changed" frame whenever any workflow row is written (content, state,
        # create, delete) so the list/editor refresh on demand instead of polling. Calqued on
        # /jobs/events/ws, watching workflow_db.workflows_version() instead of db.jobs_version().
        token = ui_token()
        if token:
            qs_token = websocket.query_params.get("token", "")
            if qs_token != token:
                await websocket.close(code=4401, reason="Invalid token")
                return
        await websocket.accept()
        last = -1
        try:
            while True:
                version = workflow_db.workflows_version()
                if version != last:
                    last = version
                    await websocket.send_text(
                        json.dumps({"type": "workflows-changed", "version": version})
                    )
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if message.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
