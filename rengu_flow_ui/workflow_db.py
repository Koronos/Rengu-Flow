"""SQLite persistence for Workflows (see docs/spec/workflows.md, "Persistence").

One row per workflow, two columns that are never written by the same caller:

* ``content`` — the graph JSON (nodes, variables, ``from``). Written only by the editor via
  :func:`update_graph`, under optimistic concurrency (a ``version`` column). This module never
  parses or validates ``content``; it is an opaque string to everything here — parsing the graph
  is ``workflow_graph``'s job, not this one's.
* ``state_json`` — the single saved run state (per-node status, output handle, exit code). Written
  only by :func:`mutate_state`, which does a compare-and-swap read-modify-write so the poller tick,
  ``/start`` and ``/cancel`` — three different threads — never lose each other's write.

Splitting those two into separate columns is the whole design: saving an edit to node 3 can never
clobber node 1-2's live ``done`` state, and a poller tick updating progress can never clobber an
edit in flight. A single JSON blob, or a single ``UPDATE ... SET state_json = ...`` without a
compare-and-swap, would let the two race and silently drop a write — and workflows are the one
object in this app with **no history** (see the spec's Non-goals), so a lost update here is
unrecoverable, unlike a `jobs` row.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rengu_flow_ui import db, settings
from rengu_flow_ui._time import now_utc_iso

# Compare-and-swap on `state_json` retries on conflict rather than failing; this bounds it so a
# genuinely stuck loop (a bug, not real contention) raises instead of hanging a request forever.
# Real contention is a handful of threads at most (poller tick, /start, /cancel), so this is orders
# of magnitude more headroom than any legitimate interleaving needs.
_MUTATE_STATE_MAX_ATTEMPTS = 10_000

# Monotonic counter bumped on every workflow row write (content, state, create, delete) — the
# workflows-events WebSocket watches it so clients refresh without polling. Calqued on
# `db._jobs_version` (db.py:14-27): a plain int suffices because writes happen under the GIL and
# readers tolerate a stale read for one tick.
_workflows_version = 0


def workflows_version() -> int:
    return _workflows_version


def _bump_workflows_version() -> None:
    global _workflows_version
    _workflows_version += 1


class StaleWorkflowError(Exception):
    """Raised by :func:`update_graph` when ``expected_version`` no longer matches the stored one.

    The HTTP layer maps this to 409. Without this check, two open tabs saving the same workflow
    would let whichever save lands second silently erase the node the first one added — and
    workflows are the only object in the app with no history, so that loss is irrecoverable.
    """

    def __init__(self, workflow_id: Any, expected_version: int, actual_version: int) -> None:
        self.workflow_id = workflow_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"workflow {workflow_id}: expected version {expected_version}, "
            f"found {actual_version}"
        )


@dataclass
class WorkflowRecord:
    id: int
    name: str
    content: str
    state_json: str
    version: int
    created_at: str
    updated_at: str


def _connect() -> sqlite3.Connection:
    """Same DB file and same pragmas (WAL, busy_timeout) as every other UI connection."""
    return db._connect()


def _coerce_workflow_id(workflow_id: str | int) -> int:
    if isinstance(workflow_id, bool):
        raise KeyError(workflow_id)
    if isinstance(workflow_id, int):
        return workflow_id
    s = str(workflow_id).strip()
    if not s.isdigit():
        raise KeyError(workflow_id)
    return int(s)


def _row_to_record(row: sqlite3.Row) -> WorkflowRecord:
    return WorkflowRecord(
        id=int(row["id"]),
        name=row["name"],
        content=row["content"],
        state_json=row["state_json"],
        version=int(row["version"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ------------------------------------------------------------------------------ CRUD


def create_workflow(name: str, content: str) -> WorkflowRecord:
    now = now_utc_iso()
    conn = _connect()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO workflows (name, content, state_json, version, created_at, updated_at)
                VALUES (?, ?, '{}', 0, ?, ?)
                """,
                (name or "", content, now, now),
            )
            wid = int(cur.lastrowid)
    finally:
        conn.close()
    _bump_workflows_version()
    return get_workflow(wid)


def get_workflow(workflow_id: str | int) -> WorkflowRecord:
    wid = _coerce_workflow_id(workflow_id)
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM workflows WHERE id = ?", (wid,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(workflow_id)
    return _row_to_record(row)


def list_workflows() -> list[WorkflowRecord]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM workflows ORDER BY id").fetchall()
    finally:
        conn.close()
    return [_row_to_record(r) for r in rows]


def delete_workflow(workflow_id: str | int) -> None:
    wid = _coerce_workflow_id(workflow_id)
    conn = _connect()
    try:
        with conn:
            cur = conn.execute("DELETE FROM workflows WHERE id = ?", (wid,))
    finally:
        conn.close()
    if cur.rowcount == 0:
        raise KeyError(workflow_id)
    _bump_workflows_version()


# ------------------------------------------------------------------------------ graph (content)


def update_graph(
    workflow_id: str | int, content: str, *, expected_version: int
) -> WorkflowRecord:
    """Optimistically-concurrent write of ``content``. Never touches ``state_json``.

    The ``UPDATE ... WHERE id = ? AND version = ?`` is the compare-and-swap: it either matches and
    lands atomically, or matches zero rows and changes nothing. On a mismatch we look the current
    version up to report it and raise :class:`StaleWorkflowError` — the caller (an open second tab)
    gets a 409 instead of silently winning a race against the first tab's edit.
    """
    wid = _coerce_workflow_id(workflow_id)
    conn = _connect()
    try:
        with conn:
            cur = conn.execute(
                """
                UPDATE workflows
                SET content = ?, version = version + 1, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (content, now_utc_iso(), wid, int(expected_version)),
            )
            if cur.rowcount == 0:
                row = conn.execute(
                    "SELECT version FROM workflows WHERE id = ?", (wid,)
                ).fetchone()
                if row is None:
                    raise KeyError(workflow_id)
                raise StaleWorkflowError(workflow_id, int(expected_version), int(row["version"]))
    finally:
        conn.close()
    _bump_workflows_version()
    return get_workflow(wid)


def clone_workflow(workflow_id: str | int, *, name: str | None = None) -> WorkflowRecord:
    """Copy ``content`` into a new row; discard ``state_json`` and reset ``version`` to 0.

    Precedent: ``library_db.duplicate_dataset``. A clone must never be born already "done" with
    another workflow's outputs, so it always goes through :func:`create_workflow`, which is the
    only path that starts ``state_json`` at ``'{}'`` and ``version`` at ``0``.
    """
    row = get_workflow(workflow_id)
    if name is not None:
        new_name = name
    else:
        new_name = f"{row.name} (copy)" if row.name else ""
    return create_workflow(new_name, row.content)


# ------------------------------------------------------------------------------ state (state_json)


def get_state(workflow_id: str | int) -> dict:
    raw = get_workflow(workflow_id).state_json
    return json.loads(raw) if raw else {}


def mutate_state(
    workflow_id: str | int, fn: Callable[[dict], dict | None]
) -> dict:
    """The only writer of ``state_json``. Read-modify-write with compare-and-swap, retried.

    ``fn`` receives the current state dict — it may mutate it in place and return ``None``, or
    return a brand-new dict. Between the read and the write another thread may have written its own
    new state (the poller tick, ``/start``, ``/cancel`` all call this from different threads); the
    ``WHERE state_json = ?`` clause on the *previously read* raw value detects that and the whole
    read-apply-write is retried against the fresh value, so no writer's update is silently dropped.
    A plain read-modify-write (no CAS) would let whichever write lands second win outright — e.g.
    the ``cancelling`` status a user just requested overwritten by a progress tick that read the
    old state first but wrote last.
    """
    wid = _coerce_workflow_id(workflow_id)
    for _ in range(_MUTATE_STATE_MAX_ATTEMPTS):
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT state_json FROM workflows WHERE id = ?", (wid,)
            ).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            old_raw = row["state_json"]
            state = json.loads(old_raw) if old_raw else {}
            result = fn(state)
            new_state = state if result is None else result
            new_raw = json.dumps(new_state)
            with conn:
                cur = conn.execute(
                    """
                    UPDATE workflows
                    SET state_json = ?, updated_at = ?
                    WHERE id = ? AND state_json = ?
                    """,
                    (new_raw, now_utc_iso(), wid, old_raw),
                )
            if cur.rowcount > 0:
                _bump_workflows_version()
                return new_state
        finally:
            conn.close()
    raise RuntimeError(
        f"mutate_state: gave up after {_MUTATE_STATE_MAX_ATTEMPTS} attempts on workflow {workflow_id}"
    )


# ------------------------------------------------------------------------------ node directory


def node_dir(workflow_id: str | int, node_id: str) -> Path:
    """``ui_data_dir()/workflows/<workflow_id>/<node_id>``, guarded against path traversal.

    A function, not a literal repeated at every call site: the spec leaves the door open to
    on-disk history (``<node_id>.<timestamp>``) later, and this keeps that a one-line change.

    The guard mirrors ``toolbox.tool_dir``: resolve both the base and the candidate, then require
    the base to be a genuine ancestor of the candidate. ``node_id`` values like ``".."`` or an
    absolute path (which ``Path.__truediv__`` lets override the whole base on both POSIX and
    Windows) resolve outside ``base`` and are rejected here rather than silently reading or writing
    outside the workflow's own directory.
    """
    base = (settings.ui_data_dir() / "workflows" / str(workflow_id)).resolve()
    p = (base / node_id).resolve()
    if p == base or base not in p.parents:
        raise KeyError(node_id)
    return p
