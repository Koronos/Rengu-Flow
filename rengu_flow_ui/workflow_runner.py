"""The workflow lane: one workflow, one node at a time (docs/spec/workflows.md, "Execution model").

This is the scheduler. ``workflow_graph`` owns the model (shapes, variables, the output rule),
``workflow_nodes`` owns the mechanics of a single node (argv, env, what it emitted, how it exited),
``workflow_db`` owns persistence — and this module owns *when*: which node runs next, who holds the
GPU, what a dead process means, and how a Stop escalates.

```
pending -> waiting_gpu -> launching -> running -> done | failed
                                          `----> stopping -> stopped
pending -> skipped (disabled)
```

The rules below are load-bearing. Each one exists because its absence is silent, not loud:

* **The tick takes a NON-BLOCKING lock.** FastAPI runs sync endpoints in a threadpool, so two
  ``/start`` calls (a double click, or two tabs) run concurrently and the poller is a third
  thread. Two threads would both see the same node ``pending`` and both spawn it; a CPU-only node
  takes no lease, so nothing else serializes them — two ``prep.quality`` processes would move files
  into ``low_quality/`` at the same time and the second ``pid`` write would orphan the first
  process forever. Mirrors ``job_queue._start_lock``.
* **``launching`` is written BEFORE the spawn.** A server killed in that window would otherwise
  leave the node ``pending`` while its process — detached by design — keeps running; on restart the
  runner would launch a *second* ``prep.tag`` and two taggers would write line 1 of the same
  sidecars concurrently, interleaving captions across the dataset with neither aware. A
  ``launching`` node is therefore **never** auto-started by :func:`reconcile_on_start`: it is failed
  with :data:`LAUNCH_INTERRUPTED_ERROR`, or adopted when its ``node.log`` is still growing.
* **``node.log`` is truncated per run**, never appended to. A workflow keeps one state and no
  history, so the previous run's tail buys nothing — and it costs the exit code: ``read_exit_code``
  returns the last marker *in the file*, so a second run killed hard (OOM, power) before printing
  its own would be certified ``done`` by run 1's ``= 0``, handle and all. That is every re-run of
  every node, which is the normal case. See :func:`_spawn`.
* **Prep extras are installed here, not by the child.** ``rengu prep <stage>`` calls
  ``ensure_profiles(["prep"])`` -> ``uv sync --inexact --extra prep``. Today that can never overlap a
  training run because the shared queue prevents it — the very guarantee this feature removes. A
  first-ever CPU-only ``prep.quality`` "running alongside training" would rewrite ``site-packages``
  under a live DeepSpeed process and hand it an ``ImportError`` hours later, at checkpoint time. So
  the install runs in the UI process, with the node in ``launching``, and is **refused while the
  training lane is busy** — which is read from the GPU lease, not from the job rows: the lease is
  taken *before* ``ensure_training_extras``, and it is exactly that multi-minute ``uv sync`` window
  in which the job is still ``pending`` (see :func:`_training_lane_busy`).
* **Reconciliation matches ``pid`` AND ``pid_create_time``.** A workflow can sit in ``running`` for
  weeks (the user shuts the machine down mid-prep). After a reboot PID 4231 is an unrelated process;
  without the create-time test the node stays "running" forever **and its GPU lease is never
  released**, permanently blocking the training lane. Zombies count as dead: ``pid_alive`` is
  ``os.kill(pid, 0)`` on POSIX and returns ``True`` for one, and nothing ever ``wait()``s these
  detached children (precedent: ``gpu_lease._pid_is_gone``).
* **Cancellation is two-phase**, because a tick comes back to look again: ``SIGNAL_QUIT`` in the node
  dir, a 20 s grace period, then ``terminate_process_tree``. Prep stages check the signal between
  batches (``runner._make_should_stop``), so a half-done ``tag`` finishes its batch and writes
  ``report.json`` instead of losing the work. Tool nodes have no signal contract by decision and are
  terminated outright.
* **exit 0 with ``report["stopped"] == true`` is ``stopped``, never ``done``.** ``run_stage`` returns
  0 when ``should_stop()`` fired (``report["stopped"]`` is a field, not an exit code). Mapping it to
  ``done`` and propagating the handle would let a caption run stopped at 60 % be treated as complete,
  skipping the remaining 40 % and training on a half-captioned dataset.
* **``saved_input`` is written next to ``output`` and ``config_hash``.**
  ``workflow_graph.compute_stale`` compares it; without it every input comparison is
  ``None != handle`` and every node reads stale forever.
"""

from __future__ import annotations

import json
import logging
import shlex
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from rengu_flow.platform_compat import pid_alive, terminate_process_tree
from rengu_flow_ui import gpu_lease, settings, workflow_db, workflow_nodes
from rengu_flow_ui._time import now_utc_iso
from rengu_flow_ui.subprocess_util import popen_repo_subprocess
from rengu_flow_ui.workflow_graph import (
    DatasetHandle,
    NodeOutputError,
    WorkflowGraph,
    WorkflowNode,
    compute_stale,
    execution_order,
    node_config_hash,
    parse_graph,
    resolve_config,
    validate,
)
from rengu_flow_ui.workflow_nodes import INLINE_TYPES

_logger = logging.getLogger("rengu_flow_ui.workflow_runner")

#: Non-blocking, module level, exactly like ``job_queue._start_lock``. See the module docstring.
_tick_lock = threading.Lock()

#: "No parallel DAG execution. One node at a time, one active workflow at a time." (spec, Non-goals)
MAX_ACTIVE_WORKFLOWS = 1

#: Grace between dropping ``SIGNAL_QUIT`` and terminating the process tree. Prep stages poll the
#: signal between batches/chunks, so this is what buys a half-done ``tag`` its ``report.json``.
CANCEL_GRACE_SECONDS = 20.0

#: A ``launching`` node whose process is unaccounted for. Never auto-started — see the docstring.
LAUNCH_INTERRUPTED_ERROR = "Interrupted while starting; check for an orphan process"

#: ``node.log`` said nothing about how the child exited. ``workflow_nodes.read_exit_code`` keeps
#: "unknown" as ``None`` precisely so this stays a decision, and the decision is *not* ``done``:
#: marking a node complete and propagating its handle on no evidence at all is how a workflow walks
#: past a crashed step.
EXIT_UNKNOWN_ERROR = (
    "This step did not report an exit code in node.log, so it cannot be treated as complete."
)

#: Node statuses that mean "a process may be out there". Anything here is reconciled, never skipped.
_ACTIVE_NODE_STATUSES = ("launching", "running", "stopping")

#: Node statuses the scheduler is allowed to (re-)launch from.
_RUNNABLE_NODE_STATUSES = ("pending", "waiting_gpu")

#: Workflow statuses that make a workflow a candidate for :func:`_advance`.
_ACTIVE_WORKFLOW_STATUSES = ("running", "cancelling")

#: How long to watch a ``launching`` node's log at reconcile time before deciding it is not growing.
_ADOPTION_SAMPLE_SECONDS = 1.0

#: An adopted node has no pid to supervise (its ``launching`` state predates the pid write), so its
#: liveness is log growth. This is how long a silent log is tolerated before it counts as dead.
_ADOPTED_STALL_SECONDS = 120.0

#: ①..⑳ — the same glyphs the editor numbers cards with, so an error names what the user sees.
_POSITION_GLYPHS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


class _WaitingForLane(Exception):
    """Not an error: the node cannot start *yet*. It goes back to ``waiting_gpu`` and retries."""


# ------------------------------------------------------------------------------ small helpers


def _holder_id(workflow_id: Any, node_id: str) -> str:
    return f"wf:{workflow_id}:{node_id}"


def _error_text(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


def _graph_of(record: workflow_db.WorkflowRecord) -> WorkflowGraph:
    try:
        data = json.loads(record.content or "{}")
    except ValueError:
        _logger.warning("workflow %s: content is not valid JSON", record.id)
        return WorkflowGraph()
    return parse_graph(data if isinstance(data, Mapping) else {})


def _state_of(record: workflow_db.WorkflowRecord) -> dict:
    try:
        state = json.loads(record.state_json or "{}")
    except ValueError:
        return {}
    return state if isinstance(state, dict) else {}


def _node_state(state: Mapping[str, Any], node_id: str) -> dict:
    info = (state.get("nodes") or {}).get(node_id)
    return dict(info) if isinstance(info, Mapping) else {}


def _find_node(graph: WorkflowGraph, node_id: str) -> WorkflowNode | None:
    for node in graph.nodes:
        if node.id == node_id:
            return node
    return None


def _position_label(index: int) -> str:
    if 0 <= index < len(_POSITION_GLYPHS):
        return _POSITION_GLYPHS[index]
    return f"#{index + 1}"


def _label(graph: WorkflowGraph, node_id: str) -> str:
    ids = [n.id for n in graph.nodes]
    return _position_label(ids.index(node_id)) if node_id in ids else node_id


def _config_hashes(graph: WorkflowGraph) -> dict[str, str]:
    """Every node's current hash. ``from`` points backwards, so one forward pass is enough."""
    hashes: dict[str, str] = {}
    for node in graph.nodes:
        parent = hashes.get(node.source, "") if node.source else ""
        hashes[node.id] = node_config_hash(node, parent, graph.variables)
    return hashes


def _resolved(node: WorkflowNode, graph: WorkflowGraph) -> WorkflowNode:
    """A copy whose config has variables substituted — everything downstream reads it literally."""
    return replace(node, config=resolve_config(node, graph.variables))


def _input_handle(state: Mapping[str, Any], source_id: str | None) -> DatasetHandle | None:
    """The handle a node consumes: its ``from`` node's **saved, resolved** output.

    Stored resolved rather than as a reference, which is exactly what makes "start from step 4"
    possible without re-running steps 1-3. A disabled ``from`` node is not special-cased: its saved
    handle is read the same way, and its absence is what fails the node.
    """
    if not source_id:
        return None
    output = _node_state(state, source_id).get("output")
    if not isinstance(output, Mapping) or not output.get("path"):
        return None
    extra = {
        key: str(output[key])
        for key in ("caption_format", "caption_ext")
        if output.get(key)
    }
    return DatasetHandle(path=str(output["path"]), **extra)


# ------------------------------------------------------------------------------ state writes


def _update_node(workflow_id: Any, node_id: str, **fields: Any) -> None:
    def _apply(state: dict) -> None:
        nodes = state.setdefault("nodes", {})
        info = nodes.setdefault(node_id, {})
        info.update(fields)

    workflow_db.mutate_state(workflow_id, _apply)


def _update_workflow(workflow_id: Any, **fields: Any) -> None:
    def _apply(state: dict) -> None:
        state.update(fields)

    workflow_db.mutate_state(workflow_id, _apply)


def _mark_launching(workflow_id: Any, node_id: str) -> None:
    """One transaction: the node is ``launching`` and the workflow points at it — before any spawn.

    Both halves must land together. A ``launching`` node nobody points at, or a ``current_node``
    whose row still reads ``pending``, is exactly the ambiguity that lets a restart start a second
    process for work that is already running.
    """
    def _apply(state: dict) -> None:
        state["current_node"] = node_id
        info = state.setdefault("nodes", {}).setdefault(node_id, {})
        info.update(
            {
                "status": "launching",
                "started_at": now_utc_iso(),
                "finished_at": None,
                "exit_code": None,
                "error": "",
                "pid": None,
                "pid_create_time": None,
                "adopted": False,
                "stop_requested_at": None,
            }
        )

    workflow_db.mutate_state(workflow_id, _apply)


def _fail_node(workflow_id: Any, node_id: str, error: str, *, exit_code: int | None = None) -> str:
    # The saved output is cleared: a step that failed did not produce one, and leaving the previous
    # run's handle behind would let a downstream node consume a folder this run never finished.
    _update_node(
        workflow_id,
        node_id,
        status="failed",
        error=error,
        exit_code=exit_code,
        finished_at=now_utc_iso(),
        pid=None,
        output=None,
    )
    return "failed"


def _stop_node(workflow_id: Any, node_id: str, exit_code: int | None) -> str:
    # Same reasoning as _fail_node, and this is the case the spec calls out by name: a caption run
    # stopped at 60 % must NOT hand its folder on as a completed step.
    _update_node(
        workflow_id,
        node_id,
        status="stopped",
        exit_code=exit_code,
        finished_at=now_utc_iso(),
        pid=None,
        output=None,
    )
    return "stopped"


def _complete_node(
    workflow_id: Any,
    graph: WorkflowGraph,
    node: WorkflowNode,
    output: DatasetHandle | None,
    inputs: DatasetHandle | None,
    *,
    exit_code: int | None = 0,
    result: Any = None,
) -> str:
    """Record a finished node: its handle, the input it consumed, and the hash that produced them.

    ``saved_input`` is not decoration — ``compute_stale`` folds it in to catch what the config chain
    cannot see (a tool returning a computed folder leaves its downstream node's config textually
    unchanged), and omitting it makes every comparison ``None != handle``.
    """
    fields: dict[str, Any] = {
        "status": "done",
        "finished_at": now_utc_iso(),
        "exit_code": exit_code,
        "error": "",
        "pid": None,
        "output": output.to_dict() if output is not None else None,
        "saved_input": inputs.to_dict() if inputs is not None else None,
        "config_hash": _config_hashes(graph).get(node.id, ""),
    }
    if result is not None:
        fields["result"] = result
    _update_node(workflow_id, node.id, **fields)
    return "done"


def _finish_workflow(workflow_id: Any, status: str) -> None:
    _update_workflow(
        workflow_id, status=status, current_node=None, finished_at=now_utc_iso()
    )


# ------------------------------------------------------------------------------ process liveness


def _pid_create_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:  # noqa: BLE001 - psutil missing, already gone, access denied
        return None


def _pid_is_alive(pid: int, create_time: float | None) -> bool:
    """``pid`` still belongs to *the same* process we launched.

    The create-time comparison is the whole point: a node can sit in ``running`` across a reboot,
    after which the number identifies something else entirely — and a node that never finalizes
    never releases its GPU lease, which is the only deadlock this system can have.
    """
    if pid is None or not pid_alive(int(pid)):
        return False
    try:
        import psutil
    except ImportError:
        return True  # cannot refine; pid_alive said it is there
    try:
        proc = psutil.Process(int(pid))
        # A zombie answers os.kill(pid, 0) on POSIX and nothing wait()s these detached children.
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None:
            return abs(proc.create_time() - float(create_time)) < 1.0
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True  # access denied and friends: assume alive rather than double-spawn
    return True


def _log_size(workflow_id: Any, node_id: str) -> int:
    path = workflow_nodes.node_log_path(workflow_db.node_dir(workflow_id, node_id))
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _adopted_is_alive(workflow_id: Any, node: WorkflowNode, info: Mapping[str, Any]) -> bool:
    """Liveness for a node adopted at reconcile time, which has no pid: is its log still moving?"""
    size = _log_size(workflow_id, node.id)
    if size != info.get("log_size"):
        _update_node(workflow_id, node.id, log_size=size, log_seen_at=time.time())
        return True
    seen = info.get("log_seen_at")
    return seen is not None and (time.time() - float(seen)) < _ADOPTED_STALL_SECONDS


def _node_is_alive(workflow_id: Any, node: WorkflowNode, info: Mapping[str, Any]) -> bool:
    if node.type in INLINE_TYPES:
        return False  # inline nodes finish inside _launch_node; an active status is leftover
    pid = info.get("pid")
    if pid is not None:
        return _pid_is_alive(int(pid), info.get("pid_create_time"))
    if info.get("adopted"):
        return _adopted_is_alive(workflow_id, node, info)
    return False


# ------------------------------------------------------------------------------ launching


def _spawn(
    launch: workflow_nodes.NodeLaunch, node_dir: Path, workflow_id: Any, node_id: str
) -> tuple[int, float | None]:
    """The one place a node process is created. Detached by design, so it survives a restart."""
    header = (
        f"--- rengu-flow-ui workflow {workflow_id} node {node_id} ---\n"
        f"CWD: {settings.repo_root()}\n"
        f"CMD: {shlex.join(launch.argv)}\n\n"
    ).encode()
    proc, _log_f = popen_repo_subprocess(
        launch.argv,
        workflow_nodes.node_log_path(node_dir),
        # "wb", NOT the default append: `read_exit_code` reads the last marker in the FILE, so an
        # appended log lets run 1's `= 0` certify run 2's crash as `done`. A node keeps one state,
        # not a history, so there is nothing to preserve. See the module docstring.
        log_mode="wb",
        log_header=header,
        env=launch.env,
    )
    return proc.pid, _pid_create_time(proc.pid)


def _check_toolbox_gate(node: WorkflowNode) -> None:
    """``[toolbox].enabled`` gates workflow tool nodes exactly as it gates ``/toolbox``.

    That switch exists because running a tool executes arbitrary user Python, and it defaults to
    ``false``. A node runs the same code by the same mechanism, so exempting it would turn Workflows
    into a way around the gate.
    """
    if node.type != "tool":
        return
    from rengu_flow.config import local_config
    from rengu_flow_ui import toolbox

    if not local_config.toolbox_enabled():
        raise toolbox.ExecutionDisabledError(
            "Execution disabled in rengu.local.toml -> [toolbox].enabled"
        )


def _training_lane_busy() -> bool:
    """Is a training run underway *or* being started? The lease is what answers, not the job row.

    ``job_queue.try_start_next`` acquires the lease and only **then** calls ``jobs.start_job`` ->
    ``ensure_training_extras`` -> ``uv sync``, which takes minutes. For that whole window the row is
    still ``pending``, so ``has_active_runner()`` alone says "idle" and lets a prep node rewrite
    ``site-packages`` under a training process that is mid-launch — precisely the hazard this
    refusal exists for. The lease covers it because it is taken before the sync.

    ``has_active_runner()`` is still asked afterwards: a run whose lease was reaped (a pid that
    outlived its lease row) is running all the same, and the job rows are the only witness left.
    """
    if any(row.get("holder_kind") == "train" for row in gpu_lease.snapshot()):
        return True
    from rengu_flow_ui import job_queue

    return job_queue.has_active_runner()


def _install_prep_extras(node: WorkflowNode) -> None:
    """``uv sync --extra prep``, here rather than inside the child. See the module docstring.

    The refusal is conditioned on there being something to install: with the extras already present
    ``ensure_profiles`` writes nothing, there is no ``site-packages`` rewrite, and refusing anyway
    would block every CPU-only prep node for the whole of any training run — which is the lane
    separation this feature is *for*. The check and the install happen together so no run can start
    in between.
    """
    if not node.type.startswith("prep."):
        return
    from rengu_flow.install.manager import ensure_profiles, missing_profiles

    if not missing_profiles(["prep"]):
        return

    if _training_lane_busy():
        raise _WaitingForLane(
            "Prep extras are not installed yet, and installing them rewrites site-packages "
            "under the running training job. Waiting for the training queue to be idle."
        )
    ensure_profiles(["prep"], root=settings.repo_root(), reason="dataset prep")


def _sweep_signal_files(node_dir: Path) -> None:
    """Clear stop signals left by a previous cancel, exactly as ``prep_jobs.requeue_prep_job`` does.

    Without this, re-running a cancelled node exits immediately: the stage reads the stale ``quit``
    on its first batch and reports itself stopped before doing any work.

    Imported lazily on purpose — ``rengu_flow/utils/signal_files.py`` imports ``torch`` at module
    scope, and paying a multi-second import at *this* module's import time would put it inside the
    poller thread's first pass, stalling training reconciliation with it.
    """
    from rengu_flow.utils.signal_files import SIGNAL_QUIT, SIGNAL_SAVE_QUIT

    for name in (SIGNAL_SAVE_QUIT, SIGNAL_QUIT):
        try:
            (node_dir / name).unlink(missing_ok=True)
        except OSError:  # a locked or read-only file: the stage will simply stop early
            _logger.warning("could not clear signal file %s", node_dir / name)


def _write_quit_signal(node_dir: Path) -> None:
    from rengu_flow.utils.signal_files import SIGNAL_QUIT

    try:
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / SIGNAL_QUIT).touch()
    except OSError:
        _logger.warning("could not write the quit signal in %s", node_dir)


def _run_inline_node(
    workflow_id: Any,
    graph: WorkflowGraph,
    node: WorkflowNode,
    resolved: WorkflowNode,
    inputs: DatasetHandle | None,
    node_dir: Path,
) -> str:
    """``folder`` and ``train``: no argv, no env, no lease, no log — they run in this process."""
    _mark_launching(workflow_id, node.id)
    # Settle the claim BEFORE run_inline: `train` bumps unconditionally *and starts the queue*, so
    # the answer to "was somebody else already at the front?" is only meaningful — and only
    # actionable — before the bump lands.
    claimed = _claim_before_bump(resolved) if node.type == "train" else None
    try:
        result = workflow_nodes.run_inline(resolved, inputs, node_dir)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 - a bad node is a node error, not a dead tick
        return _fail_node(workflow_id, node.id, _error_text(exc))
    if node.type == "train":
        _settle_queue_claim(workflow_id, claimed, result)
    try:
        output = workflow_nodes.collect_output(resolved, node_dir, inputs)
    except NodeOutputError as exc:
        return _fail_node(workflow_id, node.id, str(exc))
    return _complete_node(workflow_id, graph, node, output, inputs, result=result)


def _launch_node(
    workflow_id: Any, graph: WorkflowGraph, node: WorkflowNode, state: Mapping[str, Any]
) -> str:
    """Take one node from runnable to running (or to a terminal status). Returns that status."""
    node_id = node.id
    resolved = _resolved(node, graph)
    inputs = _input_handle(state, node.source)
    node_dir = workflow_db.node_dir(workflow_id, node_id)

    if node.type in INLINE_TYPES:
        return _run_inline_node(workflow_id, graph, node, resolved, inputs, node_dir)

    try:
        _check_toolbox_gate(node)
    except Exception as exc:  # noqa: BLE001 - the gate's message is the node's error
        _update_workflow(workflow_id, current_node=node_id)
        return _fail_node(workflow_id, node_id, _error_text(exc))

    # The lease policy lives in workflow_nodes so `wait: false` (the explicit "share the GPU"
    # escape hatch) has exactly ONE definition; it is needed here, before build_launch, because the
    # acquire decides whether this node runs at all.
    holder = _holder_id(workflow_id, node_id)
    needs_lease = workflow_nodes._needs_lease(node)
    devices = workflow_nodes._devices(node)
    if needs_lease and not gpu_lease.acquire("workflow", holder, devices):
        _update_workflow(workflow_id, current_node=node_id)
        _update_node(
            workflow_id,
            node_id,
            status="waiting_gpu",
            error=gpu_lease.wait_reason(devices) or "Waiting for the GPU.",
        )
        return "waiting_gpu"

    try:
        # BEFORE the spawn, never after. See the module docstring.
        _mark_launching(workflow_id, node_id)
        _install_prep_extras(node)
        _sweep_signal_files(node_dir)
        launch = workflow_nodes.build_launch(resolved, inputs, node_dir)
        if launch is None:  # a non-inline type with no launcher is a bug, not a node error
            raise ValueError(f"Node type {node.type!r} produced no launch")
        pid, create_time = _spawn(launch, node_dir, workflow_id, node_id)
    except _WaitingForLane as exc:
        gpu_lease.release(holder)
        _update_node(workflow_id, node_id, status="waiting_gpu", error=str(exc))
        return "waiting_gpu"
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 - ensure_profiles raises SystemExit
        gpu_lease.release(holder)
        _logger.exception("workflow %s node %s failed to launch", workflow_id, node_id)
        return _fail_node(workflow_id, node_id, _error_text(exc))

    _update_node(
        workflow_id,
        node_id,
        status="running",
        pid=int(pid),
        pid_create_time=create_time,
        error="",
    )
    if needs_lease and gpu_lease.bind_pid(holder, pid, create_time) is False:
        # The lease was reaped out from under the launch, so another holder may already own the
        # GPU. Kill what we just spawned — same pattern as job_queue.try_start_next. `is False`,
        # not falsy: None means there was simply no pid to bind.
        terminate_process_tree(pid)
        gpu_lease.release(holder)
        return _fail_node(
            workflow_id,
            node_id,
            "The GPU lease was released while this step was starting; the process was killed.",
        )
    return "running"


# ------------------------------------------------------------------------------ the train claim


def _live_queue_claim() -> Any | None:
    """The job id of the one front-of-queue claim currently outstanding, if any.

    ``job_queue.bump_pending_after`` displaces unconditionally, so two ``train`` nodes would reorder
    each other's turn indefinitely. A claim stays live only while its run is still ``pending``: once
    the run starts (or is deleted) it stops holding the front of the queue and the next workflow may
    claim it.
    """
    from rengu_flow_ui import db

    for record in workflow_db.list_workflows():
        claim = _state_of(record).get("queue_claim")
        if not isinstance(claim, Mapping):
            continue
        job_id = claim.get("job_id")
        if job_id is None:
            continue
        try:
            job = db.get_job(job_id)
        except KeyError:
            continue
        if job.state == "pending":
            return job_id
    return None


def _claim_before_bump(resolved: WorkflowNode) -> Any | None:
    """The standing front-of-queue claim, *after* giving it its turn. ``train`` nodes only.

    ``workflow_nodes._run_train`` bumps our run to position 0 and calls ``try_start_next`` in the
    same breath, so a claim refused afterwards is refused too late: the queue has already started
    **our** run from the front, which is the displacement the claim exists to prevent (spec,
    "Queueing semantics"). It is not a narrow window either — in Phase 0 nothing drains the queue
    periodically, so an idle queue is the normal state when a second ``train`` node runs.

    Draining first is what makes that inner ``try_start_next`` harmless: either the claimant starts
    (and ``has_active_runner()`` closes the door behind it) or nothing could have started at all,
    because the second call meets the very same gates. The claim is then re-read, since a claim
    whose run has started no longer holds the front and this node may take it.
    """
    claimed = _live_queue_claim()
    if claimed is None or str(claimed) == str(resolved.config.get("job_id", "")):
        # Our own standing claim. Draining here would start the run behind the node's back, and
        # `_run_train` refuses a job that is no longer `pending` — a re-run would fail instead.
        return claimed
    from rengu_flow_ui import job_queue

    try:
        job_queue.try_start_next()
    except Exception:  # noqa: BLE001 - degrade to "the claim still stands", never lose the node
        # Same posture as try_start_next's own guard around reap_dead: a transient DB failure here
        # would otherwise abort the launch with the node stuck in `launching` until a reconcile.
        _logger.exception("could not drain the queue before the train bump; keeping the claim")
        return claimed
    return _live_queue_claim()


def _settle_queue_claim(workflow_id: Any, claimed: Any | None, result: Any) -> None:
    """Register this run's claim, or put the earlier claimant back in front of it.

    ``workflow_nodes._run_train`` has already bumped by the time we get here, so refusing the second
    bump means restoring the first: ``bump_pending_after`` renumbers the rest by queue position, so
    the run we just displaced returns to 0 and ours lands immediately behind it — behind, in bump
    order, rather than displacing it.
    """
    job_id = result.get("job_id") if isinstance(result, Mapping) else None
    if job_id is None:
        return
    if claimed is None or str(claimed) == str(job_id):
        _update_workflow(workflow_id, queue_claim={"job_id": job_id})
        return
    from rengu_flow_ui import job_queue

    job_queue.bump_pending_after(claimed)


# ------------------------------------------------------------------------------ finalizing


def _read_report(node: WorkflowNode, node_dir: Path) -> Any:
    if not node.type.startswith("prep."):
        return None
    path = node_dir / workflow_nodes.REPORT_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _finalize_node(
    workflow_id: Any, graph: WorkflowGraph, node: WorkflowNode, state: Mapping[str, Any]
) -> str:
    """Turn a dead process into a verdict, and hand the GPU back no matter which verdict it is."""
    node_id = node.id
    node_dir = workflow_db.node_dir(workflow_id, node_id)
    info = _node_state(state, node_id)
    resolved = _resolved(node, graph)
    inputs = _input_handle(state, node.source)
    try:
        if info.get("status") == "launching" and info.get("pid") is None:
            return _fail_node(workflow_id, node_id, LAUNCH_INTERRUPTED_ERROR)

        exit_code = workflow_nodes.read_exit_code(resolved, node_dir)
        report = _read_report(node, node_dir)
        stopped = isinstance(report, Mapping) and bool(report.get("stopped"))

        # run_stage returns 0 when should_stop() fired: report["stopped"] is a FIELD, not an exit
        # code. Calling that `done` and propagating the handle is how a caption run stopped at 60 %
        # ends up training a half-captioned dataset.
        if info.get("status") == "stopping" or (exit_code == 0 and stopped):
            return _stop_node(workflow_id, node_id, exit_code)
        if exit_code is None:
            return _fail_node(workflow_id, node_id, EXIT_UNKNOWN_ERROR)
        if exit_code != 0:
            return _fail_node(
                workflow_id, node_id, f"Exited with code {exit_code}.", exit_code=exit_code
            )
        try:
            output = workflow_nodes.collect_output(resolved, node_dir, inputs)
        except NodeOutputError as exc:
            return _fail_node(workflow_id, node_id, str(exc), exit_code=exit_code)
        return _complete_node(
            workflow_id, graph, node, output, inputs, exit_code=exit_code
        )
    finally:
        # Under `finally` on purpose: every verdict, including the ones that raise on the way out,
        # must hand the GPU back. A node that keeps its lease blocks the training lane forever.
        gpu_lease.release(_holder_id(workflow_id, node_id))


# ------------------------------------------------------------------------------ cancellation


def _escalate_cancel(workflow_id: Any, node: WorkflowNode, info: Mapping[str, Any]) -> None:
    """One step of the two-phase stop. A tick comes back, so this is allowed to be patient."""
    if node.type in INLINE_TYPES:
        return  # nothing to stop: a queued run is stopped from the training queue
    node_dir = workflow_db.node_dir(workflow_id, node.id)
    pid = info.get("pid")

    if info.get("status") != "stopping":
        if node.type == "tool":
            # Tool nodes have no signal contract, by decision: no complexity is added to the tool
            # model. A cancelled tool is killed; if it needs cleanup it uses try/finally.
            if pid is not None:
                terminate_process_tree(int(pid))
        else:
            _write_quit_signal(node_dir)
        _update_node(
            workflow_id, node.id, status="stopping", stop_requested_at=time.time()
        )
        return

    requested = info.get("stop_requested_at")
    if requested is None:
        _update_node(workflow_id, node.id, stop_requested_at=time.time())
        return
    if time.time() - float(requested) >= CANCEL_GRACE_SECONDS and pid is not None:
        terminate_process_tree(int(pid))


# ------------------------------------------------------------------------------ scheduling


def _abandon_deleted_node(workflow_id: Any, node_id: str, info: Mapping[str, Any]) -> None:
    """Clean up after a node the editor deleted while it was running.

    ``PUT`` is supposed to 409 in this state, so this is the belt to that suspenders — but the old
    behaviour (release the lease, leave everything else) left the worst of both: a live, detached
    ``prep.tag`` still writing sidecars with nothing left in the graph that could ever finalize it,
    and a state row whose card kept saying "running" under a workflow reported ``done``. The
    process the user deleted is killed, and the orphan row goes with it.
    """
    pid = info.get("pid")
    if pid is not None and _pid_is_alive(int(pid), info.get("pid_create_time")):
        terminate_process_tree(int(pid))
    gpu_lease.release(_holder_id(workflow_id, node_id))

    def _apply(state: dict) -> None:
        (state.get("nodes") or {}).pop(node_id, None)
        if state.get("current_node") == node_id:
            state["current_node"] = None

    workflow_db.mutate_state(workflow_id, _apply)


def _stop_waiting_nodes(workflow_id: Any, state: Mapping[str, Any]) -> None:
    """A cancel must not leave a node reading ``waiting_gpu`` under a stopped workflow.

    ``waiting_gpu`` is not an active status, so nothing above finalizes it and the card would keep
    showing "Waiting for the GPU." forever, under a workflow that is never going to run it.

    The saved ``output`` is deliberately kept, unlike :func:`_stop_node`: this node never started,
    so whatever an earlier run left behind is exactly as valid as it was a moment ago, and clearing
    it would break "run from here" on the very next attempt.
    """
    for node_id, info in (state.get("nodes") or {}).items():
        if isinstance(info, Mapping) and info.get("status") == "waiting_gpu":
            _update_node(
                workflow_id,
                node_id,
                status="stopped",
                finished_at=now_utc_iso(),
                error="",
                pid=None,
            )


def _next_runnable(
    graph: WorkflowGraph, state: Mapping[str, Any]
) -> tuple[WorkflowNode | None, str | None]:
    """``(node to run, halting status)`` in list order — that is the whole scheduler.

    A ``failed`` or ``stopped`` node halts the workflow instead of being stepped over: the chain's
    whole premise is that each node's folder is the next node's input.
    """
    nodes = state.get("nodes") or {}
    for node in execution_order(graph):
        status = (nodes.get(node.id) or {}).get("status")
        if status in _RUNNABLE_NODE_STATUSES:
            return node, None
        if status in ("failed", "stopped"):
            return None, status
    return None, None


def _advance(workflow_id: Any) -> None:
    try:
        record = workflow_db.get_workflow(workflow_id)
    except KeyError:
        return
    graph = _graph_of(record)
    state = _state_of(record)
    if state.get("status") not in _ACTIVE_WORKFLOW_STATUSES:
        return

    current = state.get("current_node")
    if current:
        node = _find_node(graph, current)
        info = _node_state(state, current)
        if node is None:
            _abandon_deleted_node(workflow_id, current, info)
            state = workflow_db.get_state(workflow_id)
        elif info.get("status") in _ACTIVE_NODE_STATUSES:
            if _node_is_alive(workflow_id, node, info):
                if state.get("status") == "cancelling":
                    _escalate_cancel(workflow_id, node, info)
                return
            _finalize_node(workflow_id, graph, node, state)
            state = workflow_db.get_state(workflow_id)

    if state.get("status") == "cancelling":
        _stop_waiting_nodes(workflow_id, state)
        _finish_workflow(workflow_id, "stopped")
        return

    # Inline nodes complete synchronously, so keep going rather than costing a tick per node.
    for _ in range(len(graph.nodes) + 1):
        node, halt = _next_runnable(graph, state)
        if halt is not None:
            _finish_workflow(workflow_id, "failed" if halt == "failed" else "stopped")
            return
        if node is None:
            _finish_workflow(workflow_id, "done")
            return
        _update_workflow(workflow_id, current_node=node.id)
        state = workflow_db.get_state(workflow_id)
        outcome = _launch_node(workflow_id, graph, node, state)
        state = workflow_db.get_state(workflow_id)
        if outcome in ("failed", "stopped"):
            # A node that died at launch (a bad config, a closed toolbox gate) halts the chain
            # here rather than a tick later: the caller of /start must see the verdict it caused.
            _finish_workflow(workflow_id, outcome)
            return
        if outcome != "done":
            return  # running / waiting_gpu — the next pass looks again


def tick() -> None:
    """One pass over the workflow lane. Called from ``queue_poller._tick`` and from the routes.

    The lock is **non-blocking**: a second caller returns immediately rather than queueing behind
    the first, because the only thing a queued caller would do is re-launch work the first one has
    already started. See the module docstring for what two concurrent spawns cost.
    """
    if not _tick_lock.acquire(blocking=False):
        return
    try:
        active = 0
        for record in workflow_db.list_workflows():
            if _state_of(record).get("status") not in _ACTIVE_WORKFLOW_STATUSES:
                continue
            active += 1
            if active > MAX_ACTIVE_WORKFLOWS:
                _logger.warning(
                    "workflow %s is active but %d already are; leaving it for a later tick",
                    record.id,
                    MAX_ACTIVE_WORKFLOWS,
                )
                break
            try:
                _advance(record.id)
            except KeyboardInterrupt:
                raise
            except BaseException:  # noqa: BLE001 - one bad workflow must not stop the lane
                _logger.exception("workflow %s: advance failed", record.id)
    finally:
        _tick_lock.release()


# ------------------------------------------------------------------------------ start / cancel


def _active_workflow_id(exclude: Any = None) -> Any | None:
    for record in workflow_db.list_workflows():
        if record.id == exclude:
            continue
        if _state_of(record).get("status") in _ACTIVE_WORKFLOW_STATUSES:
            return record.id
    return None


def _with_descendants(graph: WorkflowGraph, root: str) -> set[str]:
    """*root* plus everything that reads from it, transitively.

    ``from`` only ever points backwards, so list order is already topological and one forward pass
    is the whole closure.
    """
    out = {root}
    for node in graph.nodes:
        if node.source in out:
            out.add(node.id)
    return out


def _require_saved_ancestors(
    graph: WorkflowGraph, nodes: Mapping[str, Any], from_node: str
) -> None:
    """"Run from here" is only meaningful when the steps it skips left something behind."""
    by_id = {n.id: n for n in graph.nodes}
    order = [n.id for n in graph.nodes]
    missing: list[str] = []
    seen: set[str] = set()
    stack = [by_id[from_node].source]
    while stack:
        node_id = stack.pop()
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        node = by_id.get(node_id)
        if node is None:
            continue
        if not (nodes.get(node_id) or {}).get("output"):
            missing.append(node_id)
        stack.append(node.source)
    if not missing:
        return
    index = min(order.index(node_id) for node_id in missing)
    raise ValueError(
        f"{_position_label(index)} has no saved output. "
        f"Start from {_position_label(max(0, index - 1))} or earlier."
    )


def _plan(
    graph: WorkflowGraph,
    state: Mapping[str, Any],
    *,
    from_node: str | None,
    force: bool,
) -> set[str]:
    nodes = state.get("nodes") or {}
    enabled = {n.id for n in graph.nodes if n.enabled}
    if from_node is not None:
        if from_node not in {n.id for n in graph.nodes}:
            raise ValueError(f"No node {from_node!r} in this workflow.")
        if from_node not in enabled:
            raise ValueError(f"{_label(graph, from_node)} is disabled; enable it first.")
        _require_saved_ancestors(graph, nodes, from_node)
        return _with_descendants(graph, from_node) & enabled

    # Run runs everything that is not done-and-fresh: idle + stale + failed + STOPPED. `stopped`
    # belongs in that set because prep stages resume naturally — skipping it is what would train on
    # a dataset that was only captioned up to 60 %.
    stale = compute_stale(graph, nodes)
    return {
        node.id
        for node in graph.nodes
        if node.enabled
        and (
            force
            or (nodes.get(node.id) or {}).get("status") != "done"
            or stale.get(node.id)
        )
    }


def start_workflow(
    workflow_id: Any, *, from_node: str | None = None, force: bool = False
) -> dict:
    """Plan a run. Raises ``ValueError`` with everything wrong, all at once.

    With ``from_node`` the named node and its descendants are reset to ``pending`` while its
    ancestors keep their ``done`` status **and their saved output** — which is what makes resuming
    from step 4 reuse steps 1-3 without re-running or re-typing anything.

    **Planning only: the caller ticks.** Every caller already does (the route ticks synchronously
    so the response carries the effect), and ticking here as well only bought a second full pass
    over the lane per ``/start``.
    """
    record = workflow_db.get_workflow(workflow_id)
    graph = _graph_of(record)
    errors = validate(graph)
    if errors:
        # Pre-flight reports every error at once; the promise is no mid-run surprises.
        raise ValueError("\n".join(errors))

    state = _state_of(record)
    if state.get("status") in _ACTIVE_WORKFLOW_STATUSES:
        raise ValueError("This workflow is already running. Stop it first.")
    other = _active_workflow_id(exclude=record.id)
    if other is not None:
        raise ValueError(
            f"Workflow {other} is already running; one workflow runs at a time."
        )

    plan = _plan(graph, state, from_node=from_node, force=force)

    def _apply(current: dict) -> None:
        nodes = current.setdefault("nodes", {})
        for node in graph.nodes:
            if not node.enabled:
                nodes.setdefault(node.id, {})["status"] = "skipped"
            elif node.id in plan:
                nodes.setdefault(node.id, {}).update(
                    {
                        "status": "pending",
                        "pid": None,
                        "pid_create_time": None,
                        "exit_code": None,
                        "error": "",
                        "started_at": None,
                        "finished_at": None,
                        "adopted": False,
                        "stop_requested_at": None,
                    }
                )
        current["status"] = "running"
        current["started_at"] = now_utc_iso()
        current["finished_at"] = None
        current["current_node"] = None

    workflow_db.mutate_state(workflow_id, _apply)
    return workflow_db.get_state(workflow_id)


def cancel_workflow(workflow_id: Any) -> dict:
    """Ask the active node to stop. Idempotent, and completed outputs are left intact.

    Records the intent; the caller's ``tick()`` is what escalates it (see :func:`start_workflow`).
    """
    def _apply(state: dict) -> None:
        if state.get("status") in _ACTIVE_WORKFLOW_STATUSES:
            state["status"] = "cancelling"

    workflow_db.mutate_state(workflow_id, _apply)
    return workflow_db.get_state(workflow_id)


# ------------------------------------------------------------------------------ reconciliation


def _adopt_launching(workflow_id: Any, node: WorkflowNode, info: Mapping[str, Any]) -> bool:
    """Adopt a ``launching`` node whose ``node.log`` is still growing — it is genuinely running.

    Two samples, because a size alone says nothing. The adopted node has no pid (``launching`` is
    written before the spawn, which is the point), so its liveness afterwards is that same log.
    """
    if node.type in INLINE_TYPES:
        return False
    first = _log_size(workflow_id, node.id)
    if first == 0:
        return False
    time.sleep(_ADOPTION_SAMPLE_SECONDS)
    second = _log_size(workflow_id, node.id)
    if second <= first:
        return False
    _update_node(
        workflow_id,
        node.id,
        status="running",
        adopted=True,
        log_size=second,
        log_seen_at=time.time(),
        error="",
    )
    return True


def _collect_garbage(record: workflow_db.WorkflowRecord, known: set[str]) -> bool:
    """Drop state rows for nodes the editor deleted, and free the leases they were holding."""
    state = _state_of(record)
    removed = [node_id for node_id in (state.get("nodes") or {}) if node_id not in known]
    if not removed:
        return False

    def _apply(current: dict) -> None:
        nodes = current.get("nodes") or {}
        for node_id in removed:
            nodes.pop(node_id, None)

    workflow_db.mutate_state(record.id, _apply)
    for node_id in removed:
        gpu_lease.release(_holder_id(record.id, node_id))
    return True


def _reconcile_workflow(record: workflow_db.WorkflowRecord) -> None:
    graph = _graph_of(record)
    workflow_id = record.id
    _collect_garbage(record, {node.id for node in graph.nodes})

    state = workflow_db.get_state(workflow_id)
    if state.get("status") not in _ACTIVE_WORKFLOW_STATUSES:
        return

    for node in graph.nodes:
        info = _node_state(state, node.id)
        status = info.get("status")
        if status == "waiting_gpu":
            # Re-evaluate the lease from scratch rather than trusting a decision the dead process
            # made about a GPU that may now be free (or taken by someone else).
            gpu_lease.release(_holder_id(workflow_id, node.id))
            _update_node(workflow_id, node.id, status="pending", error="")
        elif status == "launching":
            # NEVER auto-started: its process may be alive and detached, and a second `prep.tag`
            # writing the same sidecars is silent, dataset-wide corruption.
            if not _adopt_launching(workflow_id, node, info):
                _fail_node(workflow_id, node.id, LAUNCH_INTERRUPTED_ERROR)
                gpu_lease.release(_holder_id(workflow_id, node.id))
        elif status in ("running", "stopping"):
            if not _node_is_alive(workflow_id, node, info):
                _finalize_node(workflow_id, graph, node, state)
        state = workflow_db.get_state(workflow_id)


def reconcile_on_start() -> None:
    """Startup sweep for the workflow lane, run from the app lifespan.

    Node subprocesses **survive** a server restart (own process group, stdout to a file), so this
    adopts what is still alive and finalizes what is not — deliberately unlike Toolbox, which loses
    its in-memory handle and marks a still-running process ``failed``.
    """
    for record in workflow_db.list_workflows():
        try:
            _reconcile_workflow(record)
        except KeyboardInterrupt:
            raise
        except BaseException:  # noqa: BLE001 - one bad row must not block the whole lane
            _logger.exception("workflow %s: reconciliation failed", record.id)
