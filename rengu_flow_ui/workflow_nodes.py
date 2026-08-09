"""What one workflow node *does*: how it launches, what it emits, how it exited.

This is the mechanics layer between :mod:`rengu_flow_ui.workflow_graph` (pure model: shapes,
variables, the output rule) and ``workflow_runner`` (scheduling: leases, ticks, cancellation).
Nothing here schedules anything, spawns anything or touches ``state_json`` — every function takes
a node plus its incoming handle and answers one question, which is what lets the runner stay a
state machine and lets these rules be tested without a subprocess.

Three families, dispatched by an ``if node.type.startswith("prep.")`` chain and **no registry**:
the repo's registries exist for pluggable *training* components, and three node families do not
warrant one (``AGENTS.md``).

Load-bearing rules, all from ``docs/spec/workflows.md``:

* **The handle owns ``path`` / ``caption_format`` / ``caption_ext``.** A node's ``config`` is the
  stage section *minus* those three ("Graph model"); the executor injects them from the incoming
  edge and ignores any copy left in the config. They cannot be per-node settings while
  ``workflow_graph.effective_output`` passes the handle through unchanged — the node would write
  ``.json`` captions and hand its successor a handle still claiming ``sidecar``/``.txt``.
* **``validate_for_stage`` runs before the spawn.** It is the same check
  ``prep_routes.create_prep_job`` performs today, and it is what turns "the folder does not
  exist" into a clean ``error`` on the node card instead of a traceback buried in ``node.log``.
* **A node's environment is not a training environment.** ``os.environ`` + ``PYTHONUNBUFFERED``
  + ``CUDA_VISIBLE_DEVICES`` when the node pins a device. NCCL/TF32/allocator knobs are training
  concerns that prep only inherits today because ``jobs.start_job`` is shared.
* **A tool's handle injection is by convention, not configuration**: a tool that declares an
  input named ``path`` gets the incoming handle; one that does not is a pass-through. No mapping
  UI, no mapping schema.
* **A missing ``result.json`` is a failure, not a ``None``.** The shim writes it in its postlude
  on every successful exit, so an absent file means the tool raised. Reading it as ``None`` would
  take the pass-through branch and carry a green workflow past a crashed tool.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml

from rengu_flow_ui import toolbox
from rengu_flow_ui.workflow_graph import (
    NODE_TYPES,
    DatasetHandle,
    NodeOutputError,
    WorkflowNode,
    effective_output,
)

_logger = logging.getLogger("rengu_flow_ui.workflow_nodes")

#: Files inside a node directory (the spec's "Node directory"), named once and used from here.
NODE_LOG_NAME = "node.log"
PREP_CONFIG_NAME = "prep.toml"
REPORT_NAME = "report.json"
RESULT_NAME = "result.json"

#: Node types that run *in* the UI process: no argv, no env, no lease, no log.
INLINE_TYPES = ("folder", "train")

#: The three keys a prep TOML carries at its top level rather than inside the stage section.
_HANDLE_KEYS = ("path", "caption_format", "caption_ext")

#: Both markers — ``prep <stage> exits with return code = N`` (``rengu_flow/prep/runner.py:306``)
#: and ``tool exits with return code = N`` (the Toolbox shim) — end the same way.
_EXIT_CODE_RE = re.compile(r"exits with return code\s*=\s*(-?\d+)")

#: Enough tail to always contain the exit marker, which is the last thing either child prints,
#: without reading a caption run's multi-MB log on every tick. Mirrors ``jobs._PARSE_TAIL_BYTES``.
_EXIT_TAIL_BYTES = 262_144

#: Message for the failure that would otherwise be silent — see the module docstring.
RESULT_MISSING_ERROR = (
    "The tool did not write result.json, so it never reached the end of its run "
    "(see node.log). A workflow cannot continue from a tool that failed."
)


@dataclass
class NodeLaunch:
    """Everything the runner needs to spawn one node, and nothing about *when* to spawn it.

    The lease decision is deliberately **not** here: the runner has to know whether to acquire —
    and on which devices — *before* it builds a launch, because a node that cannot get its GPU is
    never built at all. It asks :func:`needs_lease` / :func:`lease_devices` directly.
    """

    argv: list[str]
    env: dict[str, str]


def node_log_path(node_dir: Path) -> Path:
    return node_dir / NODE_LOG_NAME


# ------------------------------------------------------------------------------ prep nodes


def build_prep_command(config_path: Path, *, stage: str, job_dir: Path) -> list[str]:
    """Argv for a dataset-prep run: the same `rengu prep <stage>` the CLI runs.

    The prep CLI installs its own extras on demand (uv sync --extra prep), so no
    ensure_training_extras here. ``--job-dir`` points signals + report.json at the
    node's own folder.
    """
    return [
        sys.executable,
        "-m",
        "rengu_flow.cli",
        "prep",
        stage,
        "--config",
        str(config_path),
        "--job-dir",
        str(job_dir),
    ]


def _prep_payload(node: WorkflowNode, inputs: DatasetHandle | None) -> tuple[str, dict[str, Any]]:
    """``(stage, the prep TOML body)`` for a ``prep.*`` node whose config is already resolved.

    All three top-level keys come from the edge and **only** from the edge: that is what makes
    "change the input folder" a single edit in a single node, and it is the only assignment that
    stays true downstream. A node that could pick its own ``caption_format`` would still emit the
    *input* handle — ``effective_output`` passes ``tag`` / ``caption`` / ``quality`` / ``index``
    through untouched — so the next node would read the format this one stopped writing. A copy of
    any of the three left in ``node.config`` is therefore dropped, not honoured; the spec already
    excludes them from the stage section.
    """
    stage = node.type.split(".", 1)[1]
    data = {key: value for key, value in node.config.items() if key not in _HANDLE_KEYS}
    top: dict[str, Any] = {}
    if inputs is not None:
        top = {
            "path": inputs.path,
            "caption_format": inputs.caption_format,
            "caption_ext": inputs.caption_ext,
        }
    return stage, {**top, stage: data}


def _build_prep_launch(
    node: WorkflowNode, inputs: DatasetHandle | None, node_dir: Path
) -> NodeLaunch:
    from rengu_flow.prep.config import parse_prep_config

    stage, payload = _prep_payload(node, inputs)
    config = parse_prep_config(payload)
    # BEFORE anything is written or spawned: a missing folder is a node error, not a traceback.
    config.validate_for_stage(stage)

    node_dir.mkdir(parents=True, exist_ok=True)
    config_path = node_dir / PREP_CONFIG_NAME
    # The sparse payload, not the materialized dataclass: the file then reads like the node's own
    # configuration, and the stage fills its defaults the same way it does for a hand-written TOML.
    # (``toml.dumps`` drops ``None`` values, which for every optional field means "use the
    # default" — exactly what ``None`` already means in ``rengu_flow/prep/config.py``.)
    config_path.write_text(toml.dumps(payload), encoding="utf-8")
    return NodeLaunch(
        argv=build_prep_command(config_path, stage=stage, job_dir=node_dir),
        env=_node_env(node),
    )


# ------------------------------------------------------------------------------ tool nodes


def _build_tool_launch(
    node: WorkflowNode, inputs: DatasetHandle | None, node_dir: Path
) -> NodeLaunch:
    tool_id = str(node.config.get("tool_id") or "").strip()
    if not tool_id:
        raise ValueError("A tool node needs a 'tool_id' in its config")
    data = toolbox._read_tool_json(tool_id)  # KeyError when the tool was deleted
    inputs_def = list(data.get("inputs") or [])
    declared = {str(spec.get("param")) for spec in inputs_def}

    values = dict(node.config.get("values") or {})
    # Injection by convention: a tool participates in a workflow by declaring ``path``. One that
    # does not declare it never sees the handle and the node is a pass-through — no mapping UI.
    if inputs is not None and "path" in declared:
        values["path"] = inputs.path
        if "caption_format" in declared:
            values["caption_format"] = inputs.caption_format
        if "caption_ext" in declared:
            values["caption_ext"] = inputs.caption_ext

    kwargs = toolbox.cast_inputs(inputs_def, values)
    # ``run_dir=node_dir``: each node runs the tool out of its own folder, so two nodes using the
    # same tool — or a workflow running one while ``/toolbox`` does — cannot overwrite each
    # other's ``tool.py`` / ``inputs.json`` / ``result.json``.
    argv = toolbox.materialize_run(tool_id, kwargs, run_dir=node_dir)
    return NodeLaunch(argv=argv, env=_node_env(node))


# ------------------------------------------------------------------------------ launch policy


def _node_env(node: WorkflowNode) -> dict[str, str]:
    """The inherited environment plus exactly two knobs. **Not** ``training_subprocess_env()``.

    ``CUDA_VISIBLE_DEVICES`` is merged last so the node's device choice beats anything already
    exported; with it set the child sees that GPU as ``cuda:0`` and the whole of
    ``rengu_flow/prep/`` needs no change to honour the selection.
    """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if node.gpu.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(node.gpu.device))
    return env


def needs_lease(node: WorkflowNode) -> bool:
    """``wait: false`` is the explicit escape hatch: start now, share the GPU, take no lease.

    Public because the lease policy has exactly one definition and the runner has to consult it
    before :func:`build_launch` — a node whose GPU is busy is never built.
    """
    return bool(node.gpu.required and node.gpu.wait)


def lease_devices(node: WorkflowNode) -> list[int] | None:
    """``None`` means auto / host-exclusive, which ``gpu_lease.acquire`` reads as every device."""
    return None if node.gpu.device is None else [int(node.gpu.device)]


#: Pre-promotion names, kept so the runner keeps importing either spelling.
_needs_lease = needs_lease
_devices = lease_devices


def build_launch(
    node: WorkflowNode, inputs: DatasetHandle | None, node_dir: Path
) -> NodeLaunch | None:
    """How to spawn *node*, or ``None`` when it has no subprocess (``folder``, ``train``).

    Raises before anything is written when the node cannot run at all: an unknown type, a prep
    stage whose folder does not exist, a tool node with no ``tool_id``.
    """
    if node.type not in NODE_TYPES:
        raise ValueError(f"Unknown node type {node.type!r}")
    if node.type in INLINE_TYPES:
        return None
    if node.type.startswith("prep."):
        return _build_prep_launch(node, inputs, node_dir)
    if node.type == "tool":
        return _build_tool_launch(node, inputs, node_dir)
    raise ValueError(f"Node type {node.type!r} has no launcher")


# ------------------------------------------------------------------------------ inline nodes


def _run_folder(node: WorkflowNode) -> dict[str, Any]:
    handle = effective_output(node, None)
    assert handle is not None  # a folder node always emits; see the catalog
    if not handle.path:
        raise ValueError("A folder node needs a 'path'")
    if not Path(handle.path).is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {handle.path}")
    return handle.to_dict()


def _run_train(node: WorkflowNode) -> dict[str, Any]:
    """Fire a run that is **already registered**; never build one.

    Every stage of prep writes in place, so the folder the workflow processed is the same folder
    the registered dataset already names — there is nothing to synthesize, rewrite or inject.

    The run goes to the *front* of the pending queue so forty minutes of tagging is not stuck
    behind someone else's run from yesterday, and then the queue is asked to start. It is never a
    preemption: with a run already active ``try_start_next`` is a no-op and this one simply waits
    its turn at the front.
    """
    from rengu_flow_ui import db, job_queue

    raw = node.config.get("job_id")
    if raw is None or raw == "":
        raise ValueError("A train node needs a 'job_id': pick an existing run from the queue.")
    try:
        job = db.get_job(raw)
    except KeyError:
        raise ValueError(f"Run {raw} no longer exists; pick another run.") from None

    if job.state == "new":
        job_queue.enqueue_existing(job.id)  # a saved draft is promoted to pending, not started
    elif job.state != "pending":
        raise ValueError(
            f"Run {job.id} is {job.state}; a train node can only fire a saved or queued run."
        )
    job_queue.bump_pending_after(job.id)
    job_queue.try_start_next()
    # The id, so the card can read "Queued run #123 ->" and link to /runs/jobs/123 — a train node
    # is done when the run is *enqueued*, which is not the same thing as trained.
    return {"job_id": job.id}


def run_inline(
    node: WorkflowNode, inputs: DatasetHandle | None, node_dir: Path
) -> dict[str, Any]:
    """Execute a node that has no subprocess, and return what the runner should record.

    ``folder`` returns its handle; ``train`` returns ``{"job_id": N}``.
    """
    if node.type == "folder":
        return _run_folder(node)
    if node.type == "train":
        return _run_train(node)
    raise ValueError(f"Node type {node.type!r} is not inline; use build_launch()")


# ------------------------------------------------------------------------------ results


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_output(
    node: WorkflowNode, node_dir: Path, inputs: DatasetHandle | None
) -> DatasetHandle | None:
    """The handle *node* emits after running — the spec's output table, fed from disk.

    ``workflow_graph.effective_output`` owns the rule; this only decides what to feed it: a prep
    stage's ``report.json`` (absent means "predict from config", which is what every stage but
    ``clean`` emits anyway) or a tool's ``result.json`` (absent means the tool **failed**).

    ``node.config`` is read literally, so callers must pass a node whose variables are already
    resolved. Raises :class:`~rengu_flow_ui.workflow_graph.NodeOutputError` when the node
    produced something that is not a handle.
    """
    if node.type == "tool":
        result_path = node_dir / RESULT_NAME
        if not result_path.is_file():
            raise NodeOutputError(RESULT_MISSING_ERROR)
        try:
            result = _read_json(result_path)
        except ValueError as exc:  # truncated: the tool died mid-write, same conclusion
            raise NodeOutputError(f"{RESULT_MISSING_ERROR} ({exc})") from exc
        return effective_output(node, inputs, result)

    report: Any = None
    if node.type.startswith("prep."):
        report_path = node_dir / REPORT_NAME
        if report_path.is_file():
            try:
                report = _read_json(report_path)
            except ValueError:
                # Only ``clean`` reads anything out of the report, and its fallback is the very
                # ``output_dir`` the config already names — so a corrupt report costs nothing.
                _logger.info("Ignoring unreadable %s for node %s", report_path, node.id)
    return effective_output(node, inputs, report)


def read_exit_code(node: WorkflowNode, node_dir: Path) -> int | None:
    """The exit code parsed from ``node.log``, or ``None`` when the log does not say.

    A node's process is detached by design, so after a server restart there is no ``Popen`` left
    to ``wait()`` on; both children therefore print a marker as their last line.

    **Unknown stays unknown.** ``jobs._read_exit_code`` maps an unparseable log to *success*,
    which is defensible for a training run whose own log is the only record; here it would mark a
    node ``done`` and propagate its handle on no evidence at all. The runner decides what to do
    with ``None``.
    """
    if node.type in INLINE_TYPES:
        return None
    # Imported lazily: ``jobs`` imports this module for ``build_prep_command``, and the tail
    # reader is the one thing worth borrowing from it (decode + CRLF normalization included).
    from rengu_flow_ui.jobs import read_raw_log_tail_path

    text = read_raw_log_tail_path(node_log_path(node_dir), _EXIT_TAIL_BYTES)
    codes = _EXIT_CODE_RE.findall(text)
    return int(codes[-1]) if codes else None
