"""The workflow graph: dataclasses, tolerant parsing, validation, variables, staleness.

This module is **pure** — no database, no filesystem, no subprocesses. It is the shared
vocabulary between the editor (which saves a graph), the runner (which executes one) and the
staleness ring the UI draws; keeping it free of I/O is what lets all three test it cheaply.

Three rules from ``docs/spec/workflows.md`` are load-bearing and easy to get wrong:

* **``from`` may only reference an earlier node.** That single invariant makes cycles
  structurally impossible, so there is deliberately no cycle detection and no topological
  sort here: :func:`execution_order` is list order.
* **Parsing is tolerant** (mirroring :func:`rengu_flow.prep.config.parse_prep_config`): unknown
  keys are logged and dropped, and an unknown *node type* is preserved on parse — it is fatal at
  execution only, so downgrading the app never destroys the user's graph.
* **``prep.quality`` emits its input handle.** Its ``report["output_dir"]`` is the *quarantine*
  folder (``rengu_flow/prep/quality.py:179``), not the result; ``prep.clean`` is the only stage
  whose ``output_dir`` names the resulting dataset. A generic "read ``report['output_dir']``"
  would make ``quality -> caption`` caption the reject pile. See :func:`effective_output`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_logger = logging.getLogger("rengu_flow_ui.workflow_graph")

#: Bumped when the hash *recipe* changes, or when a changed default would otherwise alter
#: behaviour while the hash held still. Part of every digest, so a bump marks every node stale.
HASH_VERSION = 1

#: The message a tool node fails with when its return value is not a handle.
TOOL_RETURN_ERROR = (
    "A tool used in a workflow must return a folder path, a dict with a 'path' key, or None."
)

_DEFAULT_CAPTION_FORMAT = "sidecar"
_DEFAULT_CAPTION_EXT = ".txt"

#: ``$$`` (escaped literal ``$``) or ``${name}``. One regex for substitution *and* reference
#: collection, so both agree on what the escape hides.
_TOKEN_RE = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ------------------------------------------------------------------------------ shapes


@dataclass(frozen=True)
class DatasetHandle:
    """The only value that travels between nodes. Connecting is always valid."""

    path: str
    caption_format: str = _DEFAULT_CAPTION_FORMAT
    caption_ext: str = _DEFAULT_CAPTION_EXT

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "caption_format": self.caption_format,
            "caption_ext": self.caption_ext,
        }


@dataclass
class NodeGpu:
    """Per-node GPU policy. ``device`` is a physical index; ``None`` means auto/host-exclusive."""

    required: bool = False
    wait: bool = True
    device: int | None = None


@dataclass
class Variable:
    """A workflow-level string constant. Configuration only — never node output."""

    name: str = ""
    value: str = ""
    description: str = ""


@dataclass
class WorkflowNode:
    """One executable step.

    ``source`` is the JSON's ``from`` (a keyword in Python); :func:`parse_graph` and
    :func:`graph_to_dict` map between the two names. ``config`` holds exactly the stage section
    minus ``path`` / ``caption_format`` / ``caption_ext`` — those are injected by the executor
    from the incoming handle, which is what makes "change the input folder" a one-field edit.
    """

    id: str = ""
    type: str = ""
    title: str = ""
    source: str | None = None
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    gpu: NodeGpu = field(default_factory=NodeGpu)


@dataclass
class WorkflowGraph:
    version: int = 1
    name: str = ""
    description: str = ""
    variables: list[Variable] = field(default_factory=list)
    nodes: list[WorkflowNode] = field(default_factory=list)


class NodeOutputError(ValueError):
    """A node produced something that cannot be turned into a handle. Fatal for that node."""


# ------------------------------------------------------------------------------ node catalog


@dataclass(frozen=True)
class NodeType:
    """Catalog entry: what a type is called, what it does with handles, and its GPU default."""

    type: str
    label: str
    consumes: bool  # reads an upstream handle
    emits: bool  # produces a handle downstream nodes can read
    needs_gpu: bool  # default for ``gpu.required``; see :func:`default_needs_gpu`
    source_optional: bool = False  # may legitimately have ``from: null``


NODE_TYPES: dict[str, NodeType] = {
    nt.type: nt
    for nt in (
        NodeType("folder", "Source folder", False, True, False, source_optional=True),
        NodeType("prep.tag", "Tag", True, True, True),
        NodeType("prep.caption", "Caption", True, True, True),
        NodeType("prep.clean", "Clean", True, True, True),
        NodeType("prep.quality", "Quality filter", True, True, True),
        NodeType("prep.index", "Quality index", True, True, True),
        NodeType("tool", "Tool", True, True, False, source_optional=True),
        NodeType("train", "Training run", True, False, False),
    )
}


def default_needs_gpu(node_type: str, config: Mapping[str, Any] | None = None) -> bool:
    """Default for ``gpu.required``. The user can always override it on the node.

    ``prep.quality`` follows its metric: ``blur`` is a pure-CPU Laplacian variance with no extra
    dependencies, while ``aesthetic`` and ``iqa`` load models.
    """
    spec = NODE_TYPES.get(node_type)
    if spec is None:
        return False
    if node_type == "prep.quality":
        return (config or {}).get("metric", "blur") != "blur"
    return spec.needs_gpu


# ------------------------------------------------------------------------------ parsing


def _fill(instance: Any, data: Mapping[str, Any], *, context: str, skip: Iterable[str] = ()) -> Any:
    """Copy known keys onto *instance*; log and drop the rest. Mirrors ``_fill_dataclass``."""
    known = set(instance.__dataclass_fields__) - set(skip)
    for key, value in data.items():
        if key in known:
            setattr(instance, key, value)
        else:
            _logger.info("Ignoring unknown workflow key %s.%s", context, key)
    return instance


def _parse_gpu(data: Any, *, context: str) -> NodeGpu:
    gpu = NodeGpu()
    if isinstance(data, dict):
        _fill(gpu, data, context=f"{context}.gpu")
    elif data is not None:
        _logger.info("Ignoring non-object %s.gpu", context)
    return gpu


#: Node keys :func:`_parse_node` maps by hand; they must not reach the generic filler.
_NODE_HANDLED_KEYS = ("from", "config", "gpu")


def _parse_node(data: Mapping[str, Any], *, context: str) -> WorkflowNode:
    node = WorkflowNode()
    # ``source`` is *not* a door: the JSON key is ``from`` and nothing else, so a stray
    # ``source`` key is dropped like any other unknown one.
    plain = {k: v for k, v in data.items() if k not in _NODE_HANDLED_KEYS}
    _fill(node, plain, context=context, skip=("source",))
    node.id = str(node.id) if node.id is not None else ""
    node.type = str(node.type) if node.type is not None else ""
    src = data.get("from")
    node.source = src if isinstance(src, str) and src else None
    raw_config = data.get("config")
    if isinstance(raw_config, dict):
        node.config = dict(raw_config)
    elif raw_config is not None:
        _logger.info("Ignoring non-object %s.config", context)
    node.gpu = _parse_gpu(data.get("gpu"), context=context)
    return node


def parse_graph(data: Mapping[str, Any]) -> WorkflowGraph:
    """Build a graph from decoded JSON. Never fatal — unknown keys are logged and dropped.

    An unknown ``type`` is **kept**: :func:`validate` reports it and execution refuses it, but a
    graph written by a newer app must survive a round-trip through an older one intact.
    """
    graph = WorkflowGraph()
    if not isinstance(data, Mapping):
        _logger.info("Ignoring non-object workflow graph")
        return graph
    _fill(graph, data, context="workflow", skip=("variables", "nodes"))

    raw_variables = data.get("variables")
    if isinstance(raw_variables, list):
        for index, item in enumerate(raw_variables):
            if isinstance(item, dict):
                graph.variables.append(_fill(Variable(), item, context=f"variables[{index}]"))
            else:
                _logger.info("Ignoring non-object variables[%d]", index)
    elif raw_variables is not None:
        _logger.info("Ignoring non-list workflow.variables")

    raw_nodes = data.get("nodes")
    if isinstance(raw_nodes, list):
        for index, item in enumerate(raw_nodes):
            if isinstance(item, dict):
                graph.nodes.append(_parse_node(item, context=f"nodes[{index}]"))
            else:
                _logger.info("Ignoring non-object nodes[%d]", index)
    elif raw_nodes is not None:
        _logger.info("Ignoring non-list workflow.nodes")
    return graph


def node_to_dict(node: WorkflowNode) -> dict[str, Any]:
    """Serialize one node, mapping ``source`` back to the JSON's ``from``."""
    return {
        "id": node.id,
        "type": node.type,
        "title": node.title,
        "from": node.source,
        "enabled": node.enabled,
        "config": node.config,
        "gpu": {
            "required": node.gpu.required,
            "wait": node.gpu.wait,
            "device": node.gpu.device,
        },
    }


def graph_to_dict(graph: WorkflowGraph) -> dict[str, Any]:
    """The JSON form. ``parse_graph(graph_to_dict(g)) == g`` for any parsed graph."""
    return {
        "version": graph.version,
        "name": graph.name,
        "description": graph.description,
        "variables": [
            {"name": v.name, "value": v.value, "description": v.description}
            for v in graph.variables
        ],
        "nodes": [node_to_dict(n) for n in graph.nodes],
    }


# ------------------------------------------------------------------------------ variables


def _variable_map(variables: Mapping[str, Any] | Iterable[Variable] | None) -> dict[str, str]:
    """Accept either a plain mapping or the graph's ``list[Variable]``."""
    if variables is None:
        return {}
    if isinstance(variables, Mapping):
        items = variables.items()
    else:
        items = [(v.name, v.value) for v in variables]
    return {str(name): value if isinstance(value, str) else str(value) for name, value in items}


def resolve_text(text: str, variables: Mapping[str, Any] | Iterable[Variable] | None) -> str:
    """Substitute ``${name}`` in *text*. ``$$`` is a literal ``$``.

    **A single pass, no recursion**: a variable whose value contains ``${other}`` is left as-is.
    An unknown variable keeps its literal ``${name}`` so :func:`validate` can point at it — it is
    never substituted with an empty string, which would silently run a stage against ``/``.
    """
    if not isinstance(text, str):
        return text
    mapping = _variable_map(variables)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name is None:  # the ``$$`` branch
            return "$"
        if name in mapping:
            return mapping[name]
        return match.group(0)

    return _TOKEN_RE.sub(_replace, text)


def _resolve_value(value: Any, variables: dict[str, str]) -> Any:
    """Strings only, recursively. Numbers and booleans are never touched."""
    if isinstance(value, str):
        return resolve_text(value, variables)
    if isinstance(value, dict):
        return {key: _resolve_value(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, variables) for item in value]
    return value


def resolve_config(
    node: WorkflowNode, variables: Mapping[str, Any] | Iterable[Variable] | None
) -> dict[str, Any]:
    """A copy of ``node.config`` with variables resolved in **string values only**.

    Substituting text into a numeric field would force coercion rules and a new class of type
    error; "variables are text in text fields" is a rule nobody has to learn.
    """
    return _resolve_value(node.config, _variable_map(variables))


def _config_label(node_type: str) -> str:
    """The prefix used when pointing at a config field: ``prep.quality`` -> ``quality``."""
    return node_type.split(".", 1)[1] if node_type.startswith("prep.") else node_type


def _walk_strings(value: Any, path: str) -> Iterator[tuple[str, str]]:
    """Yield ``(field_path, text)`` for every string in a config tree."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")


def _node_refs(node: WorkflowNode) -> Iterator[tuple[str, str]]:
    """Yield ``(field_path, variable_name)`` for every reference in a node's config."""
    for path, text in _walk_strings(node.config, _config_label(node.type)):
        for match in _TOKEN_RE.finditer(text):
            name = match.group(1)
            if name is not None:  # ``$$`` is an escape, not a reference
                yield path, name


def collect_refs(graph: WorkflowGraph) -> dict[str, list[str]]:
    """Variable name -> the places it is used, for the editor's "used by" column.

    Every referenced name is reported, including ones the workflow does not define — the UI looks
    up by name, and an undefined reference is exactly what the user needs to find.
    """
    refs: dict[str, list[str]] = {}
    for node in graph.nodes:
        for path, name in _node_refs(node):
            where = f"{node.id} · {path}"
            locations = refs.setdefault(name, [])
            if where not in locations:
                locations.append(where)
    return refs


# ------------------------------------------------------------------------------ validation


def _preflight_path() -> str:
    """A stand-in for the ``path`` the executor injects from the incoming handle.

    :meth:`PrepConfig.validate_for_stage` demands a ``path`` that is an existing directory, but in
    a workflow **no node owns its path** — the edge does, and it is only known once the upstream
    node has run (a ``prep.clean`` may not have created its ``output_dir`` yet). Judging the graph
    on it would make every pre-flight a guaranteed false positive, so the check is fed a directory
    that trivially exists and the path rules are left to the launch, which has the real handle.
    The working directory is the cheapest such value and needs no cleanup.
    """
    return str(Path.cwd())


def _prep_config_errors(node: WorkflowNode, graph: WorkflowGraph, where: str) -> list[str]:
    """``validate_for_stage`` for one ``prep.*`` node — the *launch* check, run at pre-flight.

    Structure alone does not make a graph runnable: a ``prep.index`` with an empty config is what
    "Add step" creates, and it used to pass pre-flight clean and then die inside
    ``workflow_nodes._build_prep_launch`` — mid-run, after the earlier nodes had already done their
    work. Running the identical check here is the only way the docstring above stays true.
    """
    stage = node.type.split(".", 1)[1]
    # Imported lazily, like :func:`materialize_config`: routes that only read a graph must not pay
    # for the prep package on import.
    from rengu_flow.prep import config as prep_config

    if stage not in prep_config.STAGES:
        return []
    try:
        parsed = prep_config.parse_prep_config(
            {"path": _preflight_path(), stage: materialize_config(node, graph.variables)}
        )
        parsed.validate_for_stage(stage)
    except (ValueError, OSError) as exc:
        return [f"{where} · {exc}"]
    return []


def validate(graph: WorkflowGraph) -> list[str]:
    """Every error in the graph, at once — pre-flight promises no mid-run surprises.

    Structure *and* substance: each enabled ``prep.*`` node's config is materialized and put
    through :func:`_prep_config_errors`, the same gate the launch runs, so "pre-flight passed"
    means the run will not stop on a config the editor let the user save.

    Cycles are not checked: ``from`` pointing only backwards makes them impossible to express.
    """
    errors: list[str] = []
    seen: set[str] = set()
    defined = {v.name for v in graph.variables}
    positions = {node.id: index for index, node in enumerate(graph.nodes)}

    for index, node in enumerate(graph.nodes):
        where = f"node {node.id}"
        if node.id in seen:
            errors.append(f"{where} · duplicate node id")
        seen.add(node.id)

        spec = NODE_TYPES.get(node.type)
        if spec is None:
            errors.append(f"{where} · unknown node type {node.type!r}")

        if node.source is None:
            # Deleting a ``folder`` node splices its children onto *its* ``from``, which is null.
            # Without this rule a sourceless ``prep.clean`` passes pre-flight and dies mid-run on
            # ``validate_for_stage``'s "Prep config needs a dataset 'path'" — after the earlier
            # nodes already ran.
            if spec is not None and not spec.source_optional:
                errors.append(
                    f"{where} · has no source; only folder and tool nodes may have none"
                )
        elif node.source == node.id:
            errors.append(f"{where} · from {node.source!r} points at itself")
        elif node.source not in positions:
            errors.append(f"{where} · from {node.source!r} is not a node in this workflow")
        elif positions[node.source] > index:
            errors.append(
                f"{where} · from {node.source!r} points forward; a node may only read "
                "from an earlier node"
            )

        unresolved = False
        for path, name in _node_refs(node):
            if name not in defined:
                errors.append(f"{where} · {path} → unknown variable ${{{name}}}")
                unresolved = True

        # A node whose variables do not resolve is already reported; running the stage check on a
        # config still carrying a literal ``${name}`` would pile a second, derived error on top of
        # the one the user actually has to fix.
        if node.enabled and spec is not None and node.type.startswith("prep.") and not unresolved:
            errors.extend(_prep_config_errors(node, graph, where))
    return errors


# ------------------------------------------------------------------------------ execution order


def execution_order(graph: WorkflowGraph) -> list[WorkflowNode]:
    """List order, disabled nodes skipped. That is the whole scheduler.

    ``from`` decides *which folder* a node reads, never *when* it runs, and the backward-only
    invariant guarantees list order is always a legal order.
    """
    return [node for node in graph.nodes if node.enabled]


# ------------------------------------------------------------------------------ hashing


def materialize_config(
    node: WorkflowNode, variables: Mapping[str, Any] | Iterable[Variable] | None = None
) -> dict[str, Any]:
    """The node's config with variables resolved **and defaults filled from this app version**.

    Hashing the stored partial dict instead would make a form that gains one field turn every
    saved node in every workflow amber on upgrade day. For ``prep.*`` the corresponding stage
    dataclass in ``rengu_flow/prep/config.py`` is built through the same tolerant parser the
    engine uses and serialized whole; other types have no schema to fill from.
    """
    resolved = resolve_config(node, variables)
    if not node.type.startswith("prep."):
        return resolved
    stage = node.type.split(".", 1)[1]
    # Imported lazily: the graph model is pure and is read by routes that must not pay for the
    # prep package on import.
    from rengu_flow.prep import config as prep_config

    if stage not in prep_config.STAGES:
        return resolved
    parsed = prep_config.parse_prep_config({stage: resolved})
    return asdict(getattr(parsed, stage))


def node_config_hash(
    node: WorkflowNode,
    parent_hash: str | None = None,
    variables: Mapping[str, Any] | Iterable[Variable] | None = None,
) -> str:
    """Digest over ``(HASH_VERSION, type, materialized config, gpu, parent_hash)``.

    Because the parent's hash feeds the child's, **downstream invalidation is free** — there is no
    propagation code anywhere. Computed on the server only: ``JSON.stringify`` emits ``80`` where
    ``json.dumps`` emits ``80.0``, so a client-side recomputation would disagree on every
    ``prep.quality`` node.
    """
    payload = {
        "hash_version": HASH_VERSION,
        "type": node.type,
        "config": materialize_config(node, variables),
        "gpu": {
            "required": node.gpu.required,
            "wait": node.gpu.wait,
            "device": node.gpu.device,
        },
        "parent": parent_hash or "",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _handle_key(data: Any) -> tuple[str, str, str] | None:
    """A comparable form of a saved handle, so defaults never read as a change."""
    if not isinstance(data, Mapping) or not data:
        return None
    return (
        str(data.get("path", "")),
        str(data.get("caption_format", _DEFAULT_CAPTION_FORMAT)),
        str(data.get("caption_ext", _DEFAULT_CAPTION_EXT)),
    )


def compute_stale(
    graph: WorkflowGraph, state_nodes: Mapping[str, Mapping[str, Any]] | None
) -> dict[str, bool]:
    """Node id -> whether what it produced no longer matches its configuration.

    ``stale(n) = n has already run and (current_hash != saved_hash or
    effective_input != saved_input)``. A node that never ran is never stale — there is nothing to
    distrust yet.

    "Has already run" is a ``config_hash``, **not** a saved output: ``train`` is terminal and emits
    nothing (:func:`effective_output` returns ``None``), so testing for an output would make a
    ``done`` training node permanently fresh. The user would change ``${dataset_dir}``, press Run,
    watch the tag stage re-label the new folder — and the training node would never fire again,
    while its card stayed green with yesterday's ``job_id``.

    The saved *input* is folded in because the config chain cannot see it: a tool that returns a
    computed folder (``f"{path}/export-{ts}"``) leaves its downstream node's config textually
    unchanged, so a config-only rule would call it fresh, ``Run`` would skip it, and the run bar
    would report a new empty folder as the result.
    """
    state = state_nodes or {}
    hashes: dict[str, str] = {}
    stale: dict[str, bool] = {}
    for node in graph.nodes:
        # ``from`` always points backwards, so the parent's hash is already computed.
        parent_hash = hashes.get(node.source, "") if node.source else ""
        current = node_config_hash(node, parent_hash, graph.variables)
        hashes[node.id] = current

        saved = state.get(node.id) or {}
        # ``config_hash`` is written by every completion, with or without an output.
        if not (saved.get("config_hash") or saved.get("output")):
            stale[node.id] = False
            continue
        upstream = (state.get(node.source) or {}).get("output") if node.source else None
        stale[node.id] = saved.get("config_hash") != current or _handle_key(
            upstream
        ) != _handle_key(saved.get("saved_input"))
    return stale


# ------------------------------------------------------------------------------ output rule


def _inherit(
    path: str, input_handle: DatasetHandle | None, overrides: Mapping[str, Any] | None = None
) -> DatasetHandle:
    """A handle at *path*, inheriting format/ext from the input unless overridden."""
    over = overrides or {}
    base_format = input_handle.caption_format if input_handle else _DEFAULT_CAPTION_FORMAT
    base_ext = input_handle.caption_ext if input_handle else _DEFAULT_CAPTION_EXT
    return DatasetHandle(
        path=str(path),
        caption_format=str(over.get("caption_format", base_format)),
        caption_ext=str(over.get("caption_ext", base_ext)),
    )


def effective_output(
    node: WorkflowNode,
    input_handle: DatasetHandle | None = None,
    report: Any = None,
) -> DatasetHandle | None:
    """The handle a node emits — the normative table of the spec.

    *report* is the stage's ``report.json`` for ``prep.*`` nodes and the tool's return value
    (``result.json``) for ``tool`` nodes. With no report the answer is the *predicted* handle the
    UI shows before a run; once the node has run, the report wins.

    ``node.config`` is read literally, so a caller that uses variables must pass a node whose
    config is already resolved (``dataclasses.replace(node, config=resolve_config(node, vars))``).

    Raises :class:`NodeOutputError` for an unknown node type and for a tool that returned
    something which is not a handle.
    """
    node_type = node.type
    if node_type not in NODE_TYPES:
        raise NodeOutputError(f"Unknown node type {node_type!r}")

    if node_type == "folder":
        return _inherit(node.config.get("path", ""), None, node.config)

    if node_type == "prep.clean":
        # The ONLY stage whose ``report["output_dir"]`` names the result folder.
        if node.config.get("in_place"):
            return input_handle
        out = (report or {}).get("output_dir") if isinstance(report, Mapping) else None
        if not out:
            out = node.config.get("output_dir")
        if not out and input_handle is not None:
            out = str(Path(input_handle.path) / "cleaned")
        return _inherit(out or "", input_handle)

    if node_type in ("prep.tag", "prep.caption", "prep.quality", "prep.index"):
        # ``prep.quality``'s report["output_dir"] is the QUARANTINE folder
        # (``rengu_flow/prep/quality.py:179``), not the result: the surviving dataset is still the
        # input folder. Reading it here would make ``quality -> caption`` caption the reject pile.
        return input_handle

    if node_type == "tool":
        if report is None:
            return input_handle  # in-place mutation or a pure side effect
        if isinstance(report, str):
            return _inherit(report, input_handle)
        if isinstance(report, Mapping) and report.get("path"):
            return _inherit(str(report["path"]), input_handle, report)
        raise NodeOutputError(TOOL_RETURN_ERROR)

    # ``train`` is terminal and fire-and-forget: it emits nothing.
    return None
