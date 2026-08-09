"""The workflow graph: tolerant parsing, the ``from`` invariant, variables, hashing, outputs.

No DB and no filesystem fixtures on purpose — ``workflow_graph`` is pure, and that is precisely
what makes the trickiest rules in the feature (the ``output_dir`` trap, single-pass variable
resolution, the hash chain) cheap to pin down.
"""

from __future__ import annotations

import pytest

from rengu_flow_ui import workflow_graph as wg


def _node(node_id: str, node_type: str, **kwargs) -> wg.WorkflowNode:
    return wg.WorkflowNode(id=node_id, type=node_type, **kwargs)


def _handle(path: str = "D:/datasets/aoi", **kwargs) -> wg.DatasetHandle:
    return wg.DatasetHandle(path=path, **kwargs)


GRAPH_JSON = {
    "version": 1,
    "name": "Re-tag character set",
    "description": "",
    "variables": [
        {"name": "dataset_dir", "value": "D:/datasets/aoi", "description": "Folder to process"}
    ],
    "nodes": [
        {
            "id": "n1",
            "type": "folder",
            "title": "Source folder",
            "from": None,
            "enabled": True,
            "config": {"path": "${dataset_dir}", "caption_format": "sidecar", "caption_ext": ".txt"},
            "gpu": {"required": False, "wait": True, "device": None},
        },
        {
            "id": "n2",
            "type": "prep.tag",
            "title": "Tag",
            "from": "n1",
            "enabled": True,
            "config": {"models": ["pixai-v0.9"], "max_tags": 255},
            "gpu": {"required": True, "wait": True, "device": 0},
        },
    ],
}


# ------------------------------------------------------------------------------ parsing


def test_round_trip_is_exact() -> None:
    graph = wg.parse_graph(GRAPH_JSON)
    assert wg.graph_to_dict(graph) == GRAPH_JSON
    assert wg.parse_graph(wg.graph_to_dict(graph)) == graph


def test_from_maps_to_source_in_both_directions() -> None:
    graph = wg.parse_graph(GRAPH_JSON)
    assert graph.nodes[0].source is None
    assert graph.nodes[1].source == "n1"
    assert "source" not in wg.graph_to_dict(graph)["nodes"][1]
    assert wg.graph_to_dict(graph)["nodes"][1]["from"] == "n1"


def test_a_stray_source_key_is_ignored_from_is_the_only_door() -> None:
    graph = wg.parse_graph({"nodes": [{"id": "n1", "type": "folder", "source": "n0"}]})
    assert graph.nodes[0].source is None


def test_unknown_node_type_is_preserved_on_parse() -> None:
    """A downgrade of the app must never destroy the user's graph."""
    data = {"nodes": [{"id": "n1", "type": "prep.frobnicate", "from": None, "config": {"a": 1}}]}
    graph = wg.parse_graph(data)
    assert graph.nodes[0].type == "prep.frobnicate"
    assert graph.nodes[0].config == {"a": 1}
    assert wg.graph_to_dict(graph)["nodes"][0]["type"] == "prep.frobnicate"


def test_unknown_keys_are_dropped_never_fatal() -> None:
    graph = wg.parse_graph(
        {
            "version": 1,
            "unknown_top": 1,
            "variables": [{"name": "a", "value": "1", "nope": True}],
            "nodes": [
                {"id": "n1", "type": "folder", "surprise": [1, 2], "gpu": {"required": True, "x": 1}}
            ],
        }
    )
    assert graph.variables == [wg.Variable(name="a", value="1")]
    assert graph.nodes[0].id == "n1"
    assert graph.nodes[0].gpu == wg.NodeGpu(required=True, wait=True, device=None)


def test_malformed_pieces_are_skipped_not_raised() -> None:
    graph = wg.parse_graph(
        {"variables": "nope", "nodes": ["nope", {"id": "n1", "type": "folder", "config": 7}]}
    )
    assert graph.variables == []
    assert len(graph.nodes) == 1
    assert graph.nodes[0].config == {}


def test_empty_graph_parses_to_defaults() -> None:
    graph = wg.parse_graph({})
    assert graph == wg.WorkflowGraph()
    assert wg.validate(graph) == []


# ------------------------------------------------------------------------------ validation


def test_valid_graph_has_no_errors() -> None:
    assert wg.validate(wg.parse_graph(GRAPH_JSON)) == []


def test_from_pointing_forward_is_rejected() -> None:
    graph = wg.WorkflowGraph(
        nodes=[_node("n1", "prep.tag", source="n2"), _node("n2", "folder")]
    )
    errors = wg.validate(graph)
    assert len(errors) == 1
    assert "node n1" in errors[0] and "points forward" in errors[0]


def test_from_pointing_at_itself_is_rejected() -> None:
    graph = wg.WorkflowGraph(nodes=[_node("n1", "folder"), _node("n2", "prep.tag", source="n2")])
    errors = wg.validate(graph)
    assert len(errors) == 1
    assert "points at itself" in errors[0]


def test_from_pointing_at_a_missing_node_is_rejected() -> None:
    graph = wg.WorkflowGraph(nodes=[_node("n1", "prep.tag", source="gone")])
    assert wg.validate(graph) == ["node n1 · from 'gone' is not a node in this workflow"]


def test_missing_source_is_rejected_except_on_folder_and_tool() -> None:
    """Deleting a ``folder`` splices its children onto its own null ``from``."""
    graph = wg.WorkflowGraph(
        nodes=[_node("n1", "folder"), _node("n2", "tool"), _node("n3", "prep.clean")]
    )
    errors = wg.validate(graph)
    assert len(errors) == 1
    assert errors[0] == "node n3 · has no source; only folder and tool nodes may have none"


def test_unknown_node_type_is_an_error_at_validation() -> None:
    graph = wg.WorkflowGraph(nodes=[_node("n1", "prep.frobnicate")])
    assert wg.validate(graph) == ["node n1 · unknown node type 'prep.frobnicate'"]


def test_unknown_variable_is_reported_with_node_and_field() -> None:
    graph = wg.WorkflowGraph(
        nodes=[
            _node("n1", "folder", config={"path": "D:/x"}),
            _node("n3", "prep.quality", source="n1", config={"output_dir": "${outdir}"}),
        ]
    )
    assert wg.validate(graph) == ["node n3 · quality.output_dir → unknown variable ${outdir}"]


def test_unknown_variable_is_found_inside_lists_and_nested_dicts() -> None:
    graph = wg.WorkflowGraph(
        variables=[wg.Variable(name="known", value="x")],
        nodes=[
            _node(
                "n1",
                "folder",
                config={"models": ["${known}", "${missing}"], "deep": {"k": "${gone}"}},
            )
        ],
    )
    errors = wg.validate(graph)
    assert "node n1 · folder.models[1] → unknown variable ${missing}" in errors
    assert "node n1 · folder.deep.k → unknown variable ${gone}" in errors
    assert len(errors) == 2


def test_duplicate_id_is_rejected() -> None:
    graph = wg.WorkflowGraph(nodes=[_node("n1", "folder"), _node("n1", "folder")])
    assert wg.validate(graph) == ["node n1 · duplicate node id"]


def test_every_error_is_reported_at_once() -> None:
    """Pre-flight promises the whole list up front, not the first failure."""
    graph = wg.WorkflowGraph(
        nodes=[
            _node("n1", "prep.clean"),  # no source
            _node("n2", "nope.nope", source="n1"),  # unknown type
            _node("n2", "prep.tag", source="n1", config={"prepend_tags": ["${x}"]}),  # dup + var
        ]
    )
    errors = wg.validate(graph)
    assert len(errors) == 4
    assert any("has no source" in e for e in errors)
    assert any("unknown node type" in e for e in errors)
    assert any("duplicate node id" in e for e in errors)
    assert any("unknown variable ${x}" in e for e in errors)


# ------------------------------------------------------------------------------ variables


def test_resolve_text_substitutes_known_names() -> None:
    assert wg.resolve_text("${a}/sub", {"a": "D:/x"}) == "D:/x/sub"
    assert wg.resolve_text("${a}${a}", {"a": "z"}) == "zz"


def test_resolve_text_escapes_double_dollar() -> None:
    assert wg.resolve_text("$$", {}) == "$"
    assert wg.resolve_text("$${a}", {"a": "z"}) == "${a}"
    assert wg.resolve_text("cost: $$5 and ${a}", {"a": "z"}) == "cost: $5 and z"


def test_resolve_text_is_a_single_pass_with_no_recursion() -> None:
    assert wg.resolve_text("${a}", {"a": "${b}", "b": "deep"}) == "${b}"


def test_missing_variable_is_left_literal_never_emptied() -> None:
    assert wg.resolve_text("${nope}/x", {"a": "1"}) == "${nope}/x"


def test_resolve_text_ignores_malformed_tokens() -> None:
    assert wg.resolve_text("${1bad} $notavar {a}", {"a": "z"}) == "${1bad} $notavar {a}"


def test_resolve_text_accepts_the_graphs_variable_list() -> None:
    variables = [wg.Variable(name="a", value="D:/x")]
    assert wg.resolve_text("${a}", variables) == "D:/x"


def test_resolve_config_touches_strings_only() -> None:
    node = _node(
        "n1",
        "prep.quality",
        config={
            "output_dir": "${dir}/q",
            "blur_threshold": 80.0,
            "min_side": 0,
            "action": "report",
            "flag": True,
            "nested": {"a": ["${dir}", 3, False, None]},
        },
    )
    resolved = wg.resolve_config(node, {"dir": "D:/x"})
    assert resolved["output_dir"] == "D:/x/q"
    assert resolved["blur_threshold"] == 80.0
    assert resolved["min_side"] == 0
    assert resolved["flag"] is True
    assert resolved["nested"] == {"a": ["D:/x", 3, False, None]}
    # And the node itself is untouched.
    assert node.config["output_dir"] == "${dir}/q"


def test_collect_refs_maps_variables_to_their_places() -> None:
    graph = wg.WorkflowGraph(
        variables=[wg.Variable(name="dataset_dir", value="D:/x")],
        nodes=[
            _node("n1", "folder", config={"path": "${dataset_dir}"}),
            _node(
                "n2",
                "prep.quality",
                source="n1",
                config={"output_dir": "${dataset_dir}/rejects", "iqa_model": "${missing}"},
            ),
        ],
    )
    assert wg.collect_refs(graph) == {
        "dataset_dir": ["n1 · folder.path", "n2 · quality.output_dir"],
        "missing": ["n2 · quality.iqa_model"],
    }


# ------------------------------------------------------------------------------ catalog


def test_quality_gpu_default_follows_the_metric() -> None:
    """``blur`` is a dep-free CPU Laplacian; ``aesthetic``/``iqa`` load models."""
    assert wg.default_needs_gpu("prep.quality", {}) is False
    assert wg.default_needs_gpu("prep.quality", {"metric": "blur"}) is False
    assert wg.default_needs_gpu("prep.quality", {"metric": "aesthetic"}) is True
    assert wg.default_needs_gpu("prep.quality", {"metric": "iqa"}) is True


def test_gpu_defaults_of_the_other_types() -> None:
    assert wg.default_needs_gpu("folder") is False
    assert wg.default_needs_gpu("prep.tag") is True
    assert wg.default_needs_gpu("prep.caption") is True
    assert wg.default_needs_gpu("prep.clean") is True
    assert wg.default_needs_gpu("prep.index") is True
    assert wg.default_needs_gpu("tool") is False
    assert wg.default_needs_gpu("train") is False
    assert wg.default_needs_gpu("prep.frobnicate") is False


def test_catalog_covers_the_eight_types() -> None:
    assert set(wg.NODE_TYPES) == {
        "folder",
        "prep.tag",
        "prep.caption",
        "prep.clean",
        "prep.quality",
        "prep.index",
        "tool",
        "train",
    }
    assert wg.NODE_TYPES["folder"].consumes is False
    assert wg.NODE_TYPES["train"].emits is False
    assert [t for t, spec in wg.NODE_TYPES.items() if spec.source_optional] == ["folder", "tool"]


# ------------------------------------------------------------------------------ execution order


def test_execution_order_is_list_order_minus_disabled() -> None:
    graph = wg.WorkflowGraph(
        nodes=[
            _node("n1", "folder"),
            _node("n2", "prep.tag", source="n1", enabled=False),
            _node("n3", "prep.caption", source="n1"),
        ]
    )
    assert [n.id for n in wg.execution_order(graph)] == ["n1", "n3"]


# ------------------------------------------------------------------------------ hashing


def test_hash_is_stable_across_key_order() -> None:
    a = _node("n1", "prep.tag", config={"max_tags": 200, "batch_size": 8})
    b = _node("n1", "prep.tag", config={"batch_size": 8, "max_tags": 200})
    assert wg.node_config_hash(a) == wg.node_config_hash(b)


def test_hash_is_over_the_materialized_config_not_the_stored_dict() -> None:
    """A form that gains a field must not turn every saved node amber on upgrade day."""
    from dataclasses import asdict

    from rengu_flow.prep.config import TagStageConfig

    partial = _node("n1", "prep.tag", config={"max_tags": 200})
    full = _node("n1", "prep.tag", config={**asdict(TagStageConfig()), "max_tags": 200})
    assert wg.node_config_hash(partial) == wg.node_config_hash(full)


def test_hash_ignores_keys_the_stage_does_not_know() -> None:
    plain = _node("n1", "prep.tag", config={"max_tags": 200})
    noisy = _node("n1", "prep.tag", config={"max_tags": 200, "leftover_from_v9": True})
    assert wg.node_config_hash(plain) == wg.node_config_hash(noisy)


def test_hash_changes_with_config_gpu_type_and_parent() -> None:
    base = _node("n1", "prep.tag", config={"max_tags": 200})
    reference = wg.node_config_hash(base, "parent")

    assert wg.node_config_hash(_node("n1", "prep.tag", config={"max_tags": 201}), "parent") != reference
    assert wg.node_config_hash(_node("n1", "prep.caption", config={"max_tags": 200}), "parent") != reference
    assert wg.node_config_hash(base, "other-parent") != reference
    gpu = _node("n1", "prep.tag", config={"max_tags": 200}, gpu=wg.NodeGpu(device=1))
    assert wg.node_config_hash(gpu, "parent") != reference
    # Cosmetics do not: id and title are not part of the recipe.
    twin = _node("n9", "prep.tag", title="renamed", config={"max_tags": 200})
    assert wg.node_config_hash(twin, "parent") == reference


def test_hash_resolves_variables_before_hashing() -> None:
    node = _node("n1", "prep.quality", config={"output_dir": "${dir}"})
    assert wg.node_config_hash(node, "", {"dir": "a"}) != wg.node_config_hash(node, "", {"dir": "b"})


def test_a_parent_change_changes_the_childs_hash() -> None:
    """Downstream invalidation is free — there is no propagation code."""
    child = _node("n2", "prep.tag", source="n1", config={"max_tags": 200})
    parent_a = wg.node_config_hash(_node("n1", "folder", config={"path": "a"}))
    parent_b = wg.node_config_hash(_node("n1", "folder", config={"path": "b"}))
    assert parent_a != parent_b
    assert wg.node_config_hash(child, parent_a) != wg.node_config_hash(child, parent_b)


def test_hash_of_a_non_prep_type_uses_the_config_as_written() -> None:
    node = _node("n1", "tool", config={"path": "${dir}", "count": 3})
    assert wg.materialize_config(node, {"dir": "D:/x"}) == {"path": "D:/x", "count": 3}


# ------------------------------------------------------------------------------ staleness


def _fresh_state(graph: wg.WorkflowGraph, outputs: dict[str, str | None]) -> dict[str, dict]:
    """State in which every node in *outputs* ran with its current config and input.

    A ``None`` output means "ran, emitted nothing" — the shape a terminal ``train`` node leaves.
    """
    state: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for node in graph.nodes:
        parent = hashes.get(node.source, "") if node.source else ""
        hashes[node.id] = wg.node_config_hash(node, parent, graph.variables)
        if node.id not in outputs:
            continue
        upstream = state.get(node.source, {}).get("output") if node.source else None
        path = outputs[node.id]
        state[node.id] = {
            "status": "done",
            "output": (
                {"path": path, "caption_format": "sidecar", "caption_ext": ".txt"}
                if path is not None
                else None
            ),
            "saved_input": upstream,
            "config_hash": hashes[node.id],
        }
    return state


def _chain() -> wg.WorkflowGraph:
    return wg.WorkflowGraph(
        variables=[wg.Variable(name="dataset_dir", value="D:/x")],
        nodes=[
            _node("n1", "folder", config={"path": "${dataset_dir}"}),
            _node("n2", "prep.tag", source="n1", config={"max_tags": 200}),
            _node("n3", "prep.caption", source="n2", config={"batch_size": 4}),
        ],
    )


def test_nothing_saved_is_never_stale() -> None:
    graph = _chain()
    assert wg.compute_stale(graph, {}) == {"n1": False, "n2": False, "n3": False}
    assert wg.compute_stale(graph, None) == {"n1": False, "n2": False, "n3": False}


def test_a_fresh_chain_is_not_stale() -> None:
    graph = _chain()
    state = _fresh_state(graph, {"n1": "D:/x", "n2": "D:/x", "n3": "D:/x"})
    assert wg.compute_stale(graph, state) == {"n1": False, "n2": False, "n3": False}


def test_editing_a_node_marks_it_and_everything_downstream_stale() -> None:
    graph = _chain()
    state = _fresh_state(graph, {"n1": "D:/x", "n2": "D:/x", "n3": "D:/x"})
    graph.nodes[1].config["max_tags"] = 255
    assert wg.compute_stale(graph, state) == {"n1": False, "n2": True, "n3": True}


def test_editing_a_variable_marks_the_whole_chain_stale() -> None:
    graph = _chain()
    state = _fresh_state(graph, {"n1": "D:/x", "n2": "D:/x", "n3": "D:/x"})
    graph.variables[0].value = "D:/other"
    assert wg.compute_stale(graph, state) == {"n1": True, "n2": True, "n3": True}


def test_a_changed_upstream_output_marks_the_consumer_stale() -> None:
    """What the config chain cannot see: a tool that returns a computed folder."""
    graph = _chain()
    state = _fresh_state(graph, {"n1": "D:/x", "n2": "D:/x", "n3": "D:/x"})
    state["n2"]["output"] = {"path": "D:/export-2", "caption_format": "sidecar", "caption_ext": ".txt"}
    stale = wg.compute_stale(graph, state)
    assert stale["n2"] is False  # its own config and input are unchanged
    assert stale["n3"] is True  # it consumed the old folder


def _chain_into_training() -> wg.WorkflowGraph:
    """The spec's rector case: ``folder(${dataset_dir}) -> prep.tag -> train``."""
    graph = _chain()
    graph.nodes = graph.nodes[:2] + [_node("n3", "train", source="n2", config={"job_id": 7})]
    return graph


def test_a_terminal_node_that_ran_goes_stale_like_any_other() -> None:
    """``train`` emits nothing, so "has a saved output" would keep it fresh forever.

    The user changes ``${dataset_dir}`` and presses Run: the tag stage re-labels the new folder and
    the training must be queued again. Judged on its output, a ``done`` train node is never stale,
    ``_plan`` drops it, and the card stays green with yesterday's ``job_id``.
    """
    graph = _chain_into_training()
    state = _fresh_state(graph, {"n1": "D:/x", "n2": "D:/x", "n3": None})
    assert wg.compute_stale(graph, state)["n3"] is False

    graph.variables[0].value = "D:/other"

    assert wg.compute_stale(graph, state)["n3"] is True


def test_a_terminal_node_that_never_ran_is_not_stale() -> None:
    """"Never ran" is still the thing that is never stale — there is nothing to distrust yet."""
    graph = _chain_into_training()
    state = _fresh_state(graph, {"n1": "D:/x", "n2": "D:/x"})
    graph.variables[0].value = "D:/other"

    assert wg.compute_stale(graph, state)["n3"] is False


def test_a_node_with_no_saved_output_upstream_is_judged_on_its_saved_input() -> None:
    graph = _chain()
    state = _fresh_state(graph, {"n2": "D:/x", "n3": "D:/x"})
    # n2 recorded ``saved_input: None`` because n1 had no output; still true now.
    assert wg.compute_stale(graph, state)["n2"] is False


# ------------------------------------------------------------------------------ effective_output


def test_folder_emits_its_literal_config() -> None:
    node = _node("n1", "folder", config={"path": "D:/x", "caption_format": "json", "caption_ext": ".cap"})
    assert wg.effective_output(node) == wg.DatasetHandle("D:/x", "json", ".cap")


def test_folder_falls_back_to_the_handle_defaults() -> None:
    assert wg.effective_output(_node("n1", "folder", config={"path": "D:/x"})) == wg.DatasetHandle(
        "D:/x", "sidecar", ".txt"
    )


@pytest.mark.parametrize("node_type", ["prep.tag", "prep.caption", "prep.index"])
def test_in_place_prep_stages_pass_the_handle_through(node_type: str) -> None:
    handle = _handle(caption_format="json", caption_ext=".cap")
    assert wg.effective_output(_node("n2", node_type, source="n1"), handle, {"tagged": 3}) is handle


def test_quality_emits_its_input_never_its_output_dir() -> None:
    """``quality``'s ``output_dir`` is the QUARANTINE folder — the survivors stay in the input.

    Reading it generically would make ``quality -> caption`` caption the reject pile.
    """
    handle = _handle("D:/datasets/aoi")
    node = _node("n2", "prep.quality", source="n1", config={"action": "move", "output_dir": "D:/junk"})
    report = {"metric": "blur", "flagged": 4, "output_dir": "D:/datasets/aoi/low_quality"}
    assert wg.effective_output(node, handle, report) is handle


def test_clean_in_place_emits_the_input() -> None:
    handle = _handle("D:/x")
    node = _node("n2", "prep.clean", source="n1", config={"in_place": True})
    report = {"cleaned": 2, "output_dir": None}
    assert wg.effective_output(node, handle, report) is handle


def test_clean_reads_output_dir_from_the_report() -> None:
    """``clean`` is the ONLY stage whose ``report['output_dir']`` names the result."""
    handle = _handle("D:/x", caption_format="json", caption_ext=".cap")
    node = _node("n2", "prep.clean", source="n1", config={"in_place": False})
    report = {"cleaned": 2, "output_dir": "D:/x/cleaned"}
    assert wg.effective_output(node, handle, report) == wg.DatasetHandle("D:/x/cleaned", "json", ".cap")


def test_clean_predicts_its_output_before_it_runs() -> None:
    handle = _handle("D:/x")
    configured = _node("n2", "prep.clean", source="n1", config={"output_dir": "D:/out"})
    assert wg.effective_output(configured, handle).path == "D:/out"

    default = _node("n2", "prep.clean", source="n1", config={})
    assert wg.effective_output(default, handle).path.replace("\\", "/") == "D:/x/cleaned"


def test_tool_returning_a_string_is_a_path_with_inherited_format() -> None:
    handle = _handle("D:/x", caption_format="json", caption_ext=".cap")
    node = _node("n2", "tool", source="n1")
    assert wg.effective_output(node, handle, "D:/exported") == wg.DatasetHandle(
        "D:/exported", "json", ".cap"
    )


def test_tool_returning_a_dict_wins_on_the_keys_it_supplies() -> None:
    handle = _handle("D:/x", caption_format="json", caption_ext=".cap")
    node = _node("n2", "tool", source="n1")
    result = {"path": "D:/exported", "caption_ext": ".txt"}
    assert wg.effective_output(node, handle, result) == wg.DatasetHandle(
        "D:/exported", "json", ".txt"
    )


def test_tool_returning_none_passes_through() -> None:
    handle = _handle("D:/x")
    assert wg.effective_output(_node("n2", "tool", source="n1"), handle, None) is handle


def test_tool_returning_anything_else_fails_the_node() -> None:
    node = _node("n2", "tool", source="n1")
    for bad in (42, ["D:/x"], True, {"no_path": 1}):
        with pytest.raises(wg.NodeOutputError) as excinfo:
            wg.effective_output(node, _handle(), bad)
        assert str(excinfo.value) == wg.TOOL_RETURN_ERROR


def test_train_is_terminal() -> None:
    assert wg.effective_output(_node("n9", "train", source="n1"), _handle(), None) is None


def test_an_unknown_type_is_fatal_at_execution() -> None:
    with pytest.raises(wg.NodeOutputError, match="Unknown node type"):
        wg.effective_output(_node("n1", "prep.frobnicate"), _handle())
