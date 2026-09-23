"""Edit datasets in a workflow: ``control_path`` on the handle and the ``prep.edit_caption`` node.

``control_path`` is a property of the dataset, like the caption layout: set once on the source
folder, carried down the chain on the handle, and injected into the one stage that reads it. The
rules pinned here:

* the handle **inherits** ``control_path`` through every pass-through step, and a handle saved
  before the field existed compares equal to today's (no amber ring on upgrade);
* in ``prep.edit_caption`` the handle's control folder **wins** over the node's own field, which
  is only the fallback — pre-flight and the launch read the same rule;
* pre-flight refuses an edit-instruction step with no control folder from either side, and the
  two steps that would break a pair (``clean``; ``quality`` with ``move``) on an edit dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import toml

from rengu_flow_ui import workflow_db
from rengu_flow_ui import workflow_graph as wg
from rengu_flow_ui import workflow_nodes as wn
from rengu_flow_ui import workflow_runner as wr
from rengu_flow_ui.workflow_graph import DatasetHandle, WorkflowGraph, WorkflowNode


def _node(node_id: str, node_type: str, **kwargs) -> WorkflowNode:
    return WorkflowNode(id=node_id, type=node_type, **kwargs)


@pytest.fixture
def pairs(tmp_path: Path) -> tuple[Path, Path]:
    targets, controls = tmp_path / "targets", tmp_path / "controls"
    targets.mkdir()
    controls.mkdir()
    return targets, controls


def _edit_graph(targets: Path, controls: Path | None, *middle: WorkflowNode, **edit) -> WorkflowGraph:
    folder = {"path": str(targets)}
    if controls is not None:
        folder["control_path"] = str(controls)
    last = middle[-1].id if middle else "n1"
    return WorkflowGraph(
        nodes=[
            _node("n1", "folder", config=folder),
            *middle,
            _node("n9", "prep.edit_caption", source=last, config=dict(edit)),
        ]
    )


# ------------------------------------------------------------------------------ the handle


def test_to_dict_omits_an_empty_control_path_and_keeps_a_set_one() -> None:
    assert wg.DatasetHandle("D:/x").to_dict() == {
        "path": "D:/x",
        "caption_format": "sidecar",
        "caption_ext": ".txt",
    }
    assert wg.DatasetHandle("D:/x", control_path="D:/c").to_dict()["control_path"] == "D:/c"


def test_a_handle_saved_before_control_path_existed_is_not_stale() -> None:
    old = {"path": "D:/x", "caption_format": "sidecar", "caption_ext": ".txt"}
    assert wg._handle_key(old) == wg._handle_key(wg.DatasetHandle("D:/x").to_dict())
    assert wg._handle_key(old) == wg._handle_key({**old, "control_path": ""})
    assert wg._handle_key(old) != wg._handle_key({**old, "control_path": "D:/c"})


def test_changing_the_control_folder_marks_the_consumer_stale() -> None:
    graph = WorkflowGraph(
        nodes=[
            _node("n1", "folder", config={"path": "D:/x"}),
            _node("n2", "prep.tag", source="n1", config={"models": ["pixai-v0.9"]}),
        ]
    )
    hashes = {}
    for node in graph.nodes:
        parent = hashes.get(node.source, "") if node.source else ""
        hashes[node.id] = wg.node_config_hash(node, parent, graph.variables)
    ran_with = {"path": "D:/x", "caption_format": "sidecar", "caption_ext": ".txt"}
    state = {
        "n1": {"config_hash": hashes["n1"], "output": {**ran_with, "control_path": "D:/new"}},
        "n2": {"config_hash": hashes["n2"], "output": ran_with, "saved_input": ran_with},
    }
    assert wg.compute_stale(graph, state)["n2"] is True


def test_folder_emits_its_control_path() -> None:
    node = _node("n1", "folder", config={"path": "D:/t", "control_path": " D:/c "})
    assert wg.effective_output(node) == DatasetHandle("D:/t", control_path="D:/c")


@pytest.mark.parametrize(
    "node_type, config",
    [
        ("prep.tag", {}),
        ("prep.caption", {}),
        ("prep.edit_caption", {}),
        ("prep.quality", {"action": "move"}),
        ("prep.index", {}),
        ("tool", {}),
    ],
)
def test_pass_through_steps_keep_the_control_path(node_type: str, config: dict) -> None:
    handle = DatasetHandle("D:/t", control_path="D:/c")
    assert wg.effective_output(_node("n2", node_type, source="n1", config=config), handle) == handle


def test_a_new_folder_inherits_the_control_path_unless_a_tool_overrides_it() -> None:
    handle = DatasetHandle("D:/t", control_path="D:/c")
    clean = _node("n2", "prep.clean", source="n1", config={"output_dir": "D:/out"})
    assert wg.effective_output(clean, handle).control_path == "D:/c"
    tool = _node("n2", "tool", source="n1")
    assert wg.effective_output(tool, handle, "D:/exported").control_path == "D:/c"
    assert wg.effective_output(tool, handle, {"path": "D:/e", "control_path": "D:/c2"}).control_path == "D:/c2"


def test_handle_from_dict_reads_the_control_path() -> None:
    assert wg.handle_from_dict({"path": "D:/t", "control_path": "D:/c"}) == DatasetHandle(
        "D:/t", control_path="D:/c"
    )
    assert wg.handle_from_dict({"path": ""}) is None


def test_the_runner_hands_the_saved_control_path_to_the_next_node() -> None:
    state = {"nodes": {"n1": {"output": {"path": "D:/t", "control_path": "D:/c"}}}}
    assert wr._input_handle(state, "n1") == DatasetHandle("D:/t", control_path="D:/c")


# ------------------------------------------------------------------------------ the node type


def test_edit_caption_is_a_gpu_prep_step_that_emits_its_input() -> None:
    spec = wg.NODE_TYPES["prep.edit_caption"]
    assert (spec.consumes, spec.emits, spec.needs_gpu, spec.source_optional) == (
        True,
        True,
        True,
        False,
    )
    assert wg.default_needs_gpu("prep.edit_caption") is True


# ------------------------------------------------------------------------------ the launch


@pytest.fixture
def node_dir(ui_data_tmp: Path) -> Path:
    return workflow_db.node_dir(1, "n2")


def test_the_handles_control_path_wins_over_the_nodes_own(
    node_dir: Path, pairs: tuple[Path, Path], tmp_path: Path
) -> None:
    targets, controls = pairs
    stale = tmp_path / "stale"
    stale.mkdir()
    node = _node("n2", "prep.edit_caption", config={"control_path": str(stale)})
    wn.build_launch(node, DatasetHandle(str(targets), control_path=str(controls)), node_dir)

    written = toml.loads((node_dir / "prep.toml").read_text(encoding="utf-8"))
    assert written["path"] == str(targets)
    assert written["edit_caption"]["control_path"] == str(controls)
    assert "control_path" not in written  # a stage key, never a top-level one


def test_the_nodes_control_path_is_the_fallback(node_dir: Path, pairs: tuple[Path, Path]) -> None:
    targets, controls = pairs
    node = _node("n2", "prep.edit_caption", config={"control_path": str(controls)})
    wn.build_launch(node, DatasetHandle(str(targets)), node_dir)
    written = toml.loads((node_dir / "prep.toml").read_text(encoding="utf-8"))
    assert written["edit_caption"]["control_path"] == str(controls)


def test_other_stages_never_see_the_control_path(node_dir: Path, pairs: tuple[Path, Path]) -> None:
    targets, controls = pairs
    node = _node("n2", "prep.tag", config={"models": ["pixai-v0.9"]})
    wn.build_launch(node, DatasetHandle(str(targets), control_path=str(controls)), node_dir)
    written = toml.loads((node_dir / "prep.toml").read_text(encoding="utf-8"))
    assert "control_path" not in written and "control_path" not in written["tag"]


def test_a_folder_with_a_missing_control_folder_fails_the_source_step(
    ui_data_tmp: Path, pairs: tuple[Path, Path], tmp_path: Path
) -> None:
    targets, _ = pairs
    node = _node("n1", "folder", config={"path": str(targets), "control_path": str(tmp_path / "no")})
    with pytest.raises(FileNotFoundError, match="Control images folder not found"):
        wn.run_inline(node, None, tmp_path / "nd")


def test_the_folder_step_records_its_control_path(
    ui_data_tmp: Path, pairs: tuple[Path, Path], tmp_path: Path
) -> None:
    targets, controls = pairs
    node = _node("n1", "folder", config={"path": str(targets), "control_path": str(controls)})
    assert wn.run_inline(node, None, tmp_path / "nd")["control_path"] == str(controls)


# ------------------------------------------------------------------------------ pre-flight


def test_an_edit_step_fed_by_a_folder_with_controls_passes(pairs: tuple[Path, Path]) -> None:
    assert wg.validate(_edit_graph(*pairs)) == []


def test_an_edit_step_reads_the_control_folder_through_intermediate_steps(
    pairs: tuple[Path, Path],
) -> None:
    quality = _node("n2", "prep.quality", source="n1", config={"metric": "blur"})
    assert wg.validate(_edit_graph(*pairs, quality)) == []


def test_an_edit_step_with_no_control_folder_anywhere_is_refused(pairs: tuple[Path, Path]) -> None:
    targets, _ = pairs
    assert wg.validate(_edit_graph(targets, None)) == [
        f"node n9 · {wg.EDIT_CAPTION_NO_CONTROLS_ERROR}"
    ]


def test_the_nodes_own_control_folder_is_enough(pairs: tuple[Path, Path]) -> None:
    targets, controls = pairs
    assert wg.validate(_edit_graph(targets, None, control_path=str(controls))) == []


def test_a_control_folder_that_does_not_exist_is_refused(
    pairs: tuple[Path, Path], tmp_path: Path
) -> None:
    targets, _ = pairs
    errors = wg.validate(_edit_graph(targets, tmp_path / "missing"))
    assert len(errors) == 1
    assert errors[0].startswith("node n9 · Control folder not found")


def test_the_control_folder_of_a_variable_is_resolved(pairs: tuple[Path, Path]) -> None:
    targets, controls = pairs
    graph = _edit_graph(targets, None)
    graph.nodes[0].config["control_path"] = "${controls}"
    graph.variables = [wg.Variable(name="controls", value=str(controls))]
    assert wg.validate(graph) == []


def test_clean_on_an_edit_dataset_is_refused(pairs: tuple[Path, Path]) -> None:
    for config in ({"in_place": True}, {"in_place": False}):
        clean = _node("n2", "prep.clean", source="n1", config=config)
        errors = wg.validate(_edit_graph(*pairs, clean))
        assert len(errors) == 1, errors
        assert errors[0].startswith("node n2 · Clean on an edit dataset")


def test_clean_on_a_plain_dataset_is_fine(pairs: tuple[Path, Path]) -> None:
    targets, _ = pairs
    graph = WorkflowGraph(
        nodes=[
            _node("n1", "folder", config={"path": str(targets)}),
            _node("n2", "prep.clean", source="n1", config={"in_place": True}),
        ]
    )
    assert wg.validate(graph) == []


def test_a_moving_quality_filter_on_an_edit_dataset_is_refused(pairs: tuple[Path, Path]) -> None:
    quality = _node("n2", "prep.quality", source="n1", config={"action": "move"})
    errors = wg.validate(_edit_graph(*pairs, quality))
    assert len(errors) == 1, errors
    assert errors[0].startswith("node n2 · Quality filter with 'move' on an edit dataset")


def test_a_reporting_quality_filter_on_an_edit_dataset_is_fine(pairs: tuple[Path, Path]) -> None:
    quality = _node("n2", "prep.quality", source="n1", config={"action": "report"})
    assert wg.validate(_edit_graph(*pairs, quality)) == []


def test_a_disabled_source_is_judged_on_its_saved_output(pairs: tuple[Path, Path]) -> None:
    targets, controls = pairs
    graph = _edit_graph(targets, None)
    graph.nodes[0].enabled = False
    saved = {"n1": {"output": {"path": str(targets), "control_path": str(controls)}}}
    assert wg.validate(graph, saved) == []
    assert wg.validate(graph, {"n1": {"output": {"path": str(targets)}}}) == [
        f"node n9 · {wg.EDIT_CAPTION_NO_CONTROLS_ERROR}"
    ]


def test_the_saved_state_json_round_trips_the_control_path() -> None:
    """``state_json`` is JSON: what ``to_dict`` writes is what ``handle_from_dict`` reads."""
    handle = DatasetHandle("D:/t", "json", ".cap", "D:/c")
    assert wg.handle_from_dict(json.loads(json.dumps(handle.to_dict()))) == handle
