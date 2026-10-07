"""Workflow steps' own caption output format, and the I/O a Toolbox tool declares.

* A tag / caption / edit_caption node may carry ``output_format`` (``inherit`` | ``sidecar`` |
  ``json``): ``effective_output`` emits the new layout, ``_prep_payload`` makes the prep runner
  convert the folder first, and a node that never set it hashes **exactly** as before.
* A tool's ``tool.json`` may declare ``io``; validation refuses a step that reads from a tool that
  hands nothing on, and the run fails a tool that promised a folder and returned none.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import toml

from rengu_flow.prep.config import EditCaptionStageConfig, TagStageConfig
from rengu_flow_ui import toolbox, workflow_db
from rengu_flow_ui import workflow_graph as wg
from rengu_flow_ui import workflow_nodes as wn
from rengu_flow_ui.workflow_graph import DatasetHandle, NodeOutputError, WorkflowNode


def _node(node_id: str, node_type: str, **kwargs) -> WorkflowNode:
    return WorkflowNode(id=node_id, type=node_type, **kwargs)


def _graph(*nodes: WorkflowNode) -> wg.WorkflowGraph:
    return wg.WorkflowGraph(nodes=list(nodes))


# ------------------------------------------------------------------------------ hashing


def _old_recipe_hash(node_type: str, section: dict) -> str:
    """The hash recipe as it was before these fields existed: the stage dataclass, whole."""
    payload = {
        "hash_version": wg.HASH_VERSION,
        "type": node_type,
        "config": section,
        "gpu": {"required": False, "wait": True, "device": None},
        "parent": "",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("node_type", "dataclass", "late"),
    [
        ("prep.tag", TagStageConfig, ("write_mode",)),
        ("prep.edit_caption", EditCaptionStageConfig, ("write_mode", "target_line")),
    ],
)
def test_nodes_that_never_set_the_new_keys_hash_as_before(node_type, dataclass, late) -> None:
    before = {k: v for k, v in asdict(dataclass()).items() if k not in late}
    plain = _node("n", node_type, config={})
    explicit_defaults = _node("n", node_type, config={"output_format": "inherit", "write_mode": ""})
    assert wg.node_config_hash(plain) == _old_recipe_hash(node_type, before)
    assert wg.node_config_hash(explicit_defaults) == wg.node_config_hash(plain)


def test_choosing_an_output_format_or_write_mode_changes_the_hash() -> None:
    base = wg.node_config_hash(_node("n", "prep.tag", config={}))
    as_json = wg.node_config_hash(_node("n", "prep.tag", config={"output_format": "json"}))
    as_caption = wg.node_config_hash(
        _node("n", "prep.tag", config={"output_format": "sidecar", "output_ext": ".caption"})
    )
    appended = wg.node_config_hash(_node("n", "prep.tag", config={"write_mode": "append"}))
    assert len({base, as_json, as_caption, appended}) == 4


# ------------------------------------------------------------------------------ output rule


SIDECAR = DatasetHandle(path="/data/a")
JSON = DatasetHandle(path="/data/a", caption_format="json")


@pytest.mark.parametrize("node_type", ["prep.tag", "prep.caption", "prep.edit_caption"])
@pytest.mark.parametrize(
    ("config", "incoming", "expected"),
    [
        ({}, SIDECAR, SIDECAR),  # inherit: the input handle, unchanged
        ({"output_format": "inherit"}, JSON, JSON),
        ({"output_format": "json"}, SIDECAR, DatasetHandle("/data/a", "json", ".txt")),
        ({"output_format": "sidecar"}, JSON, DatasetHandle("/data/a", "sidecar", ".txt")),
        (
            {"output_format": "sidecar", "output_ext": "caption"},
            SIDECAR,
            DatasetHandle("/data/a", "sidecar", ".caption"),
        ),
        ({"output_format": "bogus"}, SIDECAR, SIDECAR),  # unknown reads as inherit
    ],
)
def test_effective_output_carries_the_steps_output_format(
    node_type, config, incoming, expected
) -> None:
    assert wg.effective_output(_node("n", node_type, config=config), incoming) == expected


def test_a_control_folder_travels_with_a_converted_handle() -> None:
    incoming = DatasetHandle("/t", control_path="/c")
    out = wg.effective_output(_node("n", "prep.tag", config={"output_format": "json"}), incoming)
    assert out == DatasetHandle("/t", "json", ".txt", control_path="/c")


def test_a_chain_of_two_formats_predicts_each_step() -> None:
    graph = _graph(
        _node("t", "tool", config={"tool_id": "x"}),
        _node("a", "prep.tag", source="t", config={"models": ["m"], "output_format": "json"}),
        _node("b", "prep.caption", source="a", config={"output_format": "sidecar"}),
    )
    handles = wg._predicted_handles(graph, {}, {"x": {"output": "passthrough"}})
    assert handles["t"] is None  # no source, nothing to pass
    folder = _node("f", "folder", config={"path": "/d"})
    graph.nodes.insert(0, folder)
    graph.nodes[1].source = "f"
    handles = wg._predicted_handles(graph, {}, {"x": {"output": "passthrough"}})
    assert handles["a"].caption_format == "json"
    assert handles["b"].caption_format == "sidecar"


def test_validate_rejects_an_unknown_output_format() -> None:
    graph = _graph(
        _node("f", "folder", config={"path": "."}),
        _node(
            "t",
            "prep.tag",
            source="f",
            config={"models": ["pixai-v0.9"], "output_format": "yaml"},
        ),
    )
    errors = wg.validate(graph)
    assert any("output format" in e for e in errors)


# ------------------------------------------------------------------------------ _prep_payload


def test_prep_payload_keeps_output_keys_out_of_the_stage_and_converts() -> None:
    node = _node(
        "n",
        "prep.tag",
        config={"models": ["pixai-v0.9"], "output_format": "json", "output_ext": ".x"},
    )
    stage, payload = wn._prep_payload(node, DatasetHandle("/d", "sidecar", ".txt"))
    assert stage == "tag"
    assert payload["caption_format"] == "json"  # the stage runs in its own layout ...
    assert payload["convert_from_format"] == "sidecar"  # ... after converting the folder to it
    assert payload["convert_from_ext"] == ".txt"
    assert "output_format" not in payload["tag"] and "output_ext" not in payload["tag"]


def test_prep_payload_inherit_or_same_layout_does_not_convert() -> None:
    inherit = _node("n", "prep.tag", config={"models": ["m"]})
    same = _node("n", "prep.tag", config={"models": ["m"], "output_format": "json"})
    for node, handle in ((inherit, SIDECAR), (same, JSON)):
        _, payload = wn._prep_payload(node, handle)
        assert "convert_from_format" not in payload
        assert payload["caption_format"] == handle.caption_format


def test_prep_payload_passes_write_mode_and_target_line_through() -> None:
    node = _node("n", "prep.caption", config={"write_mode": "append", "target_line": 3})
    _, payload = wn._prep_payload(node, SIDECAR)
    assert payload["caption"]["write_mode"] == "append"
    assert payload["caption"]["target_line"] == 3


def test_launch_writes_a_toml_the_runner_converts_from(tmp_path: Path, dataset_dir) -> None:
    node = _node("n", "prep.tag", config={"models": ["pixai-v0.9"], "output_format": "json"})
    wn.build_launch(node, DatasetHandle(str(dataset_dir)), tmp_path / "node")
    written = toml.loads((tmp_path / "node" / "prep.toml").read_text(encoding="utf-8"))
    assert written["caption_format"] == "json"
    assert written["convert_from_format"] == "sidecar"


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "aoi"
    d.mkdir()
    return d


# ------------------------------------------------------------------------------ tool io: toolbox


def test_resolve_io_defaults_keep_todays_behaviour(ui_data_tmp) -> None:
    with_path = toolbox.create_tool(name="With path", inputs=[{"param": "path"}])
    without = toolbox.create_tool(name="Without path", inputs=[{"param": "n", "control": "number"}])
    assert with_path["io"] == {
        "input": "folder", "output": "passthrough", "input_declared": False, "output_declared": False
    }
    assert without["io"]["input"] == "none"
    assert "io" not in json.loads(
        (toolbox.tool_dir(without["id"]) / "tool.json").read_text(encoding="utf-8")
    )  # nothing declared, nothing stored


def test_declared_io_is_stored_listed_and_updatable(ui_data_tmp) -> None:
    created = toolbox.create_tool(
        name="Extract images", io={"input": "none", "output": "folder"}
    )
    assert created["io"]["output"] == "folder" and created["io"]["output_declared"]
    assert toolbox.list_tools()[0]["io"]["output"] == "folder"
    assert toolbox.get_tool(created["id"])["io"]["input"] == "none"

    toolbox.update_tool(created["id"], io={"output": "none"})
    assert toolbox.tool_io(created["id"])["output"] == "none"
    toolbox.update_tool(created["id"], io={})  # cleared: back to the defaults
    assert not toolbox.tool_io(created["id"])["output_declared"]


def test_invalid_io_is_refused(ui_data_tmp) -> None:
    with pytest.raises(ValueError, match="io.output"):
        toolbox.create_tool(name="Bad", io={"output": "everything"})


def test_a_hand_edited_bad_io_reads_as_undeclared(ui_data_tmp) -> None:
    tool = toolbox.create_tool(name="Edited")
    path = toolbox.tool_dir(tool["id"]) / "tool.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["io"] = {"output": "???"}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert toolbox.list_tools()[0]["io"]["output"] == "passthrough"


# ------------------------------------------------------------------------------ tool io: graph

EXTRACT = {"extract": {"input": "none", "output": "folder", "output_declared": True}}
SINK = {"sink": {"input": "folder", "output": "none", "output_declared": True}}
EXPLICIT_PASS = {"p": {"input": "none", "output": "passthrough", "output_declared": True}}


def _tool_then_tag(tool_id: str, tool_source: str | None = None) -> wg.WorkflowGraph:
    nodes = []
    if tool_source:
        nodes.append(_node("f", "folder", config={"path": "."}))
    nodes.append(_node("t", "tool", source=tool_source, title="My tool", config={"tool_id": tool_id}))
    nodes.append(
        _node("tag", "prep.tag", source="t", config={"models": ["pixai-v0.9"]})
    )
    return _graph(*nodes)


def test_a_source_tool_that_outputs_a_folder_can_feed_a_prep_step() -> None:
    assert wg.validate(_tool_then_tag("extract"), tool_io=EXTRACT) == []


def test_reading_from_a_tool_that_outputs_nothing_is_refused() -> None:
    errors = wg.validate(_tool_then_tag("sink", "f"), tool_io=SINK)
    assert len(errors) == 1
    assert "node tag" in errors[0] and "outputs nothing" in errors[0]


def test_reading_from_a_source_tool_without_a_folder_output_is_refused() -> None:
    errors = wg.validate(_tool_then_tag("p"), tool_io=EXPLICIT_PASS)
    assert len(errors) == 1 and "source" in errors[0] and "'folder'" in errors[0]


def test_an_undeclared_source_tool_is_left_alone() -> None:
    """Today's behaviour: it may well return a folder at run time."""
    assert wg.validate(_tool_then_tag("old"), tool_io={"old": {"output": "passthrough"}}) == []
    assert wg.validate(_tool_then_tag("old")) == []


def test_the_check_looks_through_pass_through_tools() -> None:
    graph = _graph(
        _node("t", "tool", config={"tool_id": "sink"}),
        _node("u", "tool", source="t", config={"tool_id": "inplace"}),
        _node("tag", "prep.tag", source="u", config={"models": ["pixai-v0.9"]}),
    )
    errors = wg.validate(
        graph, tool_io={**SINK, "inplace": {"output": "passthrough", "output_declared": True}}
    )
    assert any("node tag" in e and "outputs nothing" in e for e in errors)


def test_a_folder_output_tool_is_predicted_as_decided_at_run_time() -> None:
    graph = _graph(_node("t", "tool", config={"tool_id": "extract"}))
    assert wg._predicted_handles(graph, {}, EXTRACT)["t"] is None
    assert wg.effective_output(_node("t", "tool"), SIDECAR, None, tool_output="folder") is None
    assert wg.effective_output(_node("t", "tool"), SIDECAR, None, tool_output="none") is None
    out = wg.effective_output(_node("t", "tool"), None, "/frames", tool_output="folder")
    assert out == DatasetHandle("/frames")


# ------------------------------------------------------------------------------ tool io: run


@pytest.fixture
def node_dir(ui_data_tmp: Path) -> Path:
    path = workflow_db.node_dir(1, "n2")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tool_with_io(output: str) -> str:
    return toolbox.create_tool(name=f"tool {output}", io={"output": output})["id"]


@pytest.mark.parametrize("result", ["null", '""', "{}", '{"other": 1}'])
def test_a_folder_tool_that_returns_no_path_fails_the_node(node_dir: Path, result: str) -> None:
    node = _node("n2", "tool", config={"tool_id": _tool_with_io("folder")})
    (node_dir / "result.json").write_text(result, encoding="utf-8")
    with pytest.raises(NodeOutputError, match="declares it outputs a folder"):
        wn.collect_output(node, node_dir, SIDECAR)


def test_a_folder_tool_that_returns_a_path_emits_it(node_dir: Path) -> None:
    node = _node("n2", "tool", config={"tool_id": _tool_with_io("folder")})
    (node_dir / "result.json").write_text(json.dumps("/frames"), encoding="utf-8")
    assert wn.collect_output(node, node_dir, None) == DatasetHandle("/frames")


def test_a_tool_declaring_no_output_emits_nothing(node_dir: Path) -> None:
    node = _node("n2", "tool", config={"tool_id": _tool_with_io("none")})
    (node_dir / "result.json").write_text(json.dumps("/ignored"), encoding="utf-8")
    assert wn.collect_output(node, node_dir, SIDECAR) is None


def test_graph_tool_io_reads_each_tools_declaration(ui_data_tmp) -> None:
    tool_id = _tool_with_io("folder")
    graph = _graph(
        _node("a", "tool", config={"tool_id": tool_id}),
        _node("b", "tool", config={"tool_id": "deleted-tool"}),
        _node("c", "prep.tag"),
    )
    io = wn.graph_tool_io(graph)
    assert io[tool_id]["output"] == "folder"
    assert io["deleted-tool"] == {}
    assert wg.validate(graph, tool_io=io)  # the tool-less nodes still report their own errors


def test_a_write_mode_that_restates_overwrite_does_not_change_the_hash() -> None:
    """The editor writes ``write_mode`` once a mode is picked; picking what ``overwrite`` already
    said is not a change to the run."""
    skip = _node("n", "prep.caption", config={"overwrite": False})
    assert wg.node_config_hash(skip) == wg.node_config_hash(
        _node("n", "prep.caption", config={"overwrite": False, "write_mode": "skip"})
    )
    replace = _node("n", "prep.caption", config={"overwrite": True})
    assert wg.node_config_hash(replace) == wg.node_config_hash(
        _node("n", "prep.caption", config={"overwrite": True, "write_mode": "replace"})
    )
    assert wg.node_config_hash(skip) != wg.node_config_hash(
        _node("n", "prep.caption", config={"overwrite": False, "write_mode": "append"})
    )
