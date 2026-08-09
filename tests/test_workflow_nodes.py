"""Per-node execution mechanics (see docs/spec/workflows.md, "Execution model").

The rules pinned here are the ones a plausible wrong implementation gets wrong silently:

* ``path`` / ``caption_format`` / ``caption_ext`` come from the incoming edge and **only** from
  it — a node that could keep its own format would emit a handle describing the old one;
* ``validate_for_stage`` runs **before** the spawn, so "folder does not exist" is a clean node
  error instead of a traceback buried in ``node.log``;
* a prep node's env carries **no training knobs** (NCCL/TF32/allocator);
* a tool receives the handle only when it *declares* a ``path`` input;
* a missing ``result.json`` is a **failure**, never a pass-through — otherwise a green workflow
  walks straight past a crashed tool.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import toml

from rengu_flow_ui import db, job_queue, toolbox, workflow_db
from rengu_flow_ui import workflow_nodes as wn
from rengu_flow_ui.workflow_graph import DatasetHandle, NodeGpu, NodeOutputError, WorkflowNode

JOB_TOML = """
dataset = "examples/minimal_dataset.toml"
output_dir = "output"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"

[optimizer]
type = "adamw"
lr = 1.0e-4

epochs = 1
micro_batch_size_per_gpu = 1
"""


def _node(node_type: str, config: dict | None = None, **gpu: object) -> WorkflowNode:
    return WorkflowNode(
        id="n2",
        type=node_type,
        config=dict(config or {}),
        gpu=NodeGpu(**gpu),  # type: ignore[arg-type]
    )


def _handle(path: object, **kwargs: str) -> DatasetHandle:
    return DatasetHandle(path=str(path), **kwargs)


@pytest.fixture
def node_dir(ui_data_tmp: Path) -> Path:
    """The real ``data/workflows/<wf>/<node>`` path, so the shape under test is the shipped one."""
    return workflow_db.node_dir(1, "n2")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "aoi"
    d.mkdir()
    return d


def _save_draft() -> db.JobRecord:
    return job_queue.save_draft(
        content=JOB_TOML,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )


# ------------------------------------------------------------------------------ prep: config


def test_prep_handle_keys_come_only_from_the_edge(node_dir: Path, dataset_dir: Path) -> None:
    """The edge MANDA on all three, and a stale copy in the config is dropped, not honoured.

    Letting the node keep ``caption_format: json`` here is not a harmless preference: the stage
    would write ``.json`` captions while ``effective_output`` handed the next node the *input*
    handle, still saying ``sidecar``/``.txt``. The next node would then read the format this one
    stopped writing. The spec's ``config`` is the stage section minus these three keys.
    """
    node = _node(
        "prep.tag",
        {
            "models": ["pixai-v0.9"],
            "path": "D:/stale",
            "caption_format": "json",
            "caption_ext": ".caption",
        },
    )
    wn.build_launch(node, _handle(dataset_dir, caption_format="sidecar", caption_ext=".txt"), node_dir)

    written = toml.loads((node_dir / "prep.toml").read_text(encoding="utf-8"))
    assert written["path"] == str(dataset_dir)  # the edge wins over the stored path
    assert written["caption_format"] == "sidecar"  # ... and over the stored format
    assert written["caption_ext"] == ".txt"
    assert written["tag"]["models"] == ["pixai-v0.9"]
    # The three keys live at the top level, and never leak back into the stage section.
    assert not set(written["tag"]) & {"path", "caption_format", "caption_ext"}


def test_prep_inherits_caption_format_and_ext_from_the_handle(
    node_dir: Path, dataset_dir: Path
) -> None:
    node = _node("prep.caption", {"model": "joycaption-beta-one"})
    wn.build_launch(node, _handle(dataset_dir, caption_format="json", caption_ext=".caption"), node_dir)

    written = toml.loads((node_dir / "prep.toml").read_text(encoding="utf-8"))
    assert written["caption_format"] == "json"
    assert written["caption_ext"] == ".caption"


def test_prep_validate_for_stage_runs_before_the_spawn(node_dir: Path, tmp_path: Path) -> None:
    """A missing folder must be a clean error, and must not leave a materialized prep.toml."""
    node = _node("prep.tag", {})
    with pytest.raises(FileNotFoundError):
        wn.build_launch(node, _handle(tmp_path / "does-not-exist"), node_dir)
    assert not (node_dir / "prep.toml").exists()


def test_prep_without_an_input_handle_fails_validation(node_dir: Path) -> None:
    node = _node("prep.tag", {})
    with pytest.raises(ValueError, match="path"):
        wn.build_launch(node, None, node_dir)


def test_prep_index_without_models_fails_validation(node_dir: Path, dataset_dir: Path) -> None:
    node = _node("prep.index", {"models": []})
    with pytest.raises(ValueError, match="index"):
        wn.build_launch(node, _handle(dataset_dir), node_dir)


# ------------------------------------------------------------------------------ prep: launch


def test_prep_argv_shape(node_dir: Path, dataset_dir: Path) -> None:
    node = _node("prep.quality", {"metric": "blur"})
    launch = wn.build_launch(node, _handle(dataset_dir), node_dir)

    assert launch is not None
    assert launch.argv[0] == sys.executable
    assert launch.argv[1:5] == ["-m", "rengu_flow.cli", "prep", "quality"]
    assert launch.argv[5:7] == ["--config", str(node_dir / "prep.toml")]
    assert launch.argv[7:] == ["--job-dir", str(node_dir)]


def test_prep_env_carries_no_training_knobs(
    node_dir: Path, dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NCCL/TF32/allocator are training concerns prep only inherits today because start_job
    is shared. A node's env is os.environ + PYTHONUNBUFFERED (+ CUDA_VISIBLE_DEVICES)."""
    from rengu_flow.cli import train_launcher

    def _boom(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("prep nodes must not build a training env")

    assert not hasattr(wn, "training_subprocess_env")  # not imported at module scope either
    monkeypatch.setattr(train_launcher, "training_subprocess_env", _boom)
    for key in ("NCCL_P2P_DISABLE", "PYTORCH_CUDA_ALLOC_CONF", "RENGU_ENGINE"):
        monkeypatch.delenv(key, raising=False)

    launch = wn.build_launch(_node("prep.tag", {}), _handle(dataset_dir), node_dir)

    assert launch is not None
    assert launch.env["PYTHONUNBUFFERED"] == "1"
    assert set(launch.env) - set(os.environ) <= {"PYTHONUNBUFFERED", "CUDA_VISIBLE_DEVICES"}
    assert not [k for k in launch.env if k.startswith("NCCL_") and k not in os.environ]
    assert "PYTORCH_CUDA_ALLOC_CONF" not in launch.env
    assert "RENGU_ENGINE" not in launch.env


def test_prep_env_pins_cuda_visible_devices_from_the_node(
    node_dir: Path, dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")  # merged last: the node's choice wins
    launch = wn.build_launch(_node("prep.tag", {}, device=1), _handle(dataset_dir), node_dir)
    assert launch is not None
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "1"


def test_prep_env_leaves_cuda_visible_devices_alone_on_auto(
    node_dir: Path, dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    launch = wn.build_launch(_node("prep.tag", {}, device=None), _handle(dataset_dir), node_dir)
    assert launch is not None
    assert "CUDA_VISIBLE_DEVICES" not in launch.env


@pytest.mark.parametrize(
    ("required", "wait", "expected"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_needs_lease_follows_required_and_wait(required: bool, wait: bool, expected: bool) -> None:
    """``wait: false`` is the explicit escape hatch: start now, do not ask for a lease.

    Asked of the node, not of a launch: the runner needs the answer *before* it builds one, since
    a node that cannot get its GPU is never built.
    """
    assert wn.needs_lease(_node("prep.tag", {}, required=required, wait=wait)) is expected


@pytest.mark.parametrize(("device", "expected"), [(None, None), (0, [0]), (2, [2])])
def test_lease_devices_come_from_the_node_gpu_device(
    device: int | None, expected: list[int] | None
) -> None:
    assert wn.lease_devices(_node("prep.tag", {}, required=True, device=device)) == expected


# ------------------------------------------------------------------------------ tool nodes


def _tool(name: str, inputs: list[dict], script: str = "def run(**kw):\n    return None\n") -> str:
    return toolbox.create_tool(name=name, entrypoint="run", script=script, inputs=inputs)["id"]


def test_tool_receives_the_handle_when_it_declares_path(
    node_dir: Path, dataset_dir: Path
) -> None:
    tool_id = _tool(
        "Handle Tool",
        [
            {"param": "path", "control": "text"},
            {"param": "caption_ext", "control": "text"},
            {"param": "limit", "control": "number", "default": 3},
        ],
    )
    node = _node("tool", {"tool_id": tool_id, "values": {"path": "D:/stale", "limit": 5}})
    launch = wn.build_launch(node, _handle(dataset_dir, caption_ext=".caption"), node_dir)

    assert launch is not None
    kwargs = json.loads((node_dir / "inputs.json").read_text(encoding="utf-8"))
    assert kwargs["path"] == str(dataset_dir)  # injection by convention overrides the stored value
    assert kwargs["caption_ext"] == ".caption"
    assert kwargs["limit"] == 5
    assert Path(launch.argv[-1]).parent == node_dir  # runs in its own dir, not the tool's


def test_tool_without_a_path_input_is_passthrough(node_dir: Path, dataset_dir: Path) -> None:
    """``path`` is the whole convention: no ``path`` input, no handle — not even the format,
    which would otherwise arrive at a tool that was never written to receive it."""
    tool_id = _tool(
        "No Handle Tool",
        [
            {"param": "limit", "control": "number", "default": 3},
            {"param": "caption_ext", "control": "text"},
        ],
    )
    node = _node("tool", {"tool_id": tool_id, "values": {}})
    wn.build_launch(node, _handle(dataset_dir, caption_ext=".caption"), node_dir)

    kwargs = json.loads((node_dir / "inputs.json").read_text(encoding="utf-8"))
    assert "path" not in kwargs
    assert kwargs["caption_ext"] is None  # declared but left blank, never injected


def test_tool_node_without_a_tool_id_is_an_error(node_dir: Path, dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="tool_id"):
        wn.build_launch(_node("tool", {"values": {}}), _handle(dataset_dir), node_dir)


# ------------------------------------------------------------------------------ collect_output


def test_collect_output_missing_result_json_is_a_failure(
    node_dir: Path, dataset_dir: Path
) -> None:
    """The shim writes result.json in its postlude, so an absent file means the tool raised.
    Treating it as None would take the pass-through branch and carry a green workflow past a
    crashed tool."""
    node = _node("tool", {"tool_id": "whatever"})
    node_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(NodeOutputError):
        wn.collect_output(node, node_dir, _handle(dataset_dir))


def test_collect_output_null_result_is_passthrough(node_dir: Path, dataset_dir: Path) -> None:
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "result.json").write_text("null", encoding="utf-8")
    handle = _handle(dataset_dir, caption_format="json", caption_ext=".caption")
    assert wn.collect_output(_node("tool", {}), node_dir, handle) == handle


def test_collect_output_reads_the_tool_return_value(node_dir: Path, dataset_dir: Path) -> None:
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "result.json").write_text(json.dumps("D:/out/export"), encoding="utf-8")
    out = wn.collect_output(_node("tool", {}), node_dir, _handle(dataset_dir, caption_ext=".cap"))
    assert out == DatasetHandle(path="D:/out/export", caption_format="sidecar", caption_ext=".cap")


def test_collect_output_rejects_a_non_handle_return(node_dir: Path, dataset_dir: Path) -> None:
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "result.json").write_text("42", encoding="utf-8")
    with pytest.raises(NodeOutputError):
        wn.collect_output(_node("tool", {}), node_dir, _handle(dataset_dir))


def test_collect_output_quality_emits_its_input_not_the_quarantine(
    node_dir: Path, dataset_dir: Path
) -> None:
    """``quality``'s report["output_dir"] is the reject pile — reading it would caption it."""
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "report.json").write_text(
        json.dumps({"output_dir": str(dataset_dir / "low_quality")}), encoding="utf-8"
    )
    handle = _handle(dataset_dir)
    assert wn.collect_output(_node("prep.quality", {}), node_dir, handle) == handle


def test_collect_output_clean_reads_its_report_output_dir(
    node_dir: Path, dataset_dir: Path
) -> None:
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "report.json").write_text(
        json.dumps({"output_dir": str(dataset_dir / "cleaned")}), encoding="utf-8"
    )
    out = wn.collect_output(_node("prep.clean", {}), node_dir, _handle(dataset_dir))
    assert out is not None and out.path == str(dataset_dir / "cleaned")


def test_collect_output_prep_without_a_report_predicts_the_handle(
    node_dir: Path, dataset_dir: Path
) -> None:
    handle = _handle(dataset_dir)
    assert wn.collect_output(_node("prep.tag", {}), node_dir, handle) == handle


def test_collect_output_folder_emits_its_config(node_dir: Path, dataset_dir: Path) -> None:
    node = _node("folder", {"path": str(dataset_dir), "caption_ext": ".cap"})
    out = wn.collect_output(node, node_dir, None)
    assert out == DatasetHandle(path=str(dataset_dir), caption_format="sidecar", caption_ext=".cap")


def test_collect_output_train_emits_nothing(node_dir: Path, dataset_dir: Path) -> None:
    """``train`` is terminal — workflow_graph.effective_output is the authority."""
    assert wn.collect_output(_node("train", {"job_id": 1}), node_dir, _handle(dataset_dir)) is None


# ------------------------------------------------------------------------------ read_exit_code


@pytest.mark.parametrize(
    ("node_type", "log", "expected"),
    [
        ("prep.tag", "...\nprep tag exits with return code = 0\n", 0),
        ("prep.caption", "prep caption exits with return code = -15\n", -15),
        ("tool", "hello\ntool exits with return code = 3\n", 3),
        # Unknown stays unknown: unlike jobs._read_exit_code, nothing here maps it to success.
        ("prep.tag", "Traceback (most recent call last):\n  boom\n", None),
        ("tool", "", None),
    ],
)
def test_read_exit_code_parses_both_markers(
    node_dir: Path, node_type: str, log: str, expected: int | None
) -> None:
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "node.log").write_text(log, encoding="utf-8")
    assert wn.read_exit_code(_node(node_type, {}), node_dir) == expected


def test_read_exit_code_without_a_log_is_none(node_dir: Path) -> None:
    assert wn.read_exit_code(_node("prep.tag", {}), node_dir) is None


def test_read_exit_code_takes_the_last_marker(node_dir: Path) -> None:
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "node.log").write_text(
        "prep tag exits with return code = 1\nprep tag exits with return code = 0\n",
        encoding="utf-8",
    )
    assert wn.read_exit_code(_node("prep.tag", {}), node_dir) == 0


# ------------------------------------------------------------------------------ inline: folder


def test_folder_and_train_have_no_subprocess(node_dir: Path, dataset_dir: Path) -> None:
    assert wn.build_launch(_node("folder", {"path": str(dataset_dir)}), None, node_dir) is None
    assert wn.build_launch(_node("train", {"job_id": 1}), _handle(dataset_dir), node_dir) is None


def test_folder_run_inline_emits_its_handle(node_dir: Path, dataset_dir: Path) -> None:
    node = _node("folder", {"path": str(dataset_dir), "caption_format": "json"})
    assert wn.run_inline(node, None, node_dir) == {
        "path": str(dataset_dir),
        "caption_format": "json",
        "caption_ext": ".txt",
    }


def test_folder_run_inline_requires_an_existing_folder(node_dir: Path, tmp_path: Path) -> None:
    node = _node("folder", {"path": str(tmp_path / "nope")})
    with pytest.raises(FileNotFoundError):
        wn.run_inline(node, None, node_dir)


def test_folder_run_inline_requires_a_path(node_dir: Path) -> None:
    with pytest.raises(ValueError, match="path"):
        wn.run_inline(_node("folder", {}), None, node_dir)


# ------------------------------------------------------------------------------ inline: train


def test_train_enqueues_a_new_draft_and_starts_the_queue(
    node_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts: list[int] = []
    monkeypatch.setattr(job_queue, "try_start_next", lambda: starts.append(1))
    draft = _save_draft()

    result = wn.run_inline(_node("train", {"job_id": draft.id}), None, node_dir)

    assert result == {"job_id": draft.id}
    job = db.get_job(draft.id)
    assert job.state == "pending"  # promoted out of `new`
    assert job.queue_position == 0  # and put at the front
    assert starts == [1]


def test_train_bumps_an_already_pending_run_to_the_front(
    node_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job_queue, "try_start_next", lambda: None)
    first = job_queue.enqueue_existing(_save_draft().id)
    second = job_queue.enqueue_existing(_save_draft().id)
    assert first.queue_position < second.queue_position

    wn.run_inline(_node("train", {"job_id": second.id}), None, node_dir)

    assert db.get_job(second.id).queue_position == 0
    assert db.get_job(first.id).queue_position == 1
    assert db.get_job(second.id).state == "pending"


def test_train_refuses_a_run_that_is_already_underway(
    node_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never preempt, never re-queue a live run — fail with something the user can read."""
    monkeypatch.setattr(job_queue, "try_start_next", lambda: None)
    job = db.create_job(config_path="/tmp/x.toml", log_path="/tmp/x.log", state="running")
    with pytest.raises(ValueError, match="running"):
        wn.run_inline(_node("train", {"job_id": job.id}), None, node_dir)


@pytest.mark.parametrize("state", ["finished", "failed", "stopped", "stopping"])
def test_train_refuses_a_terminal_run(
    node_dir: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    monkeypatch.setattr(job_queue, "try_start_next", lambda: None)
    job = db.create_job(config_path="/tmp/x.toml", log_path="/tmp/x.log", state=state)
    with pytest.raises(ValueError, match=state):
        wn.run_inline(_node("train", {"job_id": job.id}), None, node_dir)


def test_train_without_a_job_id_is_an_error(node_dir: Path) -> None:
    with pytest.raises(ValueError, match="job_id"):
        wn.run_inline(_node("train", {}), None, node_dir)


def test_train_with_an_unknown_job_id_is_an_error(node_dir: Path) -> None:
    with pytest.raises(ValueError, match="404"):
        wn.run_inline(_node("train", {"job_id": 404}), None, node_dir)


# ------------------------------------------------------------------------------ unknown types


def test_unknown_node_type_is_refused(node_dir: Path, dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="prep.nope"):
        wn.build_launch(_node("prep.nope", {}), _handle(dataset_dir), node_dir)
    with pytest.raises(ValueError, match="mystery"):
        wn.build_launch(_node("mystery", {}), _handle(dataset_dir), node_dir)


def test_run_inline_refuses_a_subprocess_node(node_dir: Path, dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="build_launch"):
        wn.run_inline(_node("prep.tag", {}), _handle(dataset_dir), node_dir)
