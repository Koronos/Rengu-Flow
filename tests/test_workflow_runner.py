"""The workflow lane's state machine (see docs/spec/workflows.md, "Execution model").

Every test here pins a rule whose violation is **silent**: nothing crashes, the suite stays green,
and the damage shows up in the user's dataset hours later. In order:

* the handle a node emits reaches the next node, and ``saved_input`` / ``config_hash`` are written
  next to it (without ``saved_input`` every node reads stale forever);
* ``launching`` is written **before** the spawn, and a ``launching`` node is never auto-started on
  reconcile — otherwise two ``prep.tag`` processes interleave captions across the same sidecars;
* reconciliation matches ``pid`` **and** ``pid_create_time``, so a PID reused across a reboot does
  not leave a node "running" forever with its GPU lease held;
* a cancel signals first and kills after the grace period, so a half-done ``tag`` writes its report,
  and it leaves no node reading ``waiting_gpu`` under a workflow that stopped;
* exit 0 with ``report["stopped"] == true`` is ``stopped``, never ``done``;
* ``node.log`` is one run, not a history: a re-run killed before printing its marker must not
  inherit the previous run's ``= 0``;
* a ``train`` node does not start its run ahead of another workflow's standing claim, and re-fires
  when its inputs change even though it emits no output of its own;
* prep extras are refused while the *training lease* is held — the job row is still ``pending`` for
  the whole ``uv sync`` that ``start_job`` runs;
* the tick lock is non-blocking and one process is spawned no matter how many threads tick;
* deleted nodes are garbage-collected and their leases freed;
* ``[toolbox].enabled`` gates tool nodes exactly as it gates ``/toolbox``.

No GPU, no real subprocess: ``_spawn`` and ``workflow_nodes.build_launch`` are injected.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from rengu_flow_ui import db, gpu_lease, job_queue, workflow_db
from rengu_flow_ui import workflow_nodes as wn
from rengu_flow_ui import workflow_runner as wr
from rengu_flow_ui.workflow_graph import DatasetHandle, WorkflowNode

#: Captured before any fixture replaces it, for the one test that needs the genuine article.
_REAL_PID_IS_ALIVE = wr._pid_is_alive

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


# ------------------------------------------------------------------------------ graph builders


def _node(
    node_id: str,
    node_type: str,
    *,
    source: str | None = None,
    config: dict | None = None,
    enabled: bool = True,
    required: bool = False,
    wait: bool = True,
    device: int | None = None,
) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "title": node_id,
        "from": source,
        "enabled": enabled,
        "config": dict(config or {}),
        "gpu": {"required": required, "wait": wait, "device": device},
    }


def _make(nodes: list[dict], *, name: str = "wf", variables: list[dict] | None = None) -> int:
    content = json.dumps(
        {"version": 1, "name": name, "variables": variables or [], "nodes": nodes}
    )
    return workflow_db.create_workflow(name, content).id


def _chain(src: Path, **tag_gpu: Any) -> list[dict]:
    """The canonical two-step graph: a source folder feeding a tag stage."""
    return [
        _node("n1", "folder", config={"path": str(src)}),
        _node("n2", "prep.tag", source="n1", config={"models": ["pixai-v0.9"]}, **tag_gpu),
    ]


# ------------------------------------------------------------------------------ the fake lane


class FakeRuntime:
    """A launcher that records instead of spawning, plus a controllable notion of "alive"."""

    def __init__(self) -> None:
        self.spawns: list[Any] = []
        self.launch_inputs: dict[str, DatasetHandle | None] = {}
        self.alive: set[int] = set()
        self.terminated: list[int] = []
        self._next_pid = 9000
        self.on_spawn = None
        self.create_time = 1234.5

    # -- injected in place of workflow_nodes.build_launch
    def build_launch(self, node, inputs, node_dir):  # noqa: ANN001, ANN201
        if node.type in wn.INLINE_TYPES:
            return None
        node_dir.mkdir(parents=True, exist_ok=True)
        self.launch_inputs[node.id] = inputs
        return wn.NodeLaunch(argv=["noop", node.id], env={})

    # -- injected in place of workflow_runner._spawn
    def spawn(self, launch, node_dir, workflow_id, node_id):  # noqa: ANN001, ANN201
        self._next_pid += 1
        node_dir.mkdir(parents=True, exist_ok=True)
        record = type(
            "Spawn",
            (),
            {
                "pid": self._next_pid,
                "node_id": node_id,
                "node_dir": node_dir,
                "workflow_id": workflow_id,
                "argv": list(launch.argv),
            },
        )()
        # The real ``_spawn`` opens node.log with ``log_mode="wb"``: one run, one log. Mirrored
        # here — an appended log would let a re-run inherit the previous run's exit marker, which
        # is a lane-level bug and needs a lane-level test to show it.
        wn.node_log_path(node_dir).write_bytes(
            f"--- workflow {workflow_id} node {node_id} ---\n".encode()
        )
        self.spawns.append(record)
        self.alive.add(record.pid)
        if self.on_spawn is not None:
            self.on_spawn(record)
        return record.pid, self.create_time

    # -- injected in place of platform_compat.terminate_process_tree. Recording only: the tests
    #    decide when the process actually disappears, which is what lets them assert on the gap
    #    between "signalled" and "killed".
    def terminate(self, pid: int) -> None:
        self.terminated.append(int(pid))

    def last(self, node_id: str):  # noqa: ANN201
        return [s for s in self.spawns if s.node_id == node_id][-1]

    def finish(self, node_id: str, code: int = 0, *, report: dict | None = None) -> None:
        """The child exits: its log carries the marker, its report lands, its pid goes away.

        **Appends**, like a real child writing to the handle ``_spawn`` opened for it: truncation
        is the spawn's job, and doing it here instead would hide the run-to-run leak entirely.
        """
        spawn = self.last(node_id)
        with (spawn.node_dir / wn.NODE_LOG_NAME).open("a", encoding="utf-8") as log:
            log.write(f"...work...\nprep tag exits with return code = {code}\n")
        if report is not None:
            (spawn.node_dir / wn.REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
        self.alive.discard(spawn.pid)


@pytest.fixture
def rt(ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    runtime = FakeRuntime()
    monkeypatch.setattr(wr, "_spawn", runtime.spawn)
    monkeypatch.setattr(wr, "_pid_is_alive", lambda pid, ct: int(pid) in runtime.alive)
    monkeypatch.setattr(wr, "terminate_process_tree", runtime.terminate)
    # The extras install has its own tests; here it must never shell out to `uv sync`.
    monkeypatch.setattr(wr, "_install_prep_extras", lambda node: None)
    monkeypatch.setattr(wn, "build_launch", runtime.build_launch)
    # enumerate_devices shells out to nvidia-smi and caches per process.
    monkeypatch.setattr(gpu_lease, "enumerate_devices", lambda: [0])
    return runtime


@pytest.fixture
def src(tmp_path: Path) -> Path:
    folder = tmp_path / "aoi"
    folder.mkdir()
    return folder


def _start(workflow_id: int, **kwargs: Any) -> dict:
    """``start_workflow`` + the tick it owes — exactly what ``POST /start`` does, in that order.

    ``start_workflow`` plans and stops there; the route ticks synchronously right after so the
    response carries the effect. Ticking inside it as well only bought a second full pass over the
    lane per start, so the tests spell the route's two steps out rather than hiding one of them.
    """
    wr.start_workflow(workflow_id, **kwargs)
    wr.tick()
    return workflow_db.get_state(workflow_id)


def _cancel(workflow_id: int) -> dict:
    """``cancel_workflow`` + its tick — see :func:`_start`."""
    wr.cancel_workflow(workflow_id)
    wr.tick()
    return workflow_db.get_state(workflow_id)


def _state(workflow_id: int) -> dict:
    return workflow_db.get_state(workflow_id)


def _nodes(workflow_id: int) -> dict:
    return _state(workflow_id).get("nodes") or {}


def _seed_done(workflow_id: int, node_id: str, path: Path) -> None:
    wr._update_node(
        workflow_id,
        node_id,
        status="done",
        exit_code=0,
        output={"path": str(path), "caption_format": "sidecar", "caption_ext": ".txt"},
    )


def _arm(workflow_id: int, node_ids: list[str]) -> None:
    """Mark a workflow running with those nodes pending, WITHOUT ticking."""

    def _apply(state: dict) -> None:
        state["status"] = "running"
        state["current_node"] = None
        nodes = state.setdefault("nodes", {})
        for node_id in node_ids:
            nodes.setdefault(node_id, {})["status"] = "pending"

    workflow_db.mutate_state(workflow_id, _apply)


# ------------------------------------------------------------------------------ handle chain


def test_handle_propagates_along_the_chain(rt: FakeRuntime, src: Path, tmp_path: Path) -> None:
    """Each node's output is the next node's input, and the state records what it consumed."""
    cleaned = tmp_path / "cleaned"
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "prep.tag", source="n1", config={"models": ["pixai-v0.9"]}),
            _node("n3", "prep.clean", source="n2", config={"output_dir": str(cleaned)}),
        ]
    )

    _start(workflow_id)

    # n1 is inline: it completed inside the same call and n2 was spawned with its handle.
    assert _nodes(workflow_id)["n1"]["status"] == "done"
    assert _nodes(workflow_id)["n1"]["output"]["path"] == str(src)
    assert _state(workflow_id)["current_node"] == "n2"
    assert rt.launch_inputs["n2"] == DatasetHandle(str(src), "sidecar", ".txt")

    rt.finish("n2", 0)
    wr.tick()

    n2 = _nodes(workflow_id)["n2"]
    assert n2["status"] == "done"
    assert n2["output"]["path"] == str(src)  # tag writes sidecars in place: same handle
    # saved_input is what compute_stale compares against; without it every node reads stale.
    assert n2["saved_input"] == {"path": str(src), "caption_format": "sidecar", "caption_ext": ".txt"}
    assert n2["config_hash"]
    assert rt.launch_inputs["n3"] == DatasetHandle(str(src), "sidecar", ".txt")

    rt.finish("n3", 0, report={"output_dir": str(cleaned)})
    wr.tick()

    assert _nodes(workflow_id)["n3"]["output"]["path"] == str(cleaned)
    assert _state(workflow_id)["status"] == "done"
    assert _state(workflow_id)["current_node"] is None


def test_a_finished_run_is_not_stale(rt: FakeRuntime, src: Path) -> None:
    """The hash and input written on completion are the ones compute_stale recomputes."""
    from rengu_flow_ui.workflow_graph import compute_stale, parse_graph

    workflow_id = _make(_chain(src))
    _start(workflow_id)
    rt.finish("n2", 0)
    wr.tick()

    graph = parse_graph(json.loads(workflow_db.get_workflow(workflow_id).content))
    assert compute_stale(graph, _nodes(workflow_id)) == {"n1": False, "n2": False}


# ------------------------------------------------------------------------------ launching


def test_launching_is_written_before_the_spawn(rt: FakeRuntime, src: Path) -> None:
    """Not after. The window between the two is where a killed server orphans a live process."""
    seen: dict[str, Any] = {}

    def _watch(record: Any) -> None:
        state = workflow_db.get_state(record.workflow_id)
        seen["status"] = (state["nodes"].get(record.node_id) or {}).get("status")
        seen["current"] = state.get("current_node")

    rt.on_spawn = _watch
    workflow_id = _make(_chain(src))

    _start(workflow_id)

    assert seen == {"status": "launching", "current": "n2"}


def test_reconcile_never_auto_starts_a_launching_node(rt: FakeRuntime, src: Path) -> None:
    """A second `prep.tag` on the same folder interleaves sidecars with neither process aware."""
    workflow_id = _make(_chain(src))
    _start(workflow_id)
    # The server died between the state write and the spawn: launching, no pid.
    wr._update_node(workflow_id, "n2", status="launching", pid=None)
    rt.spawns.clear()

    wr.reconcile_on_start()

    n2 = _nodes(workflow_id)["n2"]
    assert n2["status"] == "failed"
    assert n2["error"] == wr.LAUNCH_INTERRUPTED_ERROR
    assert rt.spawns == []

    wr.tick()  # and the lane does not quietly resurrect it either
    assert rt.spawns == []
    assert _state(workflow_id)["status"] == "failed"


def test_reconcile_adopts_a_launching_node_whose_log_is_growing(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow_id = _make(_chain(src))
    _start(workflow_id)
    wr._update_node(workflow_id, "n2", status="launching", pid=None)

    sizes = iter([10, 2048])
    monkeypatch.setattr(wr, "_ADOPTION_SAMPLE_SECONDS", 0.0)
    monkeypatch.setattr(wr, "_log_size", lambda *args: next(sizes))

    wr.reconcile_on_start()

    n2 = _nodes(workflow_id)["n2"]
    assert n2["status"] == "running"
    assert n2["adopted"] is True


# ------------------------------------------------------------------------------ pid reuse


def test_reconcile_treats_a_reused_pid_as_dead(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same number, different process: the node must finalize and hand the GPU back.

    Without the create-time test a workflow left ``running`` across a reboot stays running forever
    and its lease is never released — the only deadlock this system can have.
    """
    workflow_id = _make(_chain(src, required=True))
    _start(workflow_id)
    holder = f"wf:{workflow_id}:n2"
    assert [row["holder_id"] for row in gpu_lease.snapshot()] == [holder]

    spawn = rt.last("n2")
    (spawn.node_dir / wn.NODE_LOG_NAME).write_text(
        "prep tag exits with return code = 0\n", encoding="utf-8"
    )
    monkeypatch.setattr(wr, "_pid_is_alive", _REAL_PID_IS_ALIVE)
    # This process is very much alive, but it is not the one we launched.
    wr._update_node(workflow_id, "n2", pid=os.getpid(), pid_create_time=1.0)

    wr.reconcile_on_start()

    assert _nodes(workflow_id)["n2"]["status"] == "done"
    assert gpu_lease.snapshot() == []


def test_reconcile_keeps_a_node_whose_pid_still_matches(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    psutil = pytest.importorskip("psutil")
    workflow_id = _make(_chain(src))
    _start(workflow_id)
    monkeypatch.setattr(wr, "_pid_is_alive", _REAL_PID_IS_ALIVE)
    wr._update_node(
        workflow_id,
        "n2",
        pid=os.getpid(),
        pid_create_time=psutil.Process(os.getpid()).create_time(),
    )

    wr.reconcile_on_start()

    assert _nodes(workflow_id)["n2"]["status"] == "running"


def test_reconcile_resets_a_waiting_node_to_pending(rt: FakeRuntime, src: Path) -> None:
    """The lease is re-evaluated from scratch: the GPU it was waiting for may be someone else's."""
    workflow_id = _make(_chain(src, required=True))
    gpu_lease.acquire("train", "job:42", None)
    _start(workflow_id)
    assert _nodes(workflow_id)["n2"]["status"] == "waiting_gpu"

    wr.reconcile_on_start()

    assert _nodes(workflow_id)["n2"]["status"] == "pending"


# ------------------------------------------------------------------------------ cancellation


def test_cancel_signals_first_and_kills_after_the_grace_period(
    rt: FakeRuntime, src: Path
) -> None:
    """Two phases, because a tick comes back to look again.

    A prep stage checks the signal between batches, so a half-done ``tag`` finishes its batch and
    writes ``report.json`` instead of losing the work.
    """
    from rengu_flow.utils.signal_files import SIGNAL_QUIT

    workflow_id = _make(_chain(src))
    _start(workflow_id)
    spawn = rt.last("n2")

    _cancel(workflow_id)

    assert (spawn.node_dir / SIGNAL_QUIT).is_file()
    assert _nodes(workflow_id)["n2"]["status"] == "stopping"
    assert rt.terminated == []  # grace, not a kill

    wr.tick()
    assert rt.terminated == []  # still inside the grace period

    wr._update_node(
        workflow_id, "n2", stop_requested_at=time.time() - wr.CANCEL_GRACE_SECONDS - 1
    )
    wr.tick()
    assert rt.terminated == [spawn.pid]

    rt.finish("n2", 0)
    wr.tick()
    assert _nodes(workflow_id)["n2"]["status"] == "stopped"
    assert _nodes(workflow_id)["n2"]["output"] is None
    assert _state(workflow_id)["status"] == "stopped"


def test_cancelling_a_tool_node_kills_it_without_a_grace_period(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool nodes have no signal contract, by decision — no complexity added to the tool model."""
    from rengu_flow.config import local_config
    from rengu_flow.utils.signal_files import SIGNAL_QUIT

    monkeypatch.setattr(local_config, "toolbox_enabled", lambda: True)
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "tool", source="n1", config={"tool_id": "t1"}),
        ]
    )
    _start(workflow_id)
    spawn = rt.last("n2")

    _cancel(workflow_id)

    assert rt.terminated == [spawn.pid]
    assert not (spawn.node_dir / SIGNAL_QUIT).exists()


def test_stale_signal_files_are_swept_when_a_node_starts(rt: FakeRuntime, src: Path) -> None:
    """Without the sweep, re-running a cancelled node exits on its first batch."""
    from rengu_flow.utils.signal_files import SIGNAL_QUIT

    workflow_id = _make(_chain(src))
    node_dir = workflow_db.node_dir(workflow_id, "n2")
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / SIGNAL_QUIT).touch()

    _start(workflow_id)

    assert not (node_dir / SIGNAL_QUIT).exists()
    assert len(rt.spawns) == 1


def test_cancelling_a_node_that_is_waiting_for_the_gpu_does_not_leave_it_waiting(
    rt: FakeRuntime, src: Path, tmp_path: Path
) -> None:
    """``waiting_gpu`` is not an active status, so nothing else finalizes it.

    Left alone the card keeps reading "Waiting for the GPU." under a workflow the user stopped
    minutes ago — and the saved output of its previous run must survive, because this node never
    started and there is nothing to distrust.
    """
    previous = tmp_path / "tagged"
    previous.mkdir()
    gpu_lease.acquire("train", "job:42", None)
    workflow_id = _make(_chain(src, required=True))
    _seed_done(workflow_id, "n2", previous)

    _start(workflow_id, force=True)
    assert _nodes(workflow_id)["n2"]["status"] == "waiting_gpu"

    _cancel(workflow_id)

    assert _state(workflow_id)["status"] == "stopped"
    assert _nodes(workflow_id)["n2"]["status"] == "stopped"
    assert _nodes(workflow_id)["n2"]["output"]["path"] == str(previous)


def test_a_node_deleted_while_it_runs_is_killed_and_swept(
    rt: FakeRuntime, src: Path
) -> None:
    """``PUT`` is supposed to 409 while running; this is what happens when the row goes anyway.

    Nothing left in the graph can finalize that process, so leaving it alive means a detached
    ``prep.tag`` still writing sidecars with no card, no pid supervision and no way to stop it.
    """
    workflow_id = _make(_chain(src, required=True))
    _start(workflow_id)
    spawn = rt.last("n2")
    record = workflow_db.get_workflow(workflow_id)
    graph = json.loads(record.content)
    graph["nodes"] = graph["nodes"][:1]
    workflow_db.update_graph(workflow_id, json.dumps(graph), expected_version=record.version)

    wr.tick()

    assert rt.terminated == [spawn.pid]
    assert "n2" not in _nodes(workflow_id)
    assert gpu_lease.snapshot() == []
    assert _state(workflow_id)["current_node"] is None


def test_signal_files_are_not_imported_at_module_scope() -> None:
    """``signal_files`` imports torch at module scope; paying that inside the poller thread's
    first cancel would stall training reconciliation with it."""
    assert not hasattr(wr, "SIGNAL_QUIT")
    assert not hasattr(wr, "SIGNAL_SAVE_QUIT")


# ------------------------------------------------------------------------------ exit mapping


def test_exit_zero_with_report_stopped_is_stopped_not_done(rt: FakeRuntime, src: Path) -> None:
    """``run_stage`` returns 0 when ``should_stop()`` fired: ``report["stopped"]`` is a FIELD.

    Read as ``done``, stopping a caption run at 60 % and pressing Run again would skip the
    remaining 40 % and train on a half-captioned dataset.
    """
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "prep.caption", source="n1", config={"model": "joycaption-beta-one"}),
            _node("n3", "prep.tag", source="n2", config={"models": ["pixai-v0.9"]}),
        ]
    )
    _start(workflow_id)

    rt.finish("n2", 0, report={"stopped": True})
    wr.tick()

    n2 = _nodes(workflow_id)["n2"]
    assert n2["status"] == "stopped"
    assert n2["output"] is None  # the handle is NOT propagated as a completed step
    assert _state(workflow_id)["status"] == "stopped"
    assert [s.node_id for s in rt.spawns] == ["n2"]  # n3 never ran


def test_a_stopped_node_is_re_run_by_a_plain_start(rt: FakeRuntime, src: Path) -> None:
    """Run covers idle + stale + failed + STOPPED — prep stages resume naturally."""
    workflow_id = _make(_chain(src))
    _start(workflow_id)
    rt.finish("n2", 0, report={"stopped": True})
    wr.tick()
    assert _nodes(workflow_id)["n2"]["status"] == "stopped"

    _start(workflow_id)

    assert _nodes(workflow_id)["n2"]["status"] == "running"
    assert len(rt.spawns) == 2


def test_a_nonzero_exit_fails_the_node_and_the_workflow(rt: FakeRuntime, src: Path) -> None:
    workflow_id = _make(_chain(src))
    _start(workflow_id)
    rt.finish("n2", 1)
    wr.tick()

    assert _nodes(workflow_id)["n2"]["status"] == "failed"
    assert _state(workflow_id)["status"] == "failed"


def test_an_unreported_exit_code_is_not_done(rt: FakeRuntime, src: Path) -> None:
    """Unknown stays unknown, **on the second run too**: the log is one run, not a history.

    ``read_exit_code`` returns the last marker in the FILE. Run 1 exits 0 and leaves ``= 0``
    behind; run 2 — the same node, re-run after a config change, which is 90 % of the use — is
    killed hard (OOM, power) before it can print anything. An appended log hands run 2 run 1's
    verdict: ``done``, exit 0, handle propagated to the next stage, workflow green.
    """
    workflow_id = _make(_chain(src))
    _start(workflow_id)
    rt.finish("n2", 0)
    wr.tick()
    assert _nodes(workflow_id)["n2"]["status"] == "done"

    _start(workflow_id, force=True)
    spawn = rt.last("n2")
    with (spawn.node_dir / wn.NODE_LOG_NAME).open("a", encoding="utf-8") as log:
        log.write("started, then nothing\n")  # killed before it could print its marker
    rt.alive.discard(spawn.pid)

    wr.tick()

    assert _nodes(workflow_id)["n2"]["status"] == "failed"
    assert _nodes(workflow_id)["n2"]["error"] == wr.EXIT_UNKNOWN_ERROR


def test_a_spawn_truncates_the_previous_run_out_of_the_log(
    ui_data_tmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production half of the rule above: ``_spawn`` opens node.log to write, not to append.

    The default in ``popen_repo_subprocess`` is ``"ab"``, so this is one keyword away from being
    wrong again, and nothing else in the lane would notice.
    """
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    log = wn.node_log_path(node_dir)
    log.write_text("prep tag exits with return code = 0\n", encoding="utf-8")

    def _fake_popen(argv, log_path, *, log_mode="ab", log_header=None, env=None):  # noqa: ANN001
        # Opened exactly the way the real one opens it, so the mode is tested, not asserted on.
        with log_path.open(log_mode) as handle:
            if log_header:
                handle.write(log_header if "b" in log_mode else log_header.decode())
        return type("Proc", (), {"pid": 4242})(), None

    monkeypatch.setattr(wr, "popen_repo_subprocess", _fake_popen)
    monkeypatch.setattr(wr, "_pid_create_time", lambda pid: 1.0)

    wr._spawn(wn.NodeLaunch(argv=["noop"], env={}), node_dir, 1, "n2")

    assert "return code = 0" not in log.read_text(encoding="utf-8")
    assert wn.read_exit_code(WorkflowNode(id="n2", type="prep.tag"), node_dir) is None


# ------------------------------------------------------------------------------ run from here


def test_run_from_here_needs_the_ancestor_output(rt: FakeRuntime, src: Path) -> None:
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "prep.tag", source="n1", config={"models": ["pixai-v0.9"]}),
            _node("n3", "prep.caption", source="n2", config={"model": "joycaption-beta-one"}),
        ]
    )
    _seed_done(workflow_id, "n1", src)  # n2 has still never run

    with pytest.raises(ValueError) as excinfo:
        _start(workflow_id, from_node="n3")

    assert str(excinfo.value) == "② has no saved output. Start from ① or earlier."
    assert rt.spawns == []
    assert _state(workflow_id).get("status") is None  # nothing was started


def test_run_from_here_keeps_ancestors_done_with_their_output(
    rt: FakeRuntime, src: Path, tmp_path: Path
) -> None:
    other = tmp_path / "tagged"
    other.mkdir()
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "prep.tag", source="n1", config={"models": ["pixai-v0.9"]}),
            _node("n3", "prep.caption", source="n2", config={"model": "joycaption-beta-one"}),
        ]
    )
    _seed_done(workflow_id, "n1", src)
    _seed_done(workflow_id, "n2", other)

    _start(workflow_id, from_node="n3")

    nodes = _nodes(workflow_id)
    assert nodes["n1"]["status"] == "done" and nodes["n1"]["output"]["path"] == str(src)
    assert nodes["n2"]["status"] == "done" and nodes["n2"]["output"]["path"] == str(other)
    assert nodes["n3"]["status"] == "running"
    assert [s.node_id for s in rt.spawns] == ["n3"]
    assert rt.launch_inputs["n3"].path == str(other)  # it consumed n2's saved handle


def test_a_fresh_done_node_is_skipped_but_a_stale_one_is_not(
    rt: FakeRuntime, src: Path
) -> None:
    workflow_id = _make(_chain(src))
    _start(workflow_id)
    rt.finish("n2", 0)
    wr.tick()
    assert _state(workflow_id)["status"] == "done"

    _start(workflow_id)  # nothing changed: nothing to do
    assert len(rt.spawns) == 1
    assert _state(workflow_id)["status"] == "done"

    record = workflow_db.get_workflow(workflow_id)
    graph = json.loads(record.content)
    graph["nodes"][1]["config"]["max_tags"] = 42  # the config chain changed -> stale
    workflow_db.update_graph(
        workflow_id, json.dumps(graph), expected_version=record.version
    )

    _start(workflow_id)
    assert len(rt.spawns) == 2


def test_force_reruns_everything(rt: FakeRuntime, src: Path) -> None:
    workflow_id = _make(_chain(src))
    _start(workflow_id)
    rt.finish("n2", 0)
    wr.tick()

    _start(workflow_id, force=True)

    assert len(rt.spawns) == 2


def test_a_disabled_node_is_skipped(rt: FakeRuntime, src: Path) -> None:
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "prep.tag", source="n1", config={"models": ["x"]}, enabled=False),
        ]
    )
    _start(workflow_id)

    assert _nodes(workflow_id)["n2"]["status"] == "skipped"
    assert rt.spawns == []
    assert _state(workflow_id)["status"] == "done"


# ------------------------------------------------------------------------------ the tick lock


def test_two_concurrent_ticks_launch_exactly_one_process(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No lease serializes a CPU-only node, so without this lock two processes would move files
    into ``low_quality/`` at once and the second pid write would orphan the first forever.

    The first thread is parked in the window that matters: after it has seen the node ``pending``
    and **before** it writes ``launching``. That is the only window in which a second thread can
    still see the same node as runnable, so it is where the lock has to do its work.
    """
    reached = threading.Event()
    release = threading.Event()
    real_input_handle = wr._input_handle

    def _park(state: dict, source_id: str | None):  # noqa: ANN202
        if not reached.is_set():
            reached.set()
            release.wait(5.0)
        return real_input_handle(state, source_id)

    monkeypatch.setattr(wr, "_input_handle", _park)

    workflow_id = _make(_chain(src))
    _seed_done(workflow_id, "n1", src)
    _arm(workflow_id, ["n2"])

    first = threading.Thread(target=wr.tick, name="tick-1")
    second = threading.Thread(target=wr.tick, name="tick-2")
    first.start()
    assert reached.wait(5.0)
    second.start()
    second.join(timeout=5.0)

    # Non-blocking: the second caller returned instead of queueing behind the first...
    assert not second.is_alive()
    # ...and it launched nothing while the first was still inside the pending window.
    assert rt.spawns == []

    release.set()
    first.join(timeout=5.0)
    assert len(rt.spawns) == 1


def test_only_one_workflow_advances_per_tick(rt: FakeRuntime, src: Path) -> None:
    first = _make(_chain(src), name="a")
    second = _make(_chain(src), name="b")
    _seed_done(first, "n1", src)
    _seed_done(second, "n1", src)
    _arm(first, ["n2"])
    _arm(second, ["n2"])

    wr.tick()

    assert [s.workflow_id for s in rt.spawns] == [first]


def test_start_refuses_while_another_workflow_runs(rt: FakeRuntime, src: Path) -> None:
    first = _make(_chain(src), name="a")
    second = _make(_chain(src), name="b")
    _start(first)

    with pytest.raises(ValueError, match="already running"):
        _start(second)


def test_start_refuses_an_invalid_graph(rt: FakeRuntime, src: Path) -> None:
    """Pre-flight reports every error at once; nothing runs until they are fixed."""
    workflow_id = _make([_node("n1", "prep.tag", source=None, config={})])

    with pytest.raises(ValueError, match="has no source"):
        _start(workflow_id)
    assert rt.spawns == []


# ------------------------------------------------------------------------------ the GPU lease


def test_a_gpu_node_waits_for_the_lease_and_starts_when_it_frees(
    rt: FakeRuntime, src: Path
) -> None:
    gpu_lease.acquire("train", "job:42", None)
    workflow_id = _make(_chain(src, required=True))

    _start(workflow_id)

    n2 = _nodes(workflow_id)["n2"]
    assert n2["status"] == "waiting_gpu"
    assert "job 42" in n2["error"]
    assert rt.spawns == []

    gpu_lease.release("job:42")
    wr.tick()

    assert _nodes(workflow_id)["n2"]["status"] == "running"
    assert len(rt.spawns) == 1


def test_a_cpu_node_takes_no_lease(rt: FakeRuntime, src: Path) -> None:
    """The whole point of the lane split: no lease means it runs alongside training."""
    gpu_lease.acquire("train", "job:42", None)
    workflow_id = _make(_chain(src, required=False))

    _start(workflow_id)

    assert _nodes(workflow_id)["n2"]["status"] == "running"
    assert [row["holder_id"] for row in gpu_lease.snapshot()] == ["job:42"]


def test_wait_false_starts_immediately_without_a_lease(rt: FakeRuntime, src: Path) -> None:
    gpu_lease.acquire("train", "job:42", None)
    workflow_id = _make(_chain(src, required=True, wait=False))

    _start(workflow_id)

    assert _nodes(workflow_id)["n2"]["status"] == "running"


def test_the_lease_is_released_on_every_verdict(rt: FakeRuntime, src: Path) -> None:
    workflow_id = _make(_chain(src, required=True))
    _start(workflow_id)
    assert gpu_lease.snapshot()

    rt.finish("n2", 1)  # failed, not done — the release is under `finally` for a reason
    wr.tick()

    assert gpu_lease.snapshot() == []


def test_a_reaped_lease_kills_the_process_it_just_spawned(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_lease, "bind_pid", lambda *args, **kwargs: False)
    workflow_id = _make(_chain(src, required=True))

    _start(workflow_id)

    spawn = rt.last("n2")
    assert rt.terminated == [spawn.pid]
    assert _nodes(workflow_id)["n2"]["status"] == "failed"
    assert gpu_lease.snapshot() == []


def test_reconcile_drops_deleted_nodes_and_frees_their_lease(
    rt: FakeRuntime, src: Path
) -> None:
    """A node the editor removed can never finalize, so its lease would block training forever."""
    workflow_id = _make(_chain(src, required=True))
    _start(workflow_id)
    holder = f"wf:{workflow_id}:n2"
    assert [row["holder_id"] for row in gpu_lease.snapshot()] == [holder]

    record = workflow_db.get_workflow(workflow_id)
    graph = json.loads(record.content)
    graph["nodes"] = graph["nodes"][:1]
    workflow_db.update_graph(
        workflow_id, json.dumps(graph), expected_version=record.version
    )

    wr.reconcile_on_start()

    assert "n2" not in _nodes(workflow_id)
    assert gpu_lease.snapshot() == []


# ------------------------------------------------------------------------------ prep extras


def _prep_node(node_type: str = "prep.quality"):  # noqa: ANN202
    from rengu_flow_ui.workflow_graph import WorkflowNode

    return WorkflowNode(id="n2", type=node_type, config={"metric": "blur"})


def test_prep_extras_are_not_installed_under_a_live_training_run(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`uv sync --extra prep` under a live DeepSpeed hands it an ImportError hours later."""
    from rengu_flow.install import manager

    monkeypatch.setattr(manager, "missing_profiles", lambda profiles: ["prep"])
    monkeypatch.setattr(job_queue, "has_active_runner", lambda: True)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not rewrite site-packages under a running job")

    monkeypatch.setattr(manager, "ensure_profiles", _boom)

    with pytest.raises(wr._WaitingForLane):
        wr._install_prep_extras(_prep_node())


def test_prep_extras_are_not_installed_while_a_training_lease_is_held(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window the job rows cannot see, and the one that matters.

    ``job_queue.try_start_next`` acquires the lease and only **then** calls ``jobs.start_job`` ->
    ``ensure_training_extras`` -> ``uv sync``, which on a cold extra takes minutes. The row is
    ``pending`` for all of it, so ``has_active_runner()`` answers "idle" for exactly the stretch in
    which a prep install would rewrite ``site-packages`` under a training run that is mid-launch.
    """
    from rengu_flow.install import manager

    monkeypatch.setattr(gpu_lease, "enumerate_devices", lambda: [0])
    monkeypatch.setattr(manager, "missing_profiles", lambda profiles: ["prep"])

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not rewrite site-packages under a starting job")

    monkeypatch.setattr(manager, "ensure_profiles", _boom)
    gpu_lease.acquire("train", "job:7", None)
    assert not job_queue.has_active_runner()  # the blind spot, spelled out

    with pytest.raises(wr._WaitingForLane):
        wr._install_prep_extras(_prep_node())


def test_a_workflow_lease_does_not_block_the_prep_install(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the *training* lane blocks. A GPU prep node is holding its own lease by the time it
    gets here, so refusing on any lease at all would make it wait for itself, forever."""
    from rengu_flow.install import manager

    calls: list[list[str]] = []
    monkeypatch.setattr(gpu_lease, "enumerate_devices", lambda: [0])
    monkeypatch.setattr(manager, "missing_profiles", lambda profiles: ["prep"])
    monkeypatch.setattr(
        manager, "ensure_profiles", lambda profiles, **kwargs: calls.append(list(profiles))
    )
    gpu_lease.acquire("workflow", "wf:1:n2", None)

    wr._install_prep_extras(_prep_node())

    assert calls == [["prep"]]


def test_prep_extras_are_installed_when_the_queue_is_idle(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rengu_flow.install import manager

    calls: list[list[str]] = []
    monkeypatch.setattr(manager, "missing_profiles", lambda profiles: ["prep"])
    monkeypatch.setattr(job_queue, "has_active_runner", lambda: False)
    monkeypatch.setattr(
        manager, "ensure_profiles", lambda profiles, **kwargs: calls.append(list(profiles))
    )

    wr._install_prep_extras(_prep_node())

    assert calls == [["prep"]]


def test_installed_prep_extras_do_not_block_a_cpu_node(
    ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to install means no site-packages rewrite — and lane separation is the feature."""
    from rengu_flow.install import manager

    monkeypatch.setattr(manager, "missing_profiles", lambda profiles: [])
    monkeypatch.setattr(job_queue, "has_active_runner", lambda: True)

    wr._install_prep_extras(_prep_node())  # must not raise


def test_a_refused_install_waits_instead_of_spawning(rt: FakeRuntime, src: Path, monkeypatch) -> None:
    def _wait(node: object) -> None:
        raise wr._WaitingForLane("Waiting for the training queue to be idle.")

    monkeypatch.setattr(wr, "_install_prep_extras", _wait)
    workflow_id = _make(_chain(src, required=True))

    _start(workflow_id)

    n2 = _nodes(workflow_id)["n2"]
    assert n2["status"] == "waiting_gpu"
    assert "training queue" in n2["error"]
    assert rt.spawns == []
    assert gpu_lease.snapshot() == []  # the lease is handed back while it waits


# ------------------------------------------------------------------------------ toolbox gate


def test_a_tool_node_fails_while_the_toolbox_gate_is_off(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authoring is always allowed; execution never is until the user opts in. A workflow node
    runs the same arbitrary user Python by the same mechanism, so it cannot be a way around it."""
    from rengu_flow.config import local_config

    monkeypatch.setattr(local_config, "toolbox_enabled", lambda: False)
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "tool", source="n1", config={"tool_id": "t1"}),
        ]
    )

    _start(workflow_id)

    n2 = _nodes(workflow_id)["n2"]
    assert n2["status"] == "failed"
    assert n2["error"] == "Execution disabled in rengu.local.toml -> [toolbox].enabled"
    assert rt.spawns == []
    assert _state(workflow_id)["status"] == "failed"


def test_a_tool_node_runs_while_the_gate_is_on(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rengu_flow.config import local_config

    monkeypatch.setattr(local_config, "toolbox_enabled", lambda: True)
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "tool", source="n1", config={"tool_id": "t1"}),
        ]
    )

    _start(workflow_id)

    assert _nodes(workflow_id)["n2"]["status"] == "running"


def test_a_tool_without_result_json_fails_the_node(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing result.json means the shim never reached its postlude, i.e. the tool raised."""
    from rengu_flow.config import local_config

    monkeypatch.setattr(local_config, "toolbox_enabled", lambda: True)
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "tool", source="n1", config={"tool_id": "t1"}),
        ]
    )
    _start(workflow_id)
    spawn = rt.last("n2")
    (spawn.node_dir / wn.NODE_LOG_NAME).write_text(
        "tool exits with return code = 0\n", encoding="utf-8"
    )
    rt.alive.discard(spawn.pid)

    wr.tick()

    assert _nodes(workflow_id)["n2"]["status"] == "failed"
    assert wn.RESULT_MISSING_ERROR in _nodes(workflow_id)["n2"]["error"]


# ------------------------------------------------------------------------------ the train claim


def _draft() -> db.JobRecord:
    return job_queue.save_draft(
        content=JOB_TOML,
        num_gpus=1,
        resume_from=None,
        output_dir=None,
        extra_args="",
        reset_dataloader=False,
        reset_optimizer=False,
    )


def _train_workflow(src: Path, job_id: Any, *, name: str) -> int:
    return _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "train", source="n1", config={"job_id": job_id}),
        ],
        name=name,
    )


def _fake_queue_start(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Let ``try_start_next`` run for real, faking only the one step that would spawn DeepSpeed.

    Stubbing ``try_start_next`` itself is what hid the displacement bug for a whole graduation
    audit: it is the very call ``_run_train`` makes, and the question these tests have to answer is
    *which* run it starts — queue positions alone cannot tell.
    """
    started: list[Any] = []

    def _start_job(job: db.JobRecord, **kwargs: Any) -> int:
        started.append(job.id)
        db.update_job(job.id, state="running", pid=os.getpid())
        return os.getpid()

    monkeypatch.setattr(job_queue.jobs, "start_job", _start_job)
    return started


def test_a_train_node_claims_the_front_of_the_queue(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _fake_queue_start(monkeypatch)
    job = _draft()
    workflow_id = _train_workflow(src, job.id, name="a")

    _start(workflow_id)

    assert _nodes(workflow_id)["n2"]["status"] == "done"
    assert _nodes(workflow_id)["n2"]["result"] == {"job_id": job.id}
    assert db.get_job(job.id).queue_position == 0
    assert started == [job.id]  # front of the queue, and the queue was started
    assert _state(workflow_id)["queue_claim"] == {"job_id": job.id}
    assert _state(workflow_id)["status"] == "done"


def test_a_second_train_node_does_not_start_its_run_ahead_of_the_standing_claim(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The question queue positions cannot answer: which run does the second node actually start?

    ``_run_train`` bumps to position 0 and calls ``try_start_next`` in the same breath, so a claim
    refused afterwards is refused too late — the queue has already started *our* run from the
    front. And this is not a narrow window: in Phase 0 nothing drains the queue periodically, so
    "idle when the second train node runs" is the normal state, not a race.
    """
    started = _fake_queue_start(monkeypatch)
    first_job, second_job = _draft(), _draft()
    first = _train_workflow(src, first_job.id, name="a")
    second = _train_workflow(src, second_job.id, name="b")

    # The first workflow claims the front while a GPU node holds the lease, so its run cannot
    # start yet — which is exactly how a live claim outlives the tick that made it.
    hog = wr._holder_id(first, "gpu-hog")
    gpu_lease.acquire("workflow", hog, None)
    _start(first)
    assert started == [] and db.get_job(first_job.id).state == "pending"
    gpu_lease.release(hog)

    _start(second)

    assert started == [first_job.id]  # the standing claim got its turn
    assert db.get_job(second_job.id).state == "pending"  # ours did not jump the queue
    assert db.get_job(first_job.id).state == "running"


def test_a_second_workflow_lands_behind_the_existing_claim(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Front-of-queue is a privilege: two workflows bumping would reorder each other's turn."""
    blocker = _draft()
    db.update_job(blocker.id, state="running")  # the lane is busy: nothing may start at all

    def _must_not_start(job: db.JobRecord, **kwargs: Any) -> int:
        raise AssertionError(f"run {job.id} was started while {blocker.id} is running")

    monkeypatch.setattr(job_queue.jobs, "start_job", _must_not_start)
    first_job, second_job = _draft(), _draft()
    first = _train_workflow(src, first_job.id, name="a")
    second = _train_workflow(src, second_job.id, name="b")

    _start(first)
    assert db.get_job(first_job.id).queue_position == 0

    _start(second)

    # The earlier claim keeps the front; the second run lands behind it, in bump order.
    assert db.get_job(first_job.id).queue_position == 0
    assert db.get_job(second_job.id).queue_position > 0
    assert _state(second).get("queue_claim") is None


def test_the_claim_expires_once_its_run_starts(
    rt: FakeRuntime, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_queue_start(monkeypatch)
    first_job, second_job = _draft(), _draft()
    first = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "train", source="n1", config={"job_id": first_job.id}),
        ],
        name="a",
    )
    second = _make(
        [
            _node("n1", "folder", config={"path": str(src)}),
            _node("n2", "train", source="n1", config={"job_id": second_job.id}),
        ],
        name="b",
    )
    _start(first)
    db.update_job(first_job.id, state="running")  # the claimed run is no longer at the front

    _start(second)

    assert db.get_job(second_job.id).queue_position == 0
    assert _state(second)["queue_claim"] == {"job_id": second_job.id}


def test_a_train_node_runs_again_when_the_dataset_variable_changes(
    rt: FakeRuntime, src: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec's rector case, end to end: ``folder(${dataset_dir}) -> prep.tag -> train``.

    Change the variable, press Run: the tag stage re-labels the new folder and the training has to
    be queued again. A ``train`` node judged on its saved output is never stale — it emits none —
    so it would be dropped from the plan while the workflow reported success with the old run.
    """

    def _must_not_start(job: db.JobRecord, **kwargs: Any) -> int:
        raise AssertionError(f"run {job.id} must stay queued for this test")

    monkeypatch.setattr(job_queue.jobs, "start_job", _must_not_start)
    db.update_job(_draft().id, state="running")  # the lane is busy, so nothing auto-starts
    job = _draft()
    other = tmp_path / "second"
    other.mkdir()
    workflow_id = _make(
        [
            _node("n1", "folder", config={"path": "${dataset_dir}"}),
            _node("n2", "prep.tag", source="n1", config={"models": ["pixai-v0.9"]}),
            _node("n3", "train", source="n2", config={"job_id": job.id}),
        ],
        variables=[{"name": "dataset_dir", "value": str(src), "description": ""}],
    )
    _start(workflow_id)
    rt.finish("n2", 0)
    wr.tick()
    assert _nodes(workflow_id)["n3"]["status"] == "done"
    db.update_job(job.id, queue_position=5)  # somebody else has bumped past it since

    record = workflow_db.get_workflow(workflow_id)
    graph = json.loads(record.content)
    graph["variables"][0]["value"] = str(other)
    workflow_db.update_graph(workflow_id, json.dumps(graph), expected_version=record.version)

    _start(workflow_id)
    rt.finish("n2", 0)
    wr.tick()

    assert _nodes(workflow_id)["n3"]["status"] == "done"
    assert db.get_job(job.id).queue_position == 0  # it really fired again
    assert _state(workflow_id)["status"] == "done"
