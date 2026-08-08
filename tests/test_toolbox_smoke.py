# tests/test_toolbox_smoke.py
import shutil
import time

import pytest


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_add_tool_runs_via_uv(ui_data_tmp, monkeypatch):
    from rengu_flow.config import local_config as lc
    from rengu_flow_ui import toolbox

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: True)
    toolbox.create_tool(
        name="Add",
        entrypoint="run",
        requirements=[],  # stdlib only → no network resolution
        script="def run(num1, num2):\n    return num1 + num2\n",
        inputs=[
            {"param": "num1", "label": "Number 1", "control": "number", "default": 0},
            {"param": "num2", "label": "Number 2", "control": "number", "default": 0},
        ],
    )
    toolbox.run_tool("add", {"num1": "2", "num2": "3"})

    deadline = time.time() + 120  # uv first run can be slow
    while time.time() < deadline:
        status = toolbox.run_status("add")["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.5)

    status = toolbox.run_status("add")
    log, _ = toolbox.read_log("add")
    assert status["status"] == "done", log
    assert status["exit_code"] == 0
    assert "5" in log


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_sys_exit_tool_reports_real_exit_code_via_uv(ui_data_tmp, monkeypatch):
    """End-to-end: a tool that calls sys.exit(3) must surface exit_code == 3.

    Regression for the shim's ``except Exception`` swallowing ``SystemExit`` and always
    logging "tool exits with return code = 0" regardless of the real code.
    """
    from rengu_flow.config import local_config as lc
    from rengu_flow_ui import toolbox

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: True)
    toolbox.create_tool(
        name="Exit Three",
        entrypoint="run",
        requirements=[],  # stdlib only -> no network resolution
        script="import sys\n\n\ndef run():\n    sys.exit(3)\n",
    )
    toolbox.run_tool("exit-three", {})

    deadline = time.time() + 120  # uv first run can be slow
    while time.time() < deadline:
        status = toolbox.run_status("exit-three")["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.5)

    status = toolbox.run_status("exit-three")
    log, _ = toolbox.read_log("exit-three")
    assert status["status"] == "failed", log
    assert status["exit_code"] == 3, log
    assert "tool exits with return code = 3" in log, log
