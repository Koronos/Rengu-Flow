# tests/test_toolbox_smoke.py
import shutil
import time

import pytest


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_sumar_tool_runs_via_uv(ui_data_tmp, monkeypatch):
    from rengu_flow.config import local_config as lc
    from rengu_flow_ui import toolbox

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: True)
    toolbox.create_tool(
        name="Sumar",
        entrypoint="run",
        requirements=[],  # stdlib only → no network resolution
        script="def run(num1, num2):\n    return num1 + num2\n",
        inputs=[
            {"param": "num1", "label": "Number 1", "control": "number", "default": 0},
            {"param": "num2", "label": "Number 2", "control": "number", "default": 0},
        ],
    )
    toolbox.run_tool("sumar", {"num1": "2", "num2": "3"})

    deadline = time.time() + 120  # uv first run can be slow
    while time.time() < deadline:
        status = toolbox.run_status("sumar")["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.5)

    status = toolbox.run_status("sumar")
    log, _ = toolbox.read_log("sumar")
    assert status["status"] == "done", log
    assert status["exit_code"] == 0
    assert "5" in log
