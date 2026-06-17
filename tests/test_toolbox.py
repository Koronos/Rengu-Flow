import json
from pathlib import Path

import pytest

from rengu_flow.config import local_config as lc


def test_toolbox_enabled_defaults_false_when_section_absent(tmp_path: Path):
    cfg = lc.parse_local_config_dict({}, root=tmp_path)
    assert cfg.toolbox.enabled is False


def test_toolbox_enabled_reads_truthy_values(tmp_path: Path):
    cfg = lc.parse_local_config_dict({"toolbox": {"enabled": "on"}}, root=tmp_path)
    assert cfg.toolbox.enabled is True

    cfg2 = lc.parse_local_config_dict({"toolbox": {"enabled": True}}, root=tmp_path)
    assert cfg2.toolbox.enabled is True


def test_toolbox_dir_under_ui_data_dir(ui_data_tmp):
    from rengu_flow_ui import settings

    assert settings.toolbox_dir() == ui_data_tmp / "toolbox"
    settings.ensure_data_dirs()
    assert settings.toolbox_dir().is_dir()


def test_create_and_get_tool_roundtrip(ui_data_tmp):
    from rengu_flow_ui import toolbox

    saved = toolbox.create_tool(
        name="Sumar dos números",
        description="Suma num1 + num2",
        entrypoint="run",
        requirements=["numpy>=2.0"],
        script="def run(num1, num2):\n    return num1 + num2\n",
        inputs=[
            {"param": "num1", "label": "Number 1", "control": "number", "default": 0},
            {"param": "num2", "label": "Number 2", "control": "number", "default": 0},
        ],
    )
    assert saved["id"] == "sumar-dos-numeros"
    assert saved["created_at"].endswith("Z")
    assert saved["created_at"] == saved["updated_at"]

    full = toolbox.get_tool("sumar-dos-numeros")
    assert full["script"].startswith("def run(")
    assert full["requirements"] == ["numpy>=2.0"]
    assert full["last_run"] is None
    # tool.json is on disk
    on_disk = json.loads((toolbox.tool_dir("sumar-dos-numeros") / "tool.json").read_text())
    assert on_disk["entrypoint"] == "run"


def test_create_tool_dedupes_slug(ui_data_tmp):
    from rengu_flow_ui import toolbox

    a = toolbox.create_tool(name="My Tool")
    b = toolbox.create_tool(name="My Tool")
    assert a["id"] == "my-tool"
    assert b["id"] == "my-tool-2"


def test_list_and_delete_tools(ui_data_tmp):
    from rengu_flow_ui import toolbox

    toolbox.create_tool(name="Alpha")
    toolbox.create_tool(name="Beta")
    ids = {t["id"] for t in toolbox.list_tools()}
    assert ids == {"alpha", "beta"}

    toolbox.delete_tool("alpha")
    assert {t["id"] for t in toolbox.list_tools()} == {"beta"}


def test_get_missing_tool_raises_keyerror(ui_data_tmp):
    from rengu_flow_ui import toolbox

    with pytest.raises(KeyError):
        toolbox.get_tool("nope")


def test_cast_inputs_types():
    from rengu_flow_ui import toolbox

    defs = [
        {"param": "n", "control": "number"},
        {"param": "f", "control": "number"},
        {"param": "t", "control": "text"},
        {"param": "b", "control": "switch"},
        {"param": "s", "control": "select", "options": ["a", "b"]},
    ]
    kwargs = toolbox.cast_inputs(
        defs, {"n": "3", "f": "1.5", "t": 42, "b": "true", "s": "b"}
    )
    assert kwargs == {"n": 3, "f": 1.5, "t": "42", "b": True, "s": "b"}


def test_cast_inputs_rejects_bad_number():
    from rengu_flow_ui import toolbox

    with pytest.raises(ValueError):
        toolbox.cast_inputs([{"param": "n", "control": "number"}], {"n": "not-a-number"})


def test_build_runner_source_has_pep723_header():
    from rengu_flow_ui import toolbox

    src = toolbox.build_runner_source("run", ["numpy>=2.0", "pillow"])
    assert "# /// script" in src
    assert '"numpy>=2.0"' in src and '"pillow"' in src
    assert 'getattr(mod, "run")(**kwargs)' in src
    # No requirements → still a valid (empty) dependencies list
    src2 = toolbox.build_runner_source("main", [])
    assert "dependencies = []" in src2
    assert 'getattr(mod, "main")(**kwargs)' in src2


def test_run_tool_rejected_when_execution_disabled(ui_data_tmp, monkeypatch):
    from rengu_flow.config import local_config as lc
    from rengu_flow_ui import toolbox

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: False)
    toolbox.create_tool(name="T", script="def run():\n    return 1\n")
    with pytest.raises(toolbox.ExecutionDisabledError):
        toolbox.run_tool("t", {})


def test_uv_run_argv_shape(ui_data_tmp):
    from rengu_flow_ui import toolbox

    toolbox.create_tool(name="T")
    argv = toolbox.uv_run_argv("t")
    assert argv[1:4] == ["run", "--no-project", "--isolated"]
    assert argv[-1].endswith("_runner.py")
