"""Toolbox: user-authored Python tools persisted under ``data/toolbox/<id>/``.

One folder per tool. A single last-run record per tool (no queue, no history).
Authoring is always allowed; execution is gated by ``[toolbox].enabled`` in the
local TOML (see ``rengu_flow.config.local_config.toolbox_enabled``).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rengu_flow.config import local_config

from rengu_flow_ui import settings

CONTROL_TYPES = ("number", "text", "textarea", "switch", "select")
DEFAULT_ENTRYPOINT = "run"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name: str) -> str:
    # Normalize Unicode (decompose accents)
    normalized = unicodedata.normalize("NFKD", name.strip().lower())
    # Remove non-ASCII
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace non-alphanumeric with dashes
    s = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    return s or "tool"


def tool_dir(tool_id: str) -> Path:
    return settings.toolbox_dir() / tool_id


def _tool_json_path(tool_id: str) -> Path:
    return tool_dir(tool_id) / "tool.json"


def _script_path(tool_id: str) -> Path:
    return tool_dir(tool_id) / "tool.py"


def _unique_id(base: str) -> str:
    candidate = base
    n = 2
    while tool_dir(candidate).exists():
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _normalize_inputs(inputs: list[dict] | None) -> list[dict]:
    out: list[dict] = []
    for raw in inputs or []:
        control = raw.get("control", "text")
        if control not in CONTROL_TYPES:
            raise ValueError(f"Unknown control type {control!r}")
        item = {
            "param": str(raw["param"]),
            "label": str(raw.get("label", raw["param"])),
            "control": control,
            "default": raw.get("default"),
            "hint": str(raw.get("hint", "")),
        }
        if control == "select":
            item["options"] = [str(o) for o in raw.get("options", [])]
        if control == "number":
            item["min"] = raw.get("min")
            item["max"] = raw.get("max")
            item["step"] = raw.get("step")
        out.append(item)
    return out


def _read_tool_json(tool_id: str) -> dict[str, Any]:
    path = _tool_json_path(tool_id)
    if not path.is_file():
        raise KeyError(tool_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tool_json(tool_id: str, data: dict[str, Any]) -> None:
    _tool_json_path(tool_id).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def create_tool(
    name: str,
    description: str = "",
    entrypoint: str = DEFAULT_ENTRYPOINT,
    requirements: list[str] | None = None,
    script: str = "",
    inputs: list[dict] | None = None,
) -> dict[str, Any]:
    settings.toolbox_dir().mkdir(parents=True, exist_ok=True)
    tool_id = _unique_id(slugify(name))
    tool_dir(tool_id).mkdir(parents=True, exist_ok=True)
    now = _now_iso()
    data = {
        "id": tool_id,
        "name": name,
        "description": description,
        "entrypoint": (entrypoint or DEFAULT_ENTRYPOINT).strip(),
        "requirements": [str(r).strip() for r in (requirements or []) if str(r).strip()],
        "inputs": _normalize_inputs(inputs),
        "created_at": now,
        "updated_at": now,
    }
    _write_tool_json(tool_id, data)
    _script_path(tool_id).write_text(script or "", encoding="utf-8")
    return data


def get_tool(tool_id: str) -> dict[str, Any]:
    data = _read_tool_json(tool_id)
    data["script"] = (
        _script_path(tool_id).read_text(encoding="utf-8")
        if _script_path(tool_id).is_file()
        else ""
    )
    data["last_run"] = _read_last_run(tool_id)
    return data


def list_tools() -> list[dict[str, Any]]:
    base = settings.toolbox_dir()
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(base.iterdir()):
        if not (d / "tool.json").is_file():
            continue
        data = json.loads((d / "tool.json").read_text(encoding="utf-8"))
        last = _read_last_run(data["id"])
        out.append(
            {
                "id": data["id"],
                "name": data["name"],
                "description": data.get("description", ""),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "last_run_status": (last or {}).get("status", "idle"),
            }
        )
    return out


def update_tool(tool_id: str, **fields: Any) -> dict[str, Any]:
    data = _read_tool_json(tool_id)
    if "name" in fields:
        data["name"] = fields["name"]
    if "description" in fields:
        data["description"] = fields["description"]
    if "entrypoint" in fields:
        data["entrypoint"] = (fields["entrypoint"] or DEFAULT_ENTRYPOINT).strip()
    if "requirements" in fields:
        data["requirements"] = [
            str(r).strip() for r in (fields["requirements"] or []) if str(r).strip()
        ]
    if "inputs" in fields:
        data["inputs"] = _normalize_inputs(fields["inputs"])
    if "script" in fields:
        _script_path(tool_id).write_text(fields["script"] or "", encoding="utf-8")
    data["updated_at"] = _now_iso()
    _write_tool_json(tool_id, data)
    return data


def delete_tool(tool_id: str) -> None:
    d = tool_dir(tool_id)
    if not d.is_dir():
        raise KeyError(tool_id)
    shutil.rmtree(d)


def _read_last_run(tool_id: str) -> dict[str, Any] | None:
    path = tool_dir(tool_id) / "last_run.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


_TRUTHY = {"1", "true", "yes", "on"}


def _cast_number(raw: Any) -> int | float:
    text = str(raw).strip()
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except ValueError as e:
        raise ValueError(f"Expected a number, got {raw!r}") from e


def cast_inputs(inputs_def: list[dict], values: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for spec in inputs_def:
        param = spec["param"]
        control = spec.get("control", "text")
        if param not in values or values[param] is None or values[param] == "":
            if spec.get("default") is not None:
                raw = spec["default"]
            else:
                raise ValueError(f"Missing value for input {param!r}")
        else:
            raw = values[param]
        if control == "number":
            kwargs[param] = _cast_number(raw)
        elif control == "switch":
            kwargs[param] = raw is True or str(raw).strip().lower() in _TRUTHY
        else:  # text, textarea, select
            kwargs[param] = str(raw)
    return kwargs


# In-process registry of the single active process per tool id.
_active: dict[str, subprocess.Popen[Any]] = {}


class ExecutionDisabledError(RuntimeError):
    pass


class RunActiveError(RuntimeError):
    pass


def _log_path(tool_id: str) -> Path:
    return tool_dir(tool_id) / "last_run.log"


def _last_run_path(tool_id: str) -> Path:
    return tool_dir(tool_id) / "last_run.json"


def _write_last_run(tool_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _last_run_path(tool_id).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return data


def uv_run_argv(tool_id: str) -> list[str]:
    uv = shutil.which("uv")
    if not uv:
        raise FileNotFoundError(
            "uv is not on PATH. Install uv (https://docs.astral.sh/uv/) to run Toolbox tools."
        )
    runner = tool_dir(tool_id) / "_runner.py"
    return [uv, "run", "--no-project", "--isolated", str(runner)]


def run_tool(tool_id: str, values: dict[str, Any]) -> dict[str, Any]:
    if not local_config.toolbox_enabled():
        raise ExecutionDisabledError(
            "Execution disabled in rengu.local.toml -> [toolbox].enabled"
        )
    data = _read_tool_json(tool_id)  # raises KeyError if missing
    existing = _active.get(tool_id)
    if existing is not None and existing.poll() is None:
        raise RunActiveError("A run is already active for this tool")

    kwargs = cast_inputs(data.get("inputs", []), values)
    d = tool_dir(tool_id)
    (d / "inputs.json").write_text(json.dumps(kwargs), encoding="utf-8")
    (d / "_runner.py").write_text(
        build_runner_source(data["entrypoint"], data.get("requirements", [])),
        encoding="utf-8",
    )
    argv = uv_run_argv(tool_id)  # raises FileNotFoundError if uv missing
    # Fresh log each run (single last-run record; overwrite).
    log_path = _log_path(tool_id)
    if log_path.exists():
        log_path.unlink()
    started = _now_iso()
    last = _write_last_run(
        tool_id,
        {"status": "running", "started_at": started, "finished_at": None,
         "exit_code": None, "inputs": kwargs},
    )
    from rengu_flow_ui.subprocess_util import popen_repo_subprocess

    header = f"--- toolbox tool {tool_id} ---\nCMD: {' '.join(argv)}\n\n".encode()
    proc, _log_f = popen_repo_subprocess(argv, log_path, log_header=header)
    _active[tool_id] = proc
    return last


def run_status(tool_id: str) -> dict[str, Any]:
    last = _read_last_run(tool_id)
    if last is None:
        return {"status": "idle"}
    proc = _active.get(tool_id)
    if last.get("status") == "running" and proc is not None:
        code = proc.poll()
        if code is not None:
            last["status"] = "done" if code == 0 else "failed"
            last["exit_code"] = code
            last["finished_at"] = _now_iso()
            _write_last_run(tool_id, last)
            _active.pop(tool_id, None)
    return last


def read_log(tool_id: str, offset: int = 0) -> tuple[str, int]:
    path = _log_path(tool_id)
    if not path.is_file():
        return "", 0
    raw = path.read_bytes()
    chunk = raw[offset:]
    return chunk.decode("utf-8", errors="replace"), len(raw)


def cancel_run(tool_id: str) -> None:
    proc = _active.get(tool_id)
    if proc is not None and proc.poll() is None:
        proc.terminate()
    last = _read_last_run(tool_id)
    if last and last.get("status") == "running":
        last["status"] = "failed"
        last["exit_code"] = -15
        last["finished_at"] = _now_iso()
        _write_last_run(tool_id, last)
    _active.pop(tool_id, None)


def build_runner_source(entrypoint: str, requirements: list[str]) -> str:
    deps = ", ".join(json.dumps(r) for r in requirements)
    entry = (entrypoint or DEFAULT_ENTRYPOINT).strip()
    entry_quoted = json.dumps(entry)
    return f'''# /// script
# requires-python = ">=3.11"
# dependencies = [{deps}]
# ///
import importlib.util
import json
from pathlib import Path

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("user_tool", here / "tool.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

kwargs = json.loads((here / "inputs.json").read_text(encoding="utf-8"))
result = getattr(mod, {entry_quoted})(**kwargs)
if result is not None:
    print(result)
'''
