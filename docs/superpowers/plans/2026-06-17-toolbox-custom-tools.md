# Toolbox — Custom Python Tools — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level **Toolbox** section where users author small Python tools (script + manually-declared input controls), persisted as a personal toolset, and run them via `uv` with PEP 723 inline requirements, isolated from rengu's venv, with a single last-run record per tool and a live log.

**Architecture:** A new backend storage module (`toolbox.py`) owns one folder per tool under `data/toolbox/<id>/` (script, metadata, single last-run). A routes module (`toolbox_routes.py`) exposes CRUD + run + log + a WebSocket log tail, registered in `app.py` next to `register_prep_routes`. Running generates a `_runner.py` with a PEP 723 header from the tool's `requirements`, writes `inputs.json`, and launches `uv run --no-project --isolated _runner.py` via the existing `popen_repo_subprocess`. An in-process registry tracks the single active `Popen` per tool (no DB, no queue). The frontend adds a Toolbox nav item, a list view, an editor view (with an inputs builder), and a run panel reusing the `useJobLogStream` WS pattern. Execution is gated by `[toolbox].enabled` in `rengu.local.toml`; authoring/saving always work.

**Tech Stack:** Python 3.11+, FastAPI, Starlette TestClient, `uv` CLI; Vue 3 + TypeScript + Element Plus + Vite (frontend); pytest.

## Global Constraints

- All repo code, docs, UI strings, and tool labels in **English**; Spanish only in conversation.
- No external users yet → config/JSON/TOML schemas may change freely; no backcompat/migration maps. Drop unknown/stale fields gracefully.
- App/recovery artifacts live under the managed data dir (`RENGU_FLOW_UI_DATA` or `<repo>/data`) — **never** hidden dotfolders in user data.
- No silent fallback on expected errors: use the capable path directly; surface failures (e.g. uv resolution errors) in the log, never swallow them.
- `uv` is a system requirement; resolve dependency isolation through `uv run` flags in the subprocess, never by degrading rengu's deps.
- Run pytest from the worktree root with: `PYTHONPATH="$PWD" uv run --extra dev pytest ...` (else it tests the main checkout or escapes to system Python).
- Commit conventions: author `koronos` with **no email**; no sensitive env data or personal paths in commits. One commit per task. No version bump unless asked.
- Verify via automated tests only. `uv run` smoke tests can look hung but aren't — arm a Monitor + timeout on any background run.
- API prefix is `/api/v1` (constant `API_PREFIX` in route modules). Existing log WS auth: token passed as `?token=` query param.

---

## File Structure

**Backend (new):**
- `rengu_flow_ui/toolbox.py` — storage + run engine: tool CRUD over `data/toolbox/<id>/`, slug-id, input→kwargs casting, `_runner.py` generation, launch/status/cancel, log snapshot/tail. Single responsibility: the Toolbox domain logic, no HTTP.
- `rengu_flow_ui/toolbox_routes.py` — FastAPI routes (`register_toolbox_routes(app)`), thin HTTP layer over `toolbox.py`, including the WS log route.

**Backend (modified):**
- `rengu_flow/config/local_config.py` — add `ToolboxConfig` + parse `[toolbox]` + `toolbox_enabled()`.
- `rengu_flow_ui/settings.py` — add `toolbox_dir()`; include it in `ensure_data_dirs()`.
- `rengu_flow_ui/app.py` — call `register_toolbox_routes(app)`.

**Frontend (new):**
- `ui/web/src/views/ToolboxView.vue` — tool list.
- `ui/web/src/views/ToolboxToolFormView.vue` — tool editor + inputs builder.
- `ui/web/src/components/ToolboxRunPanel.vue` — run form + live log.

**Frontend (modified):**
- `ui/web/src/api.ts` — Toolbox API methods + TS types.
- `ui/web/src/router.ts` — Toolbox routes.
- `ui/web/src/App.vue` — Toolbox nav item (main menu + mobile drawer).

**Tests (new):**
- `tests/test_toolbox.py` — storage + run-engine unit tests.
- `tests/test_toolbox_routes.py` — route + toggle tests via TestClient.
- `tests/test_toolbox_smoke.py` — end-to-end uv run smoke (no requirements).
- `ui/web/src/composables/*` — reuse existing; no new FE unit tests required for v1 (covered by backend + manual), but add types only.

---

## Task 1: Local TOML toggle — `[toolbox].enabled`

**Files:**
- Modify: `rengu_flow/config/local_config.py`
- Test: `tests/test_toolbox.py`

**Interfaces:**
- Produces: `ToolboxConfig(enabled: bool = False)` dataclass; `LocalConfig.toolbox: ToolboxConfig`; module-level `toolbox_enabled() -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_toolbox.py
from pathlib import Path

from rengu_flow.config import local_config as lc


def test_toolbox_enabled_defaults_false_when_section_absent(tmp_path: Path):
    cfg = lc.parse_local_config_dict({}, root=tmp_path)
    assert cfg.toolbox.enabled is False


def test_toolbox_enabled_reads_truthy_values(tmp_path: Path):
    cfg = lc.parse_local_config_dict({"toolbox": {"enabled": "on"}}, root=tmp_path)
    assert cfg.toolbox.enabled is True

    cfg2 = lc.parse_local_config_dict({"toolbox": {"enabled": True}}, root=tmp_path)
    assert cfg2.toolbox.enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k toolbox_enabled -v`
Expected: FAIL with `AttributeError: 'LocalConfig' object has no attribute 'toolbox'`.

- [ ] **Step 3: Add the dataclass, parsing, and helper**

In `rengu_flow/config/local_config.py`, add the dataclass next to the other config dataclasses:

```python
@dataclass
class ToolboxConfig:
    enabled: bool = False
```

Add the field to `LocalConfig` (alongside `ui`, `maintenance`, `training`):

```python
    toolbox: ToolboxConfig = field(default_factory=ToolboxConfig)
```

In `parse_local_config_dict`, after the `training = TrainingConfig(...)` block and before the `return`:

```python
    toolbox_raw = data.get("toolbox") if isinstance(data.get("toolbox"), dict) else {}
    toolbox = ToolboxConfig(enabled=_boolish(toolbox_raw.get("enabled", False)))
```

Update the return to include it:

```python
    return LocalConfig(
        root=root, ui=ui, maintenance=maintenance, training=training, toolbox=toolbox
    )
```

At module level (near `get_local_config`), add the helper:

```python
def toolbox_enabled() -> bool:
    """True when Toolbox tool *execution* is allowed (authoring is always allowed)."""
    return ensure_local_config_loaded().toolbox.enabled
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k toolbox_enabled -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add rengu_flow/config/local_config.py tests/test_toolbox.py
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): [toolbox].enabled local-config toggle"
```

---

## Task 2: Data dir — `toolbox_dir()`

**Files:**
- Modify: `rengu_flow_ui/settings.py`
- Test: `tests/test_toolbox.py`

**Interfaces:**
- Produces: `settings.toolbox_dir() -> Path` returning `ui_data_dir() / "toolbox"`; created by `ensure_data_dirs()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_toolbox.py  (append)
def test_toolbox_dir_under_ui_data_dir(ui_data_tmp):
    from rengu_flow_ui import settings

    assert settings.toolbox_dir() == ui_data_tmp / "toolbox"
    settings.ensure_data_dirs()
    assert settings.toolbox_dir().is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k toolbox_dir -v`
Expected: FAIL with `AttributeError: module 'rengu_flow_ui.settings' has no attribute 'toolbox_dir'`.

- [ ] **Step 3: Add the helper and wire it into `ensure_data_dirs`**

In `rengu_flow_ui/settings.py`, next to `staging_dir`/`logs_dir`:

```python
def toolbox_dir() -> Path:
    return ui_data_dir() / "toolbox"
```

Add it to `ensure_data_dirs`:

```python
def ensure_data_dirs() -> None:
    for d in (ui_data_dir(), staging_dir(), logs_dir(), toolbox_dir()):
        d.mkdir(parents=True, exist_ok=True)
```

Note: `tests/conftest.py::_patch_ui_data_paths` monkeypatches `ui_data_dir`, so `toolbox_dir()` follows the temp dir automatically (it is derived from `ui_data_dir()`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k toolbox_dir -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rengu_flow_ui/settings.py tests/test_toolbox.py
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): toolbox data dir under managed data/"
```

---

## Task 3: Tool model + slug id + CRUD storage

**Files:**
- Create: `rengu_flow_ui/toolbox.py`
- Test: `tests/test_toolbox.py`

**Interfaces:**
- Consumes: `settings.toolbox_dir()`.
- Produces (all in `rengu_flow_ui.toolbox`):
  - `slugify(name: str) -> str`
  - `ToolInput` TypedDict-like dict with keys `param, label, control, default, options, min, max, step, hint`.
  - `create_tool(name: str, description: str = "", entrypoint: str = "run", requirements: list[str] | None = None, script: str = "", inputs: list[dict] | None = None) -> dict` → returns the saved `tool.json` dict (with `id`, `created_at`, `updated_at`).
  - `list_tools() -> list[dict]` → summaries: `{id, name, description, created_at, updated_at, last_run_status}`.
  - `get_tool(tool_id: str) -> dict` → full `tool.json` plus `script` (from `tool.py`) and `last_run` (or `None`). Raises `KeyError` if missing.
  - `update_tool(tool_id: str, **fields) -> dict` → updates `tool.json`/`tool.py`, touches `updated_at`.
  - `delete_tool(tool_id: str) -> None`.
  - `tool_dir(tool_id: str) -> Path`.
- Time source: pass an injectable `now: str` is overkill; use `datetime.now(timezone.utc)` formatted ISO-8601 with trailing `Z`. Tests assert format/ordering, not exact value.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_toolbox.py  (append)
import json

import pytest


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k "roundtrip or dedupes or list_and_delete or missing_tool" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rengu_flow_ui.toolbox'`.

- [ ] **Step 3: Implement the storage module**

Create `rengu_flow_ui/toolbox.py`:

```python
"""Toolbox: user-authored Python tools persisted under ``data/toolbox/<id>/``.

One folder per tool. A single last-run record per tool (no queue, no history).
Authoring is always allowed; execution is gated by ``[toolbox].enabled`` in the
local TOML (see ``rengu_flow.config.local_config.toolbox_enabled``).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rengu_flow_ui import settings

CONTROL_TYPES = ("number", "text", "textarea", "switch", "select")
DEFAULT_ENTRYPOINT = "run"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
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
        last = _read_last_run(d.name)
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
    import shutil

    d = tool_dir(tool_id)
    if not d.is_dir():
        raise KeyError(tool_id)
    shutil.rmtree(d)


def _read_last_run(tool_id: str) -> dict[str, Any] | None:
    path = tool_dir(tool_id) / "last_run.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k "roundtrip or dedupes or list_and_delete or missing_tool" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rengu_flow_ui/toolbox.py tests/test_toolbox.py
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): tool storage CRUD over data/toolbox/<id>/"
```

---

## Task 4: Input→kwargs casting + `_runner.py` generation

**Files:**
- Modify: `rengu_flow_ui/toolbox.py`
- Test: `tests/test_toolbox.py`

**Interfaces:**
- Consumes: a tool's `inputs` list and `entrypoint`/`requirements` (from `tool.json`).
- Produces (in `rengu_flow_ui.toolbox`):
  - `cast_inputs(inputs_def: list[dict], values: dict) -> dict` → typed kwargs; raises `ValueError` on missing/unparseable.
  - `build_runner_source(entrypoint: str, requirements: list[str]) -> str` → the `_runner.py` text (PEP 723 header + dispatch).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_toolbox.py  (append)
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
    import pytest

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k "cast_inputs or runner_source" -v`
Expected: FAIL with `AttributeError: module 'rengu_flow_ui.toolbox' has no attribute 'cast_inputs'`.

- [ ] **Step 3: Implement casting and runner generation**

Append to `rengu_flow_ui/toolbox.py`:

```python
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


def build_runner_source(entrypoint: str, requirements: list[str]) -> str:
    deps = ", ".join(json.dumps(r) for r in requirements)
    entry = (entrypoint or DEFAULT_ENTRYPOINT).strip()
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
result = getattr(mod, {entry!r})(**kwargs)
if result is not None:
    print(result)
'''
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k "cast_inputs or runner_source" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rengu_flow_ui/toolbox.py tests/test_toolbox.py
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): input casting and PEP 723 runner generation"
```

---

## Task 5: Run engine — launch, status, log tail, cancel

**Files:**
- Modify: `rengu_flow_ui/toolbox.py`
- Test: `tests/test_toolbox.py`

**Interfaces:**
- Consumes: `subprocess_util.popen_repo_subprocess`, `cast_inputs`, `build_runner_source`, `rengu_flow.config.local_config.toolbox_enabled`, `shutil.which("uv")`.
- Produces (in `rengu_flow_ui.toolbox`):
  - `ExecutionDisabledError(RuntimeError)`, `RunActiveError(RuntimeError)`.
  - `run_tool(tool_id: str, values: dict) -> dict` → writes `inputs.json`, `_runner.py`, `last_run.json` (status `running`), launches `uv run --no-project --isolated _runner.py`, registers the `Popen`, returns the `last_run` dict. Raises `ExecutionDisabledError` if `toolbox_enabled()` is False, `RunActiveError` if a run is already active, `KeyError` if tool missing, `FileNotFoundError` if `uv` missing.
  - `run_status(tool_id: str) -> dict` → reaps the process if finished (writes terminal `last_run.json` with `status` `done`/`failed` and `exit_code`); returns `last_run` dict (or `{"status": "idle"}`).
  - `read_log(tool_id: str, offset: int = 0) -> tuple[str, int]` → returns `(chunk, new_offset)` from `last_run.log`.
  - `cancel_run(tool_id: str) -> None` → terminates the active process.
  - `uv_run_argv(tool_id: str) -> list[str]` → the exact command (for testability).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_toolbox.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k "execution_disabled or uv_run_argv" -v`
Expected: FAIL with `AttributeError: module 'rengu_flow_ui.toolbox' has no attribute 'ExecutionDisabledError'`.

- [ ] **Step 3: Implement the run engine**

Append to `rengu_flow_ui/toolbox.py`:

```python
import shutil
import subprocess  # noqa: E402  (grouped with run-engine code)

from rengu_flow.config import local_config

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
```

Update `_read_last_run` is already defined in Task 3; do not redefine it. (If a duplicate exists, keep the Task 3 version.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -k "execution_disabled or uv_run_argv" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full module test file**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py -v`
Expected: PASS (all tasks 1–5 tests).

- [ ] **Step 6: Commit**

```bash
git add rengu_flow_ui/toolbox.py tests/test_toolbox.py
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): run engine (launch/status/log/cancel) via uv"
```

---

## Task 6: HTTP routes + WS log + app registration

**Files:**
- Create: `rengu_flow_ui/toolbox_routes.py`
- Modify: `rengu_flow_ui/app.py` (register routes)
- Test: `tests/test_toolbox_routes.py`

**Interfaces:**
- Consumes: everything from `rengu_flow_ui.toolbox`; `rengu_flow.config.local_config.toolbox_enabled`.
- Produces: `register_toolbox_routes(app: FastAPI) -> None`; the routes listed in the spec under `/api/v1/toolbox/...`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_toolbox_routes.py
import pytest


def _create(ui_client, **body):
    body.setdefault("name", "My Tool")
    res = ui_client.post("/api/v1/toolbox/tools", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_crud_works_even_when_execution_disabled(ui_client, monkeypatch):
    from rengu_flow.config import local_config as lc

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: False)

    created = _create(ui_client, name="Sumar", script="def run(a, b):\n    return a+b\n")
    tool_id = created["id"]

    listed = ui_client.get("/api/v1/toolbox/tools").json()
    assert any(t["id"] == tool_id for t in listed)

    got = ui_client.get(f"/api/v1/toolbox/tools/{tool_id}").json()
    assert got["script"].startswith("def run(")

    upd = ui_client.put(
        f"/api/v1/toolbox/tools/{tool_id}", json={"description": "updated"}
    )
    assert upd.status_code == 200 and upd.json()["description"] == "updated"

    deleted = ui_client.delete(f"/api/v1/toolbox/tools/{tool_id}")
    assert deleted.status_code == 200


def test_run_returns_409_when_execution_disabled(ui_client, monkeypatch):
    from rengu_flow.config import local_config as lc

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: False)
    created = _create(ui_client, name="Sumar", script="def run():\n    return 1\n")
    res = ui_client.post(f"/api/v1/toolbox/tools/{created['id']}/run", json={"values": {}})
    assert res.status_code == 409, res.text


def test_enabled_endpoint_reflects_toggle(ui_client, monkeypatch):
    from rengu_flow.config import local_config as lc

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: True)
    assert ui_client.get("/api/v1/toolbox/enabled").json() == {"enabled": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox_routes.py -v`
Expected: FAIL — 404s (routes not registered).

- [ ] **Step 3: Implement the routes module**

Create `rengu_flow_ui/toolbox_routes.py`:

```python
"""HTTP + WebSocket routes for the Toolbox section. Thin layer over ``toolbox.py``."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from rengu_flow.config import local_config
from rengu_flow_ui import toolbox
from rengu_flow_ui.app_auth import ui_token  # see note in Step 3a

API_PREFIX = "/api/v1"


class ToolBody(BaseModel):
    name: str = "Untitled tool"
    description: str = ""
    entrypoint: str = "run"
    requirements: list[str] = Field(default_factory=list)
    script: str = ""
    inputs: list[dict] = Field(default_factory=list)


class ToolUpdateBody(BaseModel):
    name: str | None = None
    description: str | None = None
    entrypoint: str | None = None
    requirements: list[str] | None = None
    script: str | None = None
    inputs: list[dict] | None = None


class RunBody(BaseModel):
    values: dict = Field(default_factory=dict)


@contextmanager
def _http_errors():
    try:
        yield
    except KeyError:
        raise HTTPException(404, "Tool not found")
    except toolbox.ExecutionDisabledError as e:
        raise HTTPException(409, str(e))
    except toolbox.RunActiveError as e:
        raise HTTPException(409, str(e))
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


def register_toolbox_routes(app: FastAPI) -> None:
    @app.get(f"{API_PREFIX}/toolbox/enabled")
    def toolbox_enabled_route() -> dict[str, bool]:
        return {"enabled": local_config.toolbox_enabled()}

    @app.get(f"{API_PREFIX}/toolbox/tools")
    def list_toolbox_tools() -> list[dict]:
        return toolbox.list_tools()

    @app.post(f"{API_PREFIX}/toolbox/tools")
    def create_toolbox_tool(body: ToolBody) -> dict:
        with _http_errors():
            return toolbox.create_tool(
                name=body.name,
                description=body.description,
                entrypoint=body.entrypoint,
                requirements=body.requirements,
                script=body.script,
                inputs=body.inputs,
            )

    @app.get(f"{API_PREFIX}/toolbox/tools/{{tool_id}}")
    def get_toolbox_tool(tool_id: str) -> dict:
        with _http_errors():
            return toolbox.get_tool(tool_id)

    @app.put(f"{API_PREFIX}/toolbox/tools/{{tool_id}}")
    def update_toolbox_tool(tool_id: str, body: ToolUpdateBody) -> dict:
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        with _http_errors():
            return toolbox.update_tool(tool_id, **fields)

    @app.delete(f"{API_PREFIX}/toolbox/tools/{{tool_id}}")
    def delete_toolbox_tool(tool_id: str) -> dict:
        with _http_errors():
            toolbox.delete_tool(tool_id)
        return {"ok": True}

    @app.post(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/run")
    def run_toolbox_tool(tool_id: str, body: RunBody) -> dict:
        with _http_errors():
            return toolbox.run_tool(tool_id, body.values)

    @app.get(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/run")
    def toolbox_run_status(tool_id: str) -> dict:
        with _http_errors():
            return toolbox.run_status(tool_id)

    @app.get(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/log")
    def toolbox_log(tool_id: str, offset: int = 0) -> dict:
        chunk, new_offset = toolbox.read_log(tool_id, offset)
        status = toolbox.run_status(tool_id).get("status", "idle")
        return {"chunk": chunk, "offset": new_offset, "status": status}

    @app.post(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/run/cancel")
    def toolbox_cancel(tool_id: str) -> dict:
        with _http_errors():
            toolbox.cancel_run(tool_id)
        return {"ok": True}

    @app.websocket(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/log/ws")
    async def toolbox_log_ws(websocket: WebSocket, tool_id: str) -> None:
        token = ui_token()
        if token and websocket.query_params.get("token", "") != token:
            await websocket.close(code=4401, reason="Invalid token")
            return
        await websocket.accept()
        offset = 0
        try:
            while True:
                chunk, offset = await asyncio.to_thread(toolbox.read_log, tool_id, offset)
                if chunk:
                    await websocket.send_text(chunk)
                status = await asyncio.to_thread(
                    lambda: toolbox.run_status(tool_id).get("status", "idle")
                )
                if status != "running":
                    chunk, offset = await asyncio.to_thread(toolbox.read_log, tool_id, offset)
                    if chunk:
                        await websocket.send_text(chunk)
                    break
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
```

- [ ] **Step 3a: Resolve the `ui_token` import**

The token helper used by the existing logs WS lives in `app.py` (search for `def ui_token`). Import it from its actual module. Run:

`grep -rn "def ui_token" rengu_flow_ui/`

Then set the import in `toolbox_routes.py` to the module that defines it (e.g. `from rengu_flow_ui.app import ui_token` or its real home). If it is defined in `app.py` and importing it at module load causes a circular import, import it lazily inside `toolbox_log_ws` instead:

```python
        from rengu_flow_ui.app import ui_token
        token = ui_token()
```

Remove the top-level `from rengu_flow_ui.app_auth import ui_token` line if you use the lazy import.

- [ ] **Step 3b: Register in `app.py`**

In `rengu_flow_ui/app.py`, next to the existing `register_prep_routes(app)` call:

```python
from rengu_flow_ui.toolbox_routes import register_toolbox_routes

register_toolbox_routes(app)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox_routes.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rengu_flow_ui/toolbox_routes.py rengu_flow_ui/app.py tests/test_toolbox_routes.py
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): HTTP + WS routes registered in app"
```

---

## Task 7: End-to-end uv smoke test

**Files:**
- Create: `tests/test_toolbox_smoke.py`

**Interfaces:**
- Consumes: `toolbox.create_tool`, `toolbox.run_tool`, `toolbox.run_status`, `toolbox.read_log`; requires `uv` on PATH and `[toolbox].enabled` patched True.

- [ ] **Step 1: Write the test**

```python
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
```

- [ ] **Step 2: Run the test (arm a Monitor + timeout — uv first run can take ~1 min)**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox_smoke.py -v` with a 180s timeout and a Monitor watching for completion. (`uv run` looks hung during first resolution but is not.)
Expected: PASS (or SKIP if `uv` absent).

- [ ] **Step 3: Commit**

```bash
git add tests/test_toolbox_smoke.py
git -c user.name=koronos -c user.email= commit -m "test(toolbox): end-to-end uv run smoke test"
```

---

## Task 8: Frontend — API client + types

**Files:**
- Modify: `ui/web/src/api.ts`

**Interfaces:**
- Produces (on the exported `api` object): `toolboxEnabled`, `listToolboxTools`, `createToolboxTool`, `getToolboxTool`, `updateToolboxTool`, `deleteToolboxTool`, `runToolboxTool`, `toolboxRunStatus`, `toolboxLog`, `cancelToolboxRun`. Plus exported TS types `ToolboxToolSummary`, `ToolboxTool`, `ToolboxInput`, `ToolboxRun`.

- [ ] **Step 1: Add the types and methods**

In `ui/web/src/api.ts`, add the types near the other exported interfaces:

```typescript
export interface ToolboxInput {
  param: string;
  label: string;
  control: "number" | "text" | "textarea" | "switch" | "select";
  default?: unknown;
  options?: string[];
  min?: number | null;
  max?: number | null;
  step?: number | null;
  hint?: string;
}

export interface ToolboxToolSummary {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  last_run_status: string;
}

export interface ToolboxRun {
  status: string;
  started_at?: string;
  finished_at?: string | null;
  exit_code?: number | null;
  inputs?: Record<string, unknown>;
}

export interface ToolboxTool {
  id: string;
  name: string;
  description: string;
  entrypoint: string;
  requirements: string[];
  inputs: ToolboxInput[];
  script: string;
  created_at: string;
  updated_at: string;
  last_run: ToolboxRun | null;
}

export interface ToolboxToolWrite {
  name: string;
  description?: string;
  entrypoint?: string;
  requirements?: string[];
  script?: string;
  inputs?: ToolboxInput[];
}
```

Add the methods inside the `export const api = { ... }` object:

```typescript
  toolboxEnabled: () => request<{ enabled: boolean }>("/toolbox/enabled"),
  listToolboxTools: () => request<ToolboxToolSummary[]>("/toolbox/tools"),
  createToolboxTool: (body: ToolboxToolWrite) =>
    request<ToolboxTool>("/toolbox/tools", { method: "POST", body: JSON.stringify(body) }),
  getToolboxTool: (id: string) => request<ToolboxTool>(`/toolbox/tools/${id}`),
  updateToolboxTool: (id: string, body: Partial<ToolboxToolWrite>) =>
    request<ToolboxTool>(`/toolbox/tools/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteToolboxTool: (id: string) =>
    request<{ ok: boolean }>(`/toolbox/tools/${id}`, { method: "DELETE" }),
  runToolboxTool: (id: string, values: Record<string, unknown>) =>
    request<ToolboxRun>(`/toolbox/tools/${id}/run`, {
      method: "POST",
      body: JSON.stringify({ values }),
    }),
  toolboxRunStatus: (id: string) => request<ToolboxRun>(`/toolbox/tools/${id}/run`),
  toolboxLog: (id: string, offset = 0) =>
    request<{ chunk: string; offset: number; status: string }>(
      `/toolbox/tools/${id}/log?offset=${offset}`,
    ),
  cancelToolboxRun: (id: string) =>
    request<{ ok: boolean }>(`/toolbox/tools/${id}/run/cancel`, { method: "POST" }),
```

- [ ] **Step 2: Typecheck**

Run: `cd ui/web && npm run type-check` (or the repo's configured TS check; if unsure, run `npx vue-tsc --noEmit`).
Expected: no new type errors.

- [ ] **Step 3: Commit**

```bash
git add ui/web/src/api.ts
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): frontend API client methods and types"
```

---

## Task 9: Frontend — routes + nav item

**Files:**
- Modify: `ui/web/src/router.ts`
- Modify: `ui/web/src/App.vue`

**Interfaces:**
- Consumes: `ToolboxView.vue`, `ToolboxToolFormView.vue` (created in Task 10 — routes can reference them now; the dev server resolves lazily, but to keep the build green, create the view stubs first if building before Task 10).

- [ ] **Step 1: Add routes**

In `ui/web/src/router.ts`, add to the `routes` array:

```typescript
{ path: "/toolbox", name: "toolbox", component: () => import("./views/ToolboxView.vue") },
{
  path: "/toolbox/new",
  name: "toolbox-new",
  component: () => import("./views/ToolboxToolFormView.vue"),
},
{
  path: "/toolbox/:id/edit",
  name: "toolbox-edit",
  component: () => import("./views/ToolboxToolFormView.vue"),
},
```

- [ ] **Step 2: Add the nav item (main menu + mobile drawer)**

In `ui/web/src/App.vue`, add after the Studio `<el-menu-item index="/prep">` block, in BOTH the main `el-menu` and the drawer `el-menu`:

```vue
<el-menu-item index="/toolbox">
  <el-icon><Tools /></el-icon>
  <span>Toolbox</span>
</el-menu-item>
```

Ensure `Tools` is imported from `@element-plus/icons-vue` at the top of the `<script setup>` block (add it to the existing icon import list).

- [ ] **Step 3: Verify build resolves (after Task 10 views exist)**

Run: `cd ui/web && npm run build`
Expected: build succeeds. (If running this before Task 10, create empty `<template><div/></template>` stubs for the two views to keep the build green, then flesh them out in Task 10.)

- [ ] **Step 4: Commit**

```bash
git add ui/web/src/router.ts ui/web/src/App.vue
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): nav item and routes"
```

---

## Task 10: Frontend — list view, editor view, run panel

**Files:**
- Create: `ui/web/src/views/ToolboxView.vue`
- Create: `ui/web/src/views/ToolboxToolFormView.vue`
- Create: `ui/web/src/components/ToolboxRunPanel.vue`

**Interfaces:**
- Consumes: `api.*` methods and types from Task 8; Element Plus components; the existing `useJobLogStream` composable pattern (`ui/web/src/composables/useJobLogStream.ts`) as a reference for the REST-snapshot-then-WS log streaming. Since the Toolbox WS path differs (`/toolbox/tools/{id}/log/ws`) and frames are raw text, the run panel may either adapt `useJobLogStream` or open the WS directly with the same shape (open WS, append text frames, fall back to polling `api.toolboxLog`).

- [ ] **Step 1: Implement `ToolboxView.vue` (list)**

```vue
<template>
  <div class="toolbox-view">
    <div class="toolbar">
      <h2>Toolbox</h2>
      <el-button type="primary" @click="$router.push('/toolbox/new')">New tool</el-button>
    </div>
    <el-alert
      v-if="!executionEnabled"
      type="info"
      :closable="false"
      title="Execution disabled in rengu.local.toml → [toolbox].enabled. You can still create and edit tools."
    />
    <el-empty v-if="tools.length === 0" description="No tools yet" />
    <div v-else class="tool-grid">
      <el-card v-for="t in tools" :key="t.id" class="tool-card">
        <div class="tool-card__head">
          <strong>{{ t.name }}</strong>
          <el-tag size="small" :type="statusTagType(t.last_run_status)">{{ t.last_run_status }}</el-tag>
        </div>
        <p class="tool-card__desc">{{ t.description }}</p>
        <p class="tool-card__dates">
          Created {{ t.created_at }} · Modified {{ t.updated_at }}
        </p>
        <div class="tool-card__actions">
          <el-button size="small" @click="$router.push(`/toolbox/${t.id}/edit`)">Edit</el-button>
          <el-button size="small" type="danger" @click="remove(t.id)">Delete</el-button>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ElMessageBox } from "element-plus";
import { api, type ToolboxToolSummary } from "../api";

const tools = ref<ToolboxToolSummary[]>([]);
const executionEnabled = ref(true);

function statusTagType(s: string): string {
  return s === "done" ? "success" : s === "failed" ? "danger" : s === "running" ? "warning" : "info";
}

async function load() {
  tools.value = await api.listToolboxTools();
  executionEnabled.value = (await api.toolboxEnabled()).enabled;
}

async function remove(id: string) {
  await ElMessageBox.confirm("Delete this tool?", "Confirm", { type: "warning" });
  await api.deleteToolboxTool(id);
  await load();
}

onMounted(load);
</script>
```

- [ ] **Step 2: Implement `ToolboxToolFormView.vue` (editor + inputs builder)**

```vue
<template>
  <div class="toolbox-form">
    <h2>{{ isEdit ? "Edit tool" : "New tool" }}</h2>
    <el-form label-position="top">
      <el-form-item label="Name">
        <el-input v-model="form.name" placeholder="e.g. Resize dataset images" />
      </el-form-item>
      <el-form-item label="Description">
        <el-input v-model="form.description" placeholder="What this tool does" />
      </el-form-item>
      <el-form-item label="Entrypoint function">
        <el-input v-model="form.entrypoint" placeholder="run" />
      </el-form-item>
      <el-form-item label="Required packages (one per line — uv resolves inline)">
        <el-input v-model="requirementsText" type="textarea" :rows="3" placeholder="e.g. pillow&#10;numpy>=2.0" />
      </el-form-item>
      <el-form-item label="Python script">
        <el-input v-model="form.script" type="textarea" :rows="14" placeholder="def run(num1, num2):&#10;    return num1 + num2" />
      </el-form-item>

      <h3>Inputs</h3>
      <p class="hint">Each input maps to a keyword argument of your entrypoint function.</p>
      <div v-for="(inp, i) in form.inputs" :key="i" class="input-row">
        <el-input v-model="inp.param" placeholder="param name" style="width: 140px" />
        <el-input v-model="inp.label" placeholder="label" style="width: 160px" />
        <el-select v-model="inp.control" style="width: 130px">
          <el-option v-for="c in controls" :key="c" :label="c" :value="c" />
        </el-select>
        <el-input
          v-if="inp.control === 'select'"
          v-model="optionsText[i]"
          placeholder="opt1, opt2"
          style="width: 180px"
          @input="syncOptions(i)"
        />
        <el-input v-model="inp.hint" placeholder="hint" style="flex: 1" />
        <el-button size="small" type="danger" @click="removeInput(i)">×</el-button>
      </div>
      <el-button size="small" @click="addInput">Add input</el-button>

      <div class="form-actions">
        <el-button type="primary" @click="save">Save</el-button>
        <el-button @click="$router.push('/toolbox')">Cancel</el-button>
      </div>
    </el-form>

    <ToolboxRunPanel v-if="isEdit && savedId" :tool-id="savedId" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api, type ToolboxInput, type ToolboxToolWrite } from "../api";
import ToolboxRunPanel from "../components/ToolboxRunPanel.vue";

const route = useRoute();
const router = useRouter();
const controls = ["number", "text", "textarea", "switch", "select"] as const;

const isEdit = computed(() => Boolean(route.params.id));
const savedId = ref<string | null>((route.params.id as string) || null);

const form = reactive<ToolboxToolWrite>({
  name: "",
  description: "",
  entrypoint: "run",
  requirements: [],
  script: "",
  inputs: [],
});
const requirementsText = ref("");
const optionsText = reactive<Record<number, string>>({});

function addInput() {
  form.inputs!.push({ param: "", label: "", control: "text", hint: "" } as ToolboxInput);
}
function removeInput(i: number) {
  form.inputs!.splice(i, 1);
}
function syncOptions(i: number) {
  form.inputs![i].options = (optionsText[i] || "").split(",").map((s) => s.trim()).filter(Boolean);
}

async function save() {
  form.requirements = requirementsText.value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (isEdit.value && savedId.value) {
    await api.updateToolboxTool(savedId.value, form);
  } else {
    const created = await api.createToolboxTool(form);
    savedId.value = created.id;
    router.replace(`/toolbox/${created.id}/edit`);
  }
}

onMounted(async () => {
  if (isEdit.value && savedId.value) {
    const t = await api.getToolboxTool(savedId.value);
    form.name = t.name;
    form.description = t.description;
    form.entrypoint = t.entrypoint;
    form.script = t.script;
    form.inputs = t.inputs;
    requirementsText.value = t.requirements.join("\n");
    t.inputs.forEach((inp, i) => {
      if (inp.options) optionsText[i] = inp.options.join(", ");
    });
  }
});
</script>
```

- [ ] **Step 3: Implement `ToolboxRunPanel.vue` (run form + live log)**

```vue
<template>
  <div class="run-panel">
    <h3>Run</h3>
    <el-alert
      v-if="!enabled"
      type="info"
      :closable="false"
      title="Execution disabled in rengu.local.toml → [toolbox].enabled"
    />
    <el-form label-position="top">
      <el-form-item v-for="inp in tool?.inputs || []" :key="inp.param" :label="inp.label || inp.param">
        <el-switch v-if="inp.control === 'switch'" v-model="values[inp.param]" />
        <el-input-number v-else-if="inp.control === 'number'" v-model="values[inp.param]" />
        <el-select v-else-if="inp.control === 'select'" v-model="values[inp.param]">
          <el-option v-for="o in inp.options || []" :key="o" :label="o" :value="o" />
        </el-select>
        <el-input v-else-if="inp.control === 'textarea'" v-model="values[inp.param]" type="textarea" />
        <el-input v-else v-model="values[inp.param]" />
        <span v-if="inp.hint" class="hint">{{ inp.hint }}</span>
      </el-form-item>
    </el-form>
    <el-button type="primary" :disabled="!enabled || running" @click="run">Run</el-button>
    <el-button v-if="running" @click="cancel">Cancel</el-button>
    <el-tag v-if="status" :type="statusType" size="small">{{ status }}</el-tag>
    <pre class="log">{{ log }}</pre>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";
import { api, type ToolboxTool } from "../api";
import { wsBaseUrl } from "../lib/wsLog";

const props = defineProps<{ toolId: string }>();

const tool = ref<ToolboxTool | null>(null);
const enabled = ref(true);
const values = reactive<Record<string, unknown>>({});
const log = ref("");
const status = ref("");
const running = computed(() => status.value === "running");
const statusType = computed(() =>
  status.value === "done" ? "success" : status.value === "failed" ? "danger" : "info",
);

let ws: WebSocket | null = null;
let offset = 0;

const statusType2 = statusType; // referenced in template
void statusType2;

async function loadSnapshot() {
  const r = await api.toolboxLog(props.toolId, 0);
  log.value = r.chunk;
  offset = r.offset;
  status.value = r.status;
}

function openWs() {
  ws?.close();
  ws = new WebSocket(`${wsBaseUrl()}/api/v1/toolbox/tools/${encodeURIComponent(props.toolId)}/log/ws`);
  ws.onmessage = (ev) => {
    log.value += ev.data as string;
  };
  ws.onclose = async () => {
    await refreshStatus();
  };
}

async function refreshStatus() {
  status.value = (await api.toolboxRunStatus(props.toolId)).status;
}

async function run() {
  log.value = "";
  offset = 0;
  await api.runToolboxTool(props.toolId, { ...values });
  status.value = "running";
  openWs();
}

async function cancel() {
  await api.cancelToolboxRun(props.toolId);
  await refreshStatus();
}

onMounted(async () => {
  tool.value = await api.getToolboxTool(props.toolId);
  enabled.value = (await api.toolboxEnabled()).enabled;
  for (const inp of tool.value.inputs) {
    if (inp.default !== undefined && inp.default !== null) values[inp.param] = inp.default;
  }
  if (tool.value.last_run?.inputs) Object.assign(values, tool.value.last_run.inputs);
  await loadSnapshot();
  if (status.value === "running") openWs();
});

onUnmounted(() => ws?.close());
</script>
```

Note: `wsBaseUrl` is exported from `ui/web/src/lib/wsLog.ts` (confirmed during exploration). If the named export differs, import the actual symbol used by `useJobLogStream.ts`.

- [ ] **Step 4: Build the frontend**

Run: `cd ui/web && npm run build`
Expected: build succeeds with no type errors.

- [ ] **Step 5: Commit**

```bash
git add ui/web/src/views/ToolboxView.vue ui/web/src/views/ToolboxToolFormView.vue ui/web/src/components/ToolboxRunPanel.vue
git -c user.name=koronos -c user.email= commit -m "feat(toolbox): list view, editor, and run panel"
```

---

## Task 11: User docs + example local TOML

**Files:**
- Create: `docs/user/toolbox.md` (or the repo's user-docs location — confirm by listing `docs/user/`)
- Modify: `rengu.local.toml` (add a commented `[toolbox]` example) — only if the repo tracks an example; otherwise document it in the markdown.

**Interfaces:** none (docs).

- [ ] **Step 1: Confirm docs location**

Run: `ls docs/user/ 2>/dev/null; ls docs/`
Place the doc next to existing user docs; if there is a developer-docs conventions file, follow its hint/heading style (per the repo's documentation-conventions).

- [ ] **Step 2: Write `docs/user/toolbox.md`**

Cover, concisely (mechanism → effect → trigger, per the repo's hint rules):
- What Toolbox is (personal Python tools run by uv, isolated from rengu's venv).
- Enabling execution: add `[toolbox]\nenabled = true` to `rengu.local.toml`; default off; authoring works while off.
- Authoring: name an entrypoint function (default `run`); declare inputs and map each to a `run(...)` parameter; add required packages (resolved inline by uv).
- The single last-run record (overwritten each run; no history).
- Example: a `sumar(num1, num2)` tool.

- [ ] **Step 3: Commit**

```bash
git add docs/user/toolbox.md rengu.local.toml
git -c user.name=koronos -c user.email= commit -m "docs(toolbox): user guide and local TOML toggle example"
```

---

## Task 12: Full regression + cleanup

**Files:** none (verification).

- [ ] **Step 1: Run the full backend test suite**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_toolbox.py tests/test_toolbox_routes.py tests/test_toolbox_smoke.py -v` with a Monitor + 240s timeout.
Expected: all PASS (smoke may SKIP without `uv`).

- [ ] **Step 2: Run the broader suite to check for regressions**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/ -q -k "local_config or settings or app or prep"` with a Monitor + timeout.
Expected: no new failures introduced by the changes.

- [ ] **Step 3: Frontend build + typecheck**

Run: `cd ui/web && npm run build`
Expected: success.

- [ ] **Step 4: Final review**

Review the full diff with `git -c user.name=koronos diff main...HEAD` against the spec's goals/non-goals. Confirm: execution gated by `[toolbox].enabled`; CRUD works when disabled; uv run is `--no-project --isolated`; one folder per tool; single last-run; English-only strings.

---

## Self-Review (author checklist — completed)

**Spec coverage:** toggle (T1), data dir (T2), storage CRUD (T3), casting + runner (T4), run engine (T5), routes + WS (T6), uv smoke (T7), FE API (T8), nav/routes (T9), views (T10), docs (T11), regression (T12). All spec sections map to a task.

**Placeholder scan:** Two confirm-then-implement steps (T6 Step 3a `ui_token` import home, T11 Step 1 docs location) are deliberate verification steps with explicit grep/ls commands and fallbacks, not unresolved placeholders.

**Type consistency:** `create_tool/get_tool/update_tool/delete_tool/list_tools/run_tool/run_status/read_log/cancel_run/uv_run_argv/cast_inputs/build_runner_source/slugify/tool_dir` are used with consistent signatures across backend tasks; `ToolboxTool/ToolboxToolSummary/ToolboxInput/ToolboxRun/ToolboxToolWrite` consistent across FE tasks; route paths consistent with the spec table.
