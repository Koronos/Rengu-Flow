# Toolbox — Custom Python Tools — Design

**Date:** 2026-06-17
**Status:** Approved (pre-implementation), iterating from here.

## Summary

A new top-level **Toolbox** section that lets users author small Python "tools":
paste a Python script that exposes an entrypoint function, declare its input
controls in the UI, and run it via `uv` with PEP 723 inline requirements — fully
isolated from rengu's own virtualenv. Tools form a personal, always-at-hand
toolset shown with name, description, and creation/modification dates, intended
for custom dataset treatment or other ad-hoc processing.

Toolbox is **separate from the Studio (`/prep`) section**. Studio is built around
queued jobs and TOML configs; Toolbox is not — it has no queue, no TOML job
config, and a single last-run record per tool. They share only low-level
infrastructure (the data dir, the subprocess launcher, and the live-log WebSocket
hub).

Execution is gated by a flag in the local TOML. The section, authoring, and
saving remain available even when execution is disabled; only running is blocked.

## Goals

- Let a user paste a Python script with a named entrypoint (default `run`) and map
  its parameters to UI input controls (manual binding, declared by the user).
- Run the script with `uv run --no-project --isolated` so a `requirements` field
  becomes PEP 723 inline dependencies that uv resolves and caches in an ephemeral
  environment — never touching rengu's venv.
- Surface a live log (REST snapshot + WebSocket incremental updates) using the
  same WebSocket hub the existing jobs already use.
- Persist a personal toolset: one folder per tool, listed with name, description,
  and created/modified dates.
- Keep a single last-run record per tool (inputs + status + log); re-running
  overwrites it. No queue, no run history.
- Activate/deactivate **execution** via the local TOML.

## Non-Goals

- No queue, no multi-run history, no TOML job enqueue. (This is what keeps it out
  of the Studio/prep section.)
- No auto-parsing of the function signature (inputs are declared manually).
- No path/file-picker control type in v1 (a plain `text`/string field is used to
  paste paths, matching existing prep usage).
- No sandboxing beyond uv's environment isolation — scripts run with the user's
  privileges. The TOML toggle is the safety gate (off by default).

## Module Toggle (local TOML)

`rengu.local.toml`:

```toml
[toolbox]
enabled = false   # default OFF; gates EXECUTION only
```

Parsed in `rengu_flow/config/local_config.py` (new `ToolboxConfig` dataclass on
`LocalConfig`), exposed via a `toolbox_enabled()` helper. Behavior:

- The Toolbox nav item and section are **always visible**.
- Creating, editing, saving, and deleting tools **always works**, regardless of
  the flag.
- Only `POST /api/v1/toolbox/tools/{id}/run` validates the flag. When off it
  returns HTTP 409 with a clear message, and the UI disables the **Run** button
  and shows a banner: *"Execution disabled in rengu.local.toml → [toolbox].enabled"*.

## Storage Layout

One folder per tool under the managed data dir (`RENGU_FLOW_UI_DATA` or
`<repo>/data`), per the "no hidden folders, use data/" convention:

```
data/toolbox/<slug-id>/
├── tool.py          # user's script (contains the entrypoint function)
├── tool.json        # metadata + input definitions
├── inputs.json      # inputs of the last run (overwritten each run)
├── last_run.json    # status, exit_code, started/finished timestamps
├── last_run.log     # combined stdout+stderr of the last run
└── _runner.py       # generated at run time (PEP 723 header + dispatch)
```

A single last-run record per tool; re-running overwrites `inputs.json` and
`last_run.*`. The last run survives leaving the section.

### `tool.json` schema

Schema is free to change (no external users yet).

```json
{
  "id": "sumar",
  "name": "Sumar dos números",
  "description": "Suma num1 + num2",
  "entrypoint": "run",
  "requirements": ["numpy>=2.0"],
  "inputs": [
    { "param": "num1", "label": "Number 1", "control": "number",
      "default": 0, "min": null, "max": null, "step": null, "hint": "" },
    { "param": "num2", "label": "Number 2", "control": "number", "default": 0 }
  ],
  "created_at": "2026-06-17T00:00:00Z",
  "updated_at": "2026-06-17T00:00:00Z"
}
```

- `entrypoint` defaults to `run`; the user names a function with that name in
  `tool.py`.
- `requirements` becomes the PEP 723 `dependencies` list.
- Each `input.param` maps 1:1 to a keyword argument of the entrypoint.
- `created_at` is set on creation; `updated_at` on each save.

### Input control types (v1)

| control    | maps to     | extra fields            |
|------------|-------------|-------------------------|
| `number`   | int / float | `min`, `max`, `step`    |
| `text`     | str         | —                       |
| `textarea` | str         | —                       |
| `switch`   | bool        | —                       |
| `select`   | str         | `options` (list)        |

Type casting when building kwargs: `number` → int/float, `switch` → bool, the
rest → str. Field convention follows the repo standard: label + name + placeholder
+ hint.

## Runner (uv, isolated)

At run time the backend writes `inputs.json` and generates `_runner.py` in the
tool's folder:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]     # from tool.json.requirements
# ///
import json
import importlib.util
from pathlib import Path

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("user_tool", here / "tool.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

kwargs = json.loads((here / "inputs.json").read_text())
result = getattr(mod, "run")(**kwargs)   # entrypoint name from tool.json
if result is not None:
    print(result)
```

Launch: `uv run --no-project --isolated _runner.py`, started through the existing
`subprocess_util.popen_repo_subprocess` helper, with stdout/stderr redirected to
`last_run.log`.

- `--no-project --isolated` makes uv resolve the inline `dependencies` in an
  ephemeral, cached environment without touching rengu's `.venv` or pyproject.
- The user's `tool.py` stays clean (no PEP 723 header, no argv boilerplate); the
  runner injects requirements and dispatches kwargs.
- If `uv` is missing from PATH, the backend raises a clear error (same pattern as
  `tensorboard_server.build_tensorboard_cmd`).
- One active run per tool; starting a run while one is active is rejected (HTTP
  409).

## Backend

New module `rengu_flow_ui/toolbox.py`: CRUD over `data/toolbox/<id>/`
(create/list/read/update/delete, slug-id resolution from the name, read/write
`tool.json` and `last_run.json`), plus the runner generation, input→kwargs
casting, and run launch/status tracking (in-process registry of the active
`Popen` per tool id — no DB, no queue).

New routes module `rengu_flow_ui/toolbox_routes.py`, registered in `app.py` the
same way `prep_routes.register_prep_routes(app)` is:

| Method | Route                                     | Action                                              |
|--------|-------------------------------------------|-----------------------------------------------------|
| GET    | `/api/v1/toolbox/enabled`                 | `{enabled}` from the local TOML (drives the banner) |
| GET    | `/api/v1/toolbox/tools`                   | list: id, name, description, created/updated, last_run.status |
| POST   | `/api/v1/toolbox/tools`                   | create                                              |
| GET    | `/api/v1/toolbox/tools/{id}`              | full `tool.json` + last_run                         |
| PUT    | `/api/v1/toolbox/tools/{id}`              | update (touches `updated_at`)                       |
| DELETE | `/api/v1/toolbox/tools/{id}`              | delete folder                                       |
| POST   | `/api/v1/toolbox/tools/{id}/run`          | validate flag; cast inputs→kwargs, write `inputs.json`, launch runner (overwrites last_run) |
| GET    | `/api/v1/toolbox/tools/{id}/run`          | last_run status + inputs                            |
| GET    | `/api/v1/toolbox/tools/{id}/log`          | log snapshot (offset-based) + status                |
| POST   | `/api/v1/toolbox/tools/{id}/run/cancel`   | kill the running process                            |

Live log technique mirrors the existing jobs: the client fetches the current log
snapshot via REST (`/log`), then subscribes to incremental updates over a
WebSocket. The backend reuses the same tail mechanism behind a Toolbox-specific WS
route `/api/v1/toolbox/tools/{id}/log/ws`, so the frontend can reuse the
`useJobLogStream` composable pattern.

## Frontend (Vue 3 + Element Plus)

New top-level nav item **Toolbox** (always visible), separate from Studio.

- `ToolboxView.vue` — route `/toolbox`. Card per tool: name, description,
  created/modified dates, last-run status badge (idle/running/done/failed).
  Actions: New, Run, Edit, Delete.
- `ToolboxToolFormView.vue` — route `/toolbox/new` and `/toolbox/:id/edit`. Tool
  editor: name, description, entrypoint (prefilled `run`), requirements (one per
  line → `dependencies`), Python script editor, and the **inputs builder**:
  add/remove/reorder inputs with `param`, label, control type
  (`number`/`text`/`textarea`/`switch`/`select`), default, options (select),
  min/max/step (number), hint.
- `ToolboxRunPanel.vue` — renders the inputs as a form, a **Run** button (disabled
  + banner when execution is off), the live log panel (REST snapshot + WS reusing
  the `useJobLogStream` pattern), and the last run's inputs/status when
  re-entering.

Plumbing: API client additions in `api.ts`, routes in `router.ts`, and a nav item
in `App.vue` (both the main menu and the mobile drawer). All UI strings in English.

## Error Handling

- Execution disabled → `/run` returns HTTP 409 with a clear message; UI shows the
  banner and disables Run.
- `uv` missing → clear error with install guidance.
- Script error → status `failed`, exit code and traceback captured in
  `last_run.log`.
- Invalid inputs (required empty, unparseable number) → validated before launch.
- No silent fallback: uv resolution failures surface in the log; they are not
  swallowed (per the "no silent fallback on expected errors" rule).

## Testing (automated only)

- `toolbox.py`: create/list/update/delete, slug-id generation, created/updated
  dates, last-run overwrite, input→kwargs casting per control type, runner PEP 723
  header built from `requirements`.
- Routes: CRUD via FastAPI TestClient (reusing the `ui_client` / `ui_data_tmp`
  fixtures); `/run` rejects with 409 when the toggle is off; CRUD still works when
  off.
- Toggle: `toolbox_enabled()` reads `[toolbox].enabled` from the local TOML.
- Smoke e2e: a `sumar(num1, num2)` tool with **no requirements** run via
  `uv run --no-project --isolated`, asserting the result in the log (fast, no
  network). Guarded with a Monitor + timeout, since uv runs can take time.
