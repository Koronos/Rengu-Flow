# Studio Custom Tools — Design

**Date:** 2026-06-17
**Status:** Approved (pre-implementation), iterating from here.

## Summary

A framework in the Studio section that lets users author small Python "tools":
paste a Python script that exposes an entrypoint function, declare its input
controls in the UI, and run it via `uv` with PEP 723 inline requirements — fully
isolated from rengu's own virtualenv. Tools form a personal, always-at-hand
toolset shown with name, description, and creation/modification dates, intended
for custom dataset treatment or other ad-hoc processing.

Execution is gated by a flag in the local TOML. The section, authoring, and
saving remain available even when execution is disabled; only running is blocked.

## Goals

- Let a user paste a Python script with a named entrypoint (default `run`) and map
  its parameters to UI input controls (manual binding, declared by the user).
- Run the script with `uv run --no-project --isolated` so a `requirements` field
  becomes PEP 723 inline dependencies that uv resolves and caches in an ephemeral
  environment — never touching rengu's venv.
- Surface a live log (REST snapshot + WebSocket incremental updates) using the
  same mechanism the prep jobs already use.
- Persist a personal toolset: one folder per tool, listed with name, description,
  and created/modified dates.
- Keep a single last-run record per tool (inputs + status + log); re-running
  overwrites it. No queue, no run history.
- Activate/deactivate **execution** via the local TOML.

## Non-Goals

- No queue, no multi-run history, no TOML job enqueue.
- No auto-parsing of the function signature (inputs are declared manually).
- No path/file-picker control type in v1 (a plain `text`/string field is used to
  paste paths, matching existing prep usage).
- No sandboxing beyond uv's environment isolation — scripts run with the user's
  privileges. The TOML toggle is the safety gate (off by default).

## Module Toggle (local TOML)

`rengu.local.toml`:

```toml
[studio_tools]
enabled = false   # default OFF; gates EXECUTION only
```

Backend exposes `studio_tools_enabled()` reading the local TOML. Behavior:

- The Studio Tools nav item and section are **always visible** (unlike the
  maintenance module, which hides itself).
- Creating, editing, saving, and deleting tools **always works**, regardless of
  the flag.
- Only `POST /api/v1/studio/tools/{id}/run` validates the flag. When off it
  returns an error with a clear message, and the UI disables the **Run** button
  and shows a banner: *"Execution disabled in rengu.local.toml → [studio_tools].enabled"*.

## Storage Layout

One folder per tool under the managed data dir (`RENGU_FLOW_UI_DATA` or
`<repo>/data`), per the "no hidden folders, use data/" convention:

```
data/studio_tools/<slug-id>/
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
`subprocess_util.popen_*` helper, with stdout/stderr redirected to `last_run.log`.

- `--no-project --isolated` makes uv resolve the inline `dependencies` in an
  ephemeral, cached environment without touching rengu's `.venv` or pyproject.
- The user's `tool.py` stays clean (no PEP 723 header, no argv boilerplate); the
  runner injects requirements and dispatches kwargs.
- If `uv` is missing from PATH, the backend raises a clear error (same pattern as
  `tensorboard_server.build_tensorboard_cmd`).
- One active run per tool; starting a run while one is active is rejected (or
  cancel-and-replace — finalized in the implementation plan).

## Backend

New module `rengu_flow_ui/studio_tools.py`: CRUD over `data/studio_tools/<id>/`
(create/list/read/update/delete, slug-id resolution from the name, read/write
`tool.json` and `last_run.json`), plus the runner launch and input→kwargs casting.

New routes module `rengu_flow_ui/studio_routes.py`, registered in `app.py` the
same way `prep_routes.register_prep_routes(app)` is:

| Method | Route                                   | Action                                              |
|--------|-----------------------------------------|-----------------------------------------------------|
| GET    | `/api/v1/studio/enabled`                | `{enabled}` from the local TOML (drives the banner) |
| GET    | `/api/v1/studio/tools`                  | list: id, name, description, created/updated, last_run.status |
| POST   | `/api/v1/studio/tools`                  | create                                              |
| GET    | `/api/v1/studio/tools/{id}`             | full `tool.json` + last_run                         |
| PUT    | `/api/v1/studio/tools/{id}`             | update (touches `updated_at`)                       |
| DELETE | `/api/v1/studio/tools/{id}`             | delete folder                                       |
| POST   | `/api/v1/studio/tools/{id}/run`         | validate flag; cast inputs→kwargs, write `inputs.json`, launch runner (overwrites last_run) |
| GET    | `/api/v1/studio/tools/{id}/run/log`     | log snapshot + status                               |
| POST   | `/api/v1/studio/tools/{id}/run/cancel`  | kill the running process                            |

Live log technique (mirrors prep): the client fetches the current log snapshot
via REST (`/run/log`), then subscribes to incremental updates over the existing
WebSocket hub used by prep jobs until `done`/`failed`. The exact WS channel is
confirmed in the implementation plan so the existing `PrepJobLivePanel` / WS
client is reused rather than duplicated.

## Frontend (Vue 3 + Element Plus)

- `StudioToolsView.vue` — route `/studio/tools`, nav item always visible. Card per
  tool: name, description, created/modified dates, last-run status badge
  (idle/running/done/failed). Actions: New, Run, Edit, Delete.
- `StudioToolFormView.vue` — tool editor: name, description, entrypoint (prefilled
  `run`), requirements (one per line → `dependencies`), Python script editor, and
  the **inputs builder**: add/remove/reorder inputs with `param`, label, control
  type (`number`/`text`/`textarea`/`switch`/`select`), default, options (select),
  min/max/step (number), hint.
- `StudioToolRunPanel` — renders the inputs as a form (reusing `ConfigFormField`),
  a **Run** button (disabled + banner when execution is off), the live log panel
  (REST snapshot + WS), and the last run's inputs/status when re-entering.

Plumbing: API client additions in `api.ts`, a route in `router.ts`, and a nav item
in `App.vue`. All UI strings in English.

## Error Handling

- Execution disabled → `/run` returns an error with a clear message; UI shows the
  banner and disables Run.
- `uv` missing → clear error with install guidance.
- Script error → status `failed`, exit code and traceback captured in
  `last_run.log`.
- Invalid inputs (required empty, unparseable number) → validated before launch.
- No silent fallback: uv resolution failures surface in the log; they are not
  swallowed (per the "no silent fallback on expected errors" rule).

## Testing (automated only)

- `studio_tools.py`: create/list/update/delete, slug-id generation, created/updated
  dates, last-run overwrite.
- Runner: correct PEP 723 header built from `requirements`; input→kwargs casting per
  control type.
- Toggle: `studio_tools_enabled()` reads the local TOML; `/run` rejects when off;
  CRUD still works when off.
- Smoke e2e: a `sumar(num1, num2)` tool with **no requirements** run via
  `uv run --no-project --isolated`, asserting the result in the log (fast, no
  network). Guarded with a Monitor + timeout, since uv runs can take time.
