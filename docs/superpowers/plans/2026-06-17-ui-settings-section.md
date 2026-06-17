# UI Settings Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Configuration section to the UI that edits `rengu.local.toml` directly (training, maintenance, and restart-only `public`/`token`), and relocate the existing theme control into it.

**Architecture:** A new backend module `settings_store.py` reads/writes `rengu.local.toml` with `tomlkit` (preserving comments), exposed via `GET`/`PUT {API_PREFIX}/settings`. A new `SettingsView.vue` (route `/settings`) renders the editable form plus the moved `<ThemeToggle/>`. The theme stays a per-browser `localStorage` preference (`useTheme.ts` unchanged) — only the widget moves.

**Tech Stack:** Python 3 / FastAPI / `tomlkit` (backend); Vue 3 + TypeScript + Element Plus + Vitest (frontend); pytest (backend tests).

## Global Constraints

- **English only** in code, comments, UI strings, and docs (Spanish only in conversation).
- **No new dependency:** `tomlkit` is already in `pyproject.toml`; do not add libraries.
- **No version bump** and **no changes to `rengu.local.toml.example` or `local_config.py`** — the theme is NOT persisted to TOML; `UiConfig` is unchanged.
- **No silent fallback:** invalid input must raise/return a typed error, never be swallowed.
- **Form-field convention** (per `docs/developer/documentation-conventions.md`): every editable form field has a label + the TOML key shown beneath + a `placeholder` describing the empty-state behavior + a one-line hint.
- **Backend test command:** `uv run --extra dev --extra ui pytest <path> -v` (run from repo root).
- **Frontend test command:** `cd ui/web && npm run test` (Vitest), and `npm run typecheck` for types.
- **Commits:** one per task, author is `koronos` (no email). End commit messages with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer. No secrets/personal paths in commits.

---

### Task 1: Backend `settings_store.py` — read/write `rengu.local.toml`

**Files:**
- Create: `rengu_flow_ui/settings_store.py`
- Test: `tests/test_settings_store.py`

**Interfaces:**
- Consumes: `rengu_flow.config.local_config.local_config_path`, `parse_local_config_dict`, `default_local_config`, `repo_root`.
- Produces:
  - `config_path() -> Path` — wraps `local_config_path()` (tests monkeypatch this).
  - `class SettingsError(ValueError)`.
  - `read_settings(path: Path | None = None) -> dict` — shape:
    ```python
    {
      "path": str, "exists": bool,
      "editable": {
        "training": {"num_gpus": int, "master_port": int, "extra_args": str, "env": dict[str, str]},
        "maintenance": {"enabled": bool, "allow_pip": bool},
      },
      "restartRequired": {"ui": {"public": bool, "token": str | None}},
      "readOnly": {"ui": {"host": str, "port": int, "data_dir": str}},
    }
    ```
  - `write_settings(patch: dict, path: Path | None = None) -> dict` — validates, merges editable+restart fields into the existing tomlkit document (non-destructive), writes atomically, returns `read_settings(path)`.
  - `apply_maintenance_env(settings: dict) -> None` — pushes `maintenance.enabled`/`allow_pip` from a `read_settings()` result into `os.environ` (override) so they take effect live.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_settings_store.py`:

```python
"""rengu.local.toml read/write via tomlkit (settings_store)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rengu_flow_ui import settings_store
from rengu_flow_ui.settings_store import SettingsError

EXAMPLE = """\
# top comment
[ui]
host = "127.0.0.1"
port = 8765
public = false
data_dir = "data"
# token comment
# token = "change-me"

[maintenance]
enabled = false
allow_pip = false

[training]
num_gpus = 1
master_port = 29500
extra_args = ""

[training.env]
# keep me
NCCL_P2P_DISABLE = "1"
"""


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "rengu.local.toml"
    p.write_text(EXAMPLE, encoding="utf-8")
    return p


def test_read_settings_groups_fields(cfg_file: Path) -> None:
    s = settings_store.read_settings(cfg_file)
    assert s["exists"] is True
    assert s["editable"]["training"]["num_gpus"] == 1
    assert s["editable"]["training"]["env"] == {"NCCL_P2P_DISABLE": "1"}
    assert s["editable"]["maintenance"]["enabled"] is False
    assert s["restartRequired"]["ui"]["public"] is False
    assert s["restartRequired"]["ui"]["token"] is None
    assert s["readOnly"]["ui"] == {"host": "127.0.0.1", "port": 8765, "data_dir": "data"}


def test_read_settings_missing_file_uses_defaults(tmp_path: Path) -> None:
    s = settings_store.read_settings(tmp_path / "absent.toml")
    assert s["exists"] is False
    assert s["editable"]["training"]["num_gpus"] == 1
    assert s["readOnly"]["ui"]["port"] == 8765


def test_write_preserves_comments_and_untouched_keys(cfg_file: Path) -> None:
    settings_store.write_settings({"training": {"num_gpus": 2}}, cfg_file)
    text = cfg_file.read_text(encoding="utf-8")
    assert "# top comment" in text
    assert "# keep me" in text
    assert "num_gpus = 2" in text
    # untouched key kept
    assert 'master_port = 29500' in text


def test_write_replaces_env_table(cfg_file: Path) -> None:
    out = settings_store.write_settings(
        {"training": {"env": {"FOO": "bar"}}}, cfg_file
    )
    assert out["editable"]["training"]["env"] == {"FOO": "bar"}
    assert "NCCL_P2P_DISABLE" not in cfg_file.read_text(encoding="utf-8")


def test_write_token_empty_string_clears_key(cfg_file: Path) -> None:
    out = settings_store.write_settings({"ui": {"token": ""}}, cfg_file)
    assert out["restartRequired"]["ui"]["token"] is None


def test_write_rejects_bad_num_gpus(cfg_file: Path) -> None:
    with pytest.raises(SettingsError):
        settings_store.write_settings({"training": {"num_gpus": 0}}, cfg_file)


def test_write_rejects_bad_port(cfg_file: Path) -> None:
    with pytest.raises(SettingsError):
        settings_store.write_settings({"training": {"master_port": 70000}}, cfg_file)


def test_write_rejects_non_editable_key(cfg_file: Path) -> None:
    with pytest.raises(SettingsError):
        settings_store.write_settings({"ui": {"host": "0.0.0.0"}}, cfg_file)


def test_write_creates_file_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "new.toml"
    out = settings_store.write_settings({"maintenance": {"enabled": True}}, target)
    assert target.is_file()
    assert out["editable"]["maintenance"]["enabled"] is True


def test_apply_maintenance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RENGUFLOW_MAINTENANCE", raising=False)
    settings = {"editable": {"maintenance": {"enabled": True, "allow_pip": False}}}
    settings_store.apply_maintenance_env(settings)
    import os

    assert os.environ["RENGUFLOW_MAINTENANCE"] == "1"
    assert os.environ["RENGUFLOW_MAINTENANCE_ALLOW_PIP"] == "0"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev --extra ui pytest tests/test_settings_store.py -v`
Expected: FAIL — `ModuleNotFoundError: rengu_flow_ui.settings_store` (module not created yet).

- [ ] **Step 3: Implement `settings_store.py`**

Create `rengu_flow_ui/settings_store.py`:

```python
"""Read and write ``rengu.local.toml`` for the UI Settings section.

Uses tomlkit so writes preserve the file's comments and formatting. Only the editable and
restart-required fields are ever written; everything else in the document is left untouched.
Binding fields (host/port/data_dir) are surfaced read-only — they only take effect at startup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomlkit

from rengu_flow.config.local_config import (
    default_local_config,
    local_config_path,
    parse_local_config_dict,
    repo_root,
)

# Editable field whitelists, by TOML section. A patch may only set these keys.
_EDITABLE_TRAINING_SCALARS = ("num_gpus", "master_port", "extra_args")
_EDITABLE_MAINTENANCE = ("enabled", "allow_pip")
_RESTART_UI = ("public", "token")


class SettingsError(ValueError):
    """Invalid settings patch (bad value or non-editable key)."""


def config_path() -> Path:
    """Path to ``rengu.local.toml``. Indirection point so tests can target a temp file."""
    return local_config_path()


def read_settings(path: Path | None = None) -> dict[str, Any]:
    p = path or config_path()
    exists = p.is_file()
    if exists:
        cfg = parse_local_config_dict(tomlkit.parse(p.read_text(encoding="utf-8")), root=repo_root())
    else:
        cfg = default_local_config()
    return {
        "path": str(p),
        "exists": exists,
        "editable": {
            "training": {
                "num_gpus": cfg.training.num_gpus,
                "master_port": cfg.training.master_port,
                "extra_args": cfg.training.extra_args,
                "env": dict(cfg.training.env),
            },
            "maintenance": {
                "enabled": cfg.maintenance.enabled,
                "allow_pip": cfg.maintenance.allow_pip,
            },
        },
        "restartRequired": {"ui": {"public": cfg.ui.public, "token": cfg.ui.token}},
        "readOnly": {
            "ui": {"host": cfg.ui.host, "port": cfg.ui.port, "data_dir": cfg.ui.data_dir}
        },
    }


def _validate_patch(patch: dict[str, Any]) -> None:
    allowed_sections = {"training", "maintenance", "ui"}
    unknown = set(patch) - allowed_sections
    if unknown:
        raise SettingsError(f"Unknown settings section(s): {', '.join(sorted(unknown))}")

    training = patch.get("training", {})
    bad = set(training) - set(_EDITABLE_TRAINING_SCALARS) - {"env"}
    if bad:
        raise SettingsError(f"Non-editable training key(s): {', '.join(sorted(bad))}")
    if "num_gpus" in training and (not isinstance(training["num_gpus"], int) or training["num_gpus"] < 1):
        raise SettingsError("num_gpus must be an integer >= 1")
    for port_key in ("master_port",):
        if port_key in training:
            v = training[port_key]
            if not isinstance(v, int) or not (1 <= v <= 65535):
                raise SettingsError(f"{port_key} must be an integer in 1..65535")
    if "extra_args" in training and not isinstance(training["extra_args"], str):
        raise SettingsError("extra_args must be a string")
    if "env" in training:
        env = training["env"]
        if not isinstance(env, dict):
            raise SettingsError("training.env must be a table")
        for k in env:
            if not isinstance(k, str) or not k.strip():
                raise SettingsError("training.env keys must be non-empty strings")

    maintenance = patch.get("maintenance", {})
    bad_m = set(maintenance) - set(_EDITABLE_MAINTENANCE)
    if bad_m:
        raise SettingsError(f"Non-editable maintenance key(s): {', '.join(sorted(bad_m))}")
    for k in _EDITABLE_MAINTENANCE:
        if k in maintenance and not isinstance(maintenance[k], bool):
            raise SettingsError(f"maintenance.{k} must be a boolean")

    ui = patch.get("ui", {})
    bad_u = set(ui) - set(_RESTART_UI)
    if bad_u:
        raise SettingsError(f"Non-editable ui key(s): {', '.join(sorted(bad_u))}")
    if "public" in ui and not isinstance(ui["public"], bool):
        raise SettingsError("ui.public must be a boolean")
    if "token" in ui and ui["token"] is not None and not isinstance(ui["token"], str):
        raise SettingsError("ui.token must be a string or null")


def _table(doc: Any, name: str) -> Any:
    if name not in doc:
        doc[name] = tomlkit.table()
    return doc[name]


def write_settings(patch: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    _validate_patch(patch)
    p = path or config_path()
    if p.is_file():
        doc = tomlkit.parse(p.read_text(encoding="utf-8"))
    else:
        doc = tomlkit.document()

    if "training" in patch:
        t = _table(doc, "training")
        for key in _EDITABLE_TRAINING_SCALARS:
            if key in patch["training"]:
                t[key] = patch["training"][key]
        if "env" in patch["training"]:
            env_tbl = tomlkit.table()
            for k, v in patch["training"]["env"].items():
                env_tbl[k] = str(v)
            t["env"] = env_tbl

    if "maintenance" in patch:
        m = _table(doc, "maintenance")
        for key in _EDITABLE_MAINTENANCE:
            if key in patch["maintenance"]:
                m[key] = patch["maintenance"][key]

    if "ui" in patch:
        u = _table(doc, "ui")
        if "public" in patch["ui"]:
            u["public"] = patch["ui"]["public"]
        if "token" in patch["ui"]:
            tok = patch["ui"]["token"]
            if tok:  # non-empty string
                u["token"] = tok
            elif "token" in u:  # empty/None clears it
                del u["token"]

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
    os.replace(tmp, p)
    return read_settings(p)


def apply_maintenance_env(settings: dict[str, Any]) -> None:
    """Push maintenance flags from a read_settings() result into os.environ (override)."""
    m = settings["editable"]["maintenance"]
    os.environ["RENGUFLOW_MAINTENANCE"] = "1" if m["enabled"] else "0"
    os.environ["RENGUFLOW_MAINTENANCE_ALLOW_PIP"] = "1" if m["allow_pip"] else "0"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev --extra ui pytest tests/test_settings_store.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add rengu_flow_ui/settings_store.py tests/test_settings_store.py
git commit -m "feat(ui): settings_store read/write rengu.local.toml via tomlkit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Backend endpoints `GET`/`PUT {API_PREFIX}/settings`

**Files:**
- Modify: `rengu_flow_ui/app.py` (add body model near the other `BaseModel`s ~line 76-220; add routes near the maintenance block ~line 1286)
- Test: `tests/test_settings_api.py`

**Interfaces:**
- Consumes: `settings_store.read_settings`, `write_settings`, `apply_maintenance_env`, `SettingsError` (Task 1); `rengu_flow.config.local_config.load_local_config`.
- Produces:
  - `GET {API_PREFIX}/settings` → `read_settings()` dict.
  - `PUT {API_PREFIX}/settings` (body `SettingsUpdateBody`) → re-read settings dict; reloads cached config and re-applies maintenance env.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_settings_api.py`:

```python
"""Settings API: GET/PUT /api/v1/settings."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def cfg_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "rengu.local.toml"
    p.write_text(
        "[ui]\nhost = \"127.0.0.1\"\nport = 8765\npublic = false\ndata_dir = \"data\"\n\n"
        "[maintenance]\nenabled = false\nallow_pip = false\n\n"
        "[training]\nnum_gpus = 1\nmaster_port = 29500\nextra_args = \"\"\n\n"
        "[training.env]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("rengu_flow_ui.settings_store.config_path", lambda: p)
    return p


def test_get_settings_shape(ui_client, cfg_file: Path) -> None:
    r = ui_client.get("/api/v1/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["editable"]["training"]["num_gpus"] == 1
    assert body["readOnly"]["ui"]["port"] == 8765
    assert body["restartRequired"]["ui"]["public"] is False


def test_put_settings_writes_training(ui_client, cfg_file: Path) -> None:
    r = ui_client.put("/api/v1/settings", json={"training": {"num_gpus": 2, "extra_args": "--x"}})
    assert r.status_code == 200
    assert r.json()["editable"]["training"]["num_gpus"] == 2
    assert "num_gpus = 2" in cfg_file.read_text(encoding="utf-8")


def test_put_settings_maintenance_applies_to_env(ui_client, cfg_file: Path) -> None:
    import os

    os.environ.pop("RENGUFLOW_MAINTENANCE", None)
    r = ui_client.put("/api/v1/settings", json={"maintenance": {"enabled": True}})
    assert r.status_code == 200
    # maintenance now reads as enabled live
    assert ui_client.get("/api/v1/maintenance/enabled").json()["enabled"] is True


def test_put_settings_invalid_returns_422(ui_client, cfg_file: Path) -> None:
    r = ui_client.put("/api/v1/settings", json={"training": {"num_gpus": 0}})
    assert r.status_code == 422


def test_put_settings_rejects_non_editable(ui_client, cfg_file: Path) -> None:
    r = ui_client.put("/api/v1/settings", json={"ui": {"host": "0.0.0.0"}})
    assert r.status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev --extra ui pytest tests/test_settings_api.py -v`
Expected: FAIL — 404 on `/api/v1/settings` (routes not registered).

- [ ] **Step 3: Add the body model**

In `rengu_flow_ui/app.py`, add near the other body models (after `class DepsInstallBody` ~line 124):

```python
class SettingsUpdateBody(BaseModel):
    training: dict[str, Any] | None = None
    maintenance: dict[str, Any] | None = None
    ui: dict[str, Any] | None = None
```

- [ ] **Step 4: Add the routes**

In `rengu_flow_ui/app.py`, add immediately after the `maintenance_deps_install` route (~line 1330):

```python
    @app.get(f"{API_PREFIX}/settings")
    def get_settings() -> dict[str, Any]:
        from rengu_flow_ui import settings_store

        return settings_store.read_settings()

    @app.put(f"{API_PREFIX}/settings")
    def put_settings(body: SettingsUpdateBody) -> dict[str, Any]:
        from rengu_flow.config.local_config import load_local_config
        from rengu_flow_ui import settings_store

        patch = body.model_dump(exclude_none=True)
        try:
            result = settings_store.write_settings(patch)
        except settings_store.SettingsError as e:
            raise HTTPException(422, str(e))
        # Refresh the cached LocalConfig and re-apply maintenance flags so they take effect
        # without a server restart (training.* is read fresh per job subprocess already).
        load_local_config()
        settings_store.apply_maintenance_env(result)
        return result
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev --extra ui pytest tests/test_settings_api.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add rengu_flow_ui/app.py tests/test_settings_api.py
git commit -m "feat(ui): GET/PUT /api/v1/settings endpoints for rengu.local.toml

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Frontend API client + types

**Files:**
- Modify: `ui/web/src/types/api.ts` (add interfaces near `MaintenanceEnabledResult` ~line 668)
- Modify: `ui/web/src/api.ts` (add methods near `maintenanceEnabled` ~line 491)
- Test: `ui/web/src/api.settings.test.ts`

**Interfaces:**
- Produces:
  - Type `LocalSettings` (mirrors Task 1 `read_settings` shape).
  - Type `LocalSettingsPatch`.
  - `api.getSettings(): Promise<LocalSettings>`
  - `api.updateSettings(patch: LocalSettingsPatch): Promise<LocalSettings>`

- [ ] **Step 1: Write the failing test**

Create `ui/web/src/api.settings.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

afterEach(() => {
  vi.restoreAllMocks();
});

const SAMPLE = {
  path: "/repo/rengu.local.toml",
  exists: true,
  editable: {
    training: { num_gpus: 1, master_port: 29500, extra_args: "", env: {} },
    maintenance: { enabled: false, allow_pip: false },
  },
  restartRequired: { ui: { public: false, token: null } },
  readOnly: { ui: { host: "127.0.0.1", port: 8765, data_dir: "data" } },
};

describe("settings api", () => {
  it("getSettings fetches /settings", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify(SAMPLE), { status: 200 }));
    const out = await api.getSettings();
    expect(out.editable.training.num_gpus).toBe(1);
    expect(fetchMock).toHaveBeenCalledWith("/api/v1/settings", expect.anything());
  });

  it("updateSettings PUTs the patch", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify(SAMPLE), { status: 200 }));
    await api.updateSettings({ training: { num_gpus: 2 } });
    const [, opts] = fetchMock.mock.calls[0];
    expect(opts?.method).toBe("PUT");
    expect(JSON.parse(opts?.body as string)).toEqual({ training: { num_gpus: 2 } });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui/web && npm run test -- api.settings`
Expected: FAIL — `api.getSettings is not a function`.

- [ ] **Step 3: Add the types**

In `ui/web/src/types/api.ts`, add (near `MaintenanceEnabledResult` ~line 668):

```ts
export interface LocalSettings {
  path: string;
  exists: boolean;
  editable: {
    training: {
      num_gpus: number;
      master_port: number;
      extra_args: string;
      env: Record<string, string>;
    };
    maintenance: { enabled: boolean; allow_pip: boolean };
  };
  restartRequired: { ui: { public: boolean; token: string | null } };
  readOnly: { ui: { host: string; port: number; data_dir: string } };
}

export interface LocalSettingsPatch {
  training?: Partial<{
    num_gpus: number;
    master_port: number;
    extra_args: string;
    env: Record<string, string>;
  }>;
  maintenance?: Partial<{ enabled: boolean; allow_pip: boolean }>;
  ui?: Partial<{ public: boolean; token: string | null }>;
}
```

- [ ] **Step 4: Add the API methods**

In `ui/web/src/api.ts`, first extend the type import block at the top (the `} from "./types/api";` import) to include `LocalSettings` and `LocalSettingsPatch`. Then add, right after the `maintenanceEnabled` method (~line 491):

```ts
  /** Read rengu.local.toml settings (editable + restart-required + read-only groups). */
  getSettings: () => request<LocalSettings>("/settings"),

  /** Write the editable subset of rengu.local.toml; returns the re-read settings. */
  updateSettings: (patch: LocalSettingsPatch) =>
    request<LocalSettings>("/settings", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),
```

- [ ] **Step 5: Run the test + typecheck to verify they pass**

Run: `cd ui/web && npm run test -- api.settings && npm run typecheck`
Expected: PASS, no type errors.

- [ ] **Step 6: Commit**

```bash
git add ui/web/src/types/api.ts ui/web/src/api.ts ui/web/src/api.settings.test.ts
git commit -m "feat(ui): settings API client (getSettings/updateSettings)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Frontend `SettingsView.vue` + route + nav, relocate `<ThemeToggle/>`

**Files:**
- Create: `ui/web/src/views/SettingsView.vue`
- Modify: `ui/web/src/router.ts` (add `/settings` route)
- Modify: `ui/web/src/App.vue` (add nav entry in both nav blocks; add to `activeMenu` + `pageTitle`; remove `<ThemeToggle/>` from both blocks and its import)
- Test: `ui/web/src/views/SettingsView.test.ts`

**Interfaces:**
- Consumes: `api.getSettings`, `api.updateSettings`, `LocalSettings`, `LocalSettingsPatch` (Task 3); `ThemeToggle.vue`; `KeyValueListField.vue` (props: `model-value` is `Record<string,string>`, emits `update:model-value`); Element Plus `Setting` icon from `@element-plus/icons-vue`.
- Produces: route `name: "settings"`, path `/settings`.

- [ ] **Step 1: Write the failing smoke test**

Create `ui/web/src/views/SettingsView.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import ElementPlus from "element-plus";

vi.mock("../api", () => ({
  api: {
    getSettings: vi.fn().mockResolvedValue({
      path: "/repo/rengu.local.toml",
      exists: true,
      editable: {
        training: { num_gpus: 1, master_port: 29500, extra_args: "", env: {} },
        maintenance: { enabled: false, allow_pip: false },
      },
      restartRequired: { ui: { public: false, token: null } },
      readOnly: { ui: { host: "127.0.0.1", port: 8765, data_dir: "data" } },
    }),
    updateSettings: vi.fn(),
  },
}));

import SettingsView from "./SettingsView.vue";

describe("SettingsView", () => {
  it("loads settings and renders the theme toggle", async () => {
    const wrapper = mount(SettingsView, { global: { plugins: [ElementPlus] } });
    await new Promise((r) => setTimeout(r, 0));
    await wrapper.vm.$nextTick();
    expect(wrapper.findComponent({ name: "ThemeToggle" }).exists()).toBe(true);
    expect(wrapper.text()).toContain("127.0.0.1");
  });
});
```

> Note: if `ThemeToggle` has no explicit `name`, find it by its root class instead — `expect(wrapper.find(".theme-toggle").exists()).toBe(true)`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui/web && npm run test -- SettingsView`
Expected: FAIL — cannot resolve `./SettingsView.vue`.

- [ ] **Step 3: Create `SettingsView.vue`**

Create `ui/web/src/views/SettingsView.vue`. Follow the form-field convention (label + TOML key beneath + placeholder + hint). Use read-only display for the Server card binding fields and the restart badge for `public`/`token`.

```vue
<template>
  <div class="settings-view page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">
          Edit <code>rengu.local.toml</code> ({{ form?.path }}). Training changes apply to the next
          run; maintenance applies immediately; server fields need a restart.
        </p>
      </div>
    </div>

    <el-alert v-if="error" type="error" show-icon class="mb-12" :title="error" :closable="false" />

    <el-skeleton v-if="loading" :rows="8" animated />

    <template v-else-if="form">
      <el-card shadow="never" class="mb-12">
        <template #header>Appearance</template>
        <div class="field">
          <label class="field-label">Color theme</label>
          <div class="field-name">browser preference (not stored in TOML)</div>
          <ThemeToggle />
          <p class="field-hint">Per-browser; saved locally, applied before paint to avoid a flash.</p>
        </div>
      </el-card>

      <el-card shadow="never" class="mb-12">
        <template #header>Training</template>
        <el-form label-position="top">
          <el-form-item>
            <template #label>GPUs <code class="toml-key">training.num_gpus</code></template>
            <el-input-number v-model="form.editable.training.num_gpus" :min="1" />
            <p class="field-hint">DeepSpeed <code>--num_gpus</code> for <code>rengu train</code>. CLI flags override this.</p>
          </el-form-item>
          <el-form-item>
            <template #label>Master port <code class="toml-key">training.master_port</code></template>
            <el-input-number v-model="form.editable.training.master_port" :min="1" :max="65535" />
            <p class="field-hint">Rendezvous port for the local DeepSpeed launcher. Default 29500.</p>
          </el-form-item>
          <el-form-item>
            <template #label>Extra args <code class="toml-key">training.extra_args</code></template>
            <el-input v-model="form.editable.training.extra_args" placeholder="e.g. --validate-only" />
            <p class="field-hint">Space-separated args appended after <code>--config</code>.</p>
          </el-form-item>
        </el-form>
      </el-card>

      <el-card shadow="never" class="mb-12">
        <template #header>Training environment</template>
        <div class="field">
          <label class="field-label">Subprocess env vars <code class="toml-key">training.env</code></label>
          <KeyValueListField
            v-model="form.editable.training.env"
            hint="Literal os.environ keys for the training subprocess (values are strings). Empty = inherit only the parent env."
          />
        </div>
      </el-card>

      <el-card shadow="never" class="mb-12">
        <template #header>Maintenance</template>
        <el-form label-position="top">
          <el-form-item>
            <template #label>Enable maintenance tools <code class="toml-key">maintenance.enabled</code></template>
            <el-switch v-model="form.editable.maintenance.enabled" />
            <p class="field-hint">Shows the Maintenance page (destructive DB reset, submodule update). Applies immediately.</p>
          </el-form-item>
          <el-form-item>
            <template #label>Allow pip in maintenance <code class="toml-key">maintenance.allow_pip</code></template>
            <el-switch v-model="form.editable.maintenance.allow_pip" />
            <p class="field-hint">Lets the deps-install action run pip. Off by default.</p>
          </el-form-item>
        </el-form>
      </el-card>

      <el-card shadow="never" class="mb-12">
        <template #header>Server</template>
        <el-form label-position="top">
          <el-form-item>
            <template #label>
              Expose on local network <code class="toml-key">ui.public</code>
              <el-tag size="small" type="warning" class="ml-6">restart to apply</el-tag>
            </template>
            <el-switch v-model="form.restartRequired.ui.public" />
            <p class="field-hint">Binds 0.0.0.0 so other devices can reach the UI. Set a token when on.</p>
          </el-form-item>
          <el-form-item>
            <template #label>
              API token <code class="toml-key">ui.token</code>
              <el-tag size="small" type="warning" class="ml-6">restart to apply</el-tag>
            </template>
            <el-input
              v-model="tokenField"
              type="password"
              show-password
              placeholder="empty = no token required"
            />
            <p class="field-hint">Required on every request (header X-Rengu-Flow-Token). Strongly recommended when public.</p>
          </el-form-item>
          <el-descriptions :column="1" size="small" border class="mt-12">
            <el-descriptions-item label="host (ui.host)">{{ form.readOnly.ui.host }}</el-descriptions-item>
            <el-descriptions-item label="port (ui.port)">{{ form.readOnly.ui.port }}</el-descriptions-item>
            <el-descriptions-item label="data_dir (ui.data_dir)">{{ form.readOnly.ui.data_dir }}</el-descriptions-item>
          </el-descriptions>
          <p class="field-hint">Read-only here — they only take effect at startup. Edit the file directly to change them.</p>
        </el-form>
      </el-card>

      <div class="actions">
        <el-button type="primary" :loading="saving" @click="onSave">Save changes</el-button>
        <el-text v-if="savedAt" type="success" size="small" class="ml-12">Saved.</el-text>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import type { LocalSettings, LocalSettingsPatch } from "../types/api";
import ThemeToggle from "../components/ThemeToggle.vue";
import KeyValueListField from "../components/KeyValueListField.vue";

const form = ref<LocalSettings | null>(null);
const loading = ref(true);
const saving = ref(false);
const error = ref("");
const savedAt = ref(false);

const tokenField = computed<string>({
  get: () => form.value?.restartRequired.ui.token ?? "",
  set: (v: string) => {
    if (form.value) form.value.restartRequired.ui.token = v ? v : null;
  },
});

async function load(): Promise<void> {
  loading.value = true;
  error.value = "";
  try {
    form.value = await api.getSettings();
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to load settings";
  } finally {
    loading.value = false;
  }
}

async function onSave(): Promise<void> {
  if (!form.value) return;
  saving.value = true;
  savedAt.value = false;
  error.value = "";
  const patch: LocalSettingsPatch = {
    training: { ...form.value.editable.training },
    maintenance: { ...form.value.editable.maintenance },
    ui: { ...form.value.restartRequired.ui },
  };
  try {
    form.value = await api.updateSettings(patch);
    savedAt.value = true;
    ElMessage.success("Settings saved");
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to save settings";
  } finally {
    saving.value = false;
  }
}

onMounted(load);
</script>

<style scoped>
.field { margin-bottom: 8px; }
.field-label { font-weight: 600; display: block; margin-bottom: 2px; }
.field-name, .toml-key { font-size: 12px; color: var(--el-text-color-secondary); }
.field-hint { margin: 4px 0 0; font-size: 12px; color: var(--el-text-color-secondary); }
.actions { margin-top: 16px; }
.ml-6 { margin-left: 6px; }
.ml-12 { margin-left: 12px; }
.mt-12 { margin-top: 12px; }
.mb-12 { margin-bottom: 12px; }
</style>
```

> The file ends with `</style>` — there is no trailing `</template>`.

- [ ] **Step 4: Add the route**

In `ui/web/src/router.ts`, add inside `routes` (after the `/maintenance` route, ~line 19):

```ts
    {
      path: "/settings",
      name: "settings",
      component: () => import("./views/SettingsView.vue"),
    },
```

- [ ] **Step 5: Wire nav in `App.vue` and remove the sidebar `<ThemeToggle/>`**

In `ui/web/src/App.vue`:

1. Add `Setting` to the icon import (line ~137):
```ts
import { Document, Files, MagicStick, Menu, Setting, Tools, TrendCharts, VideoPlay } from "@element-plus/icons-vue";
```

2. In **both** `app-menu--footer` menus (the `<el-aside>` block ~line 49 and the drawer block ~line 119), add a Settings item before the Docs item:
```html
            <el-menu-item index="/settings">
              <el-icon><Setting /></el-icon>
              <span>Configuration</span>
            </el-menu-item>
```

3. Remove `<ThemeToggle />` from **both** `app-menu-bottom` blocks (line ~44 and ~108).

4. Remove the now-unused import: delete `import ThemeToggle from "./components/ThemeToggle.vue";` (line ~144).

5. Add to `activeMenu` computed (after the `maintenance` line ~178):
```ts
  if (name === "settings") return "/settings";
```

6. Add to `pageTitle` map (in the `names` record ~line 188):
```ts
    settings: "Configuration",
```

- [ ] **Step 6: Run the test + typecheck to verify they pass**

Run: `cd ui/web && npm run test -- SettingsView && npm run typecheck`
Expected: PASS, no type errors (including: no unused `ThemeToggle` import remains in App.vue).

- [ ] **Step 7: Commit**

```bash
git add ui/web/src/views/SettingsView.vue ui/web/src/views/SettingsView.test.ts ui/web/src/router.ts ui/web/src/App.vue
git commit -m "feat(ui): Configuration view editing rengu.local.toml; move theme toggle into it

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: User documentation

**Files:**
- Modify: `docs/user/web-ui.md` (add a "Configuration" section)

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the docs section**

Append to `docs/user/web-ui.md` a section describing the new page. Match the file's existing heading style and tone. Content to cover:

```markdown
## Configuration

The **Configuration** page (sidebar gear icon) edits `rengu.local.toml` from the browser:

- **Appearance** — the color theme. This is a per-browser preference saved in your browser, not
  written to the config file.
- **Training** — `num_gpus`, `master_port`, `extra_args`. Applied to the next `rengu train` run
  (each run reloads the file); CLI flags still override these.
- **Training environment** — the `training.env` table of environment variables for the training
  subprocess.
- **Maintenance** — toggles the Maintenance tools. Applied immediately.
- **Server** — `ui.public` and `ui.token` are editable but **require a server restart** to take
  effect; `ui.host`, `ui.port`, and `ui.data_dir` are shown read-only (edit the file directly to
  change them, then restart).

Saving writes only these fields back to `rengu.local.toml`; comments and any other content in the
file are preserved.
```

- [ ] **Step 2: Verify the docs render in the UI Docs view (optional manual check)**

Run: `cd ui/web && npm run typecheck` (sanity; docs are markdown, no build impact).
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/user/web-ui.md
git commit -m "docs(ui): document the Configuration page

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** backend read/write with comment preservation (Task 1), GET/PUT + live maintenance re-apply + cached-config reload (Task 2), API client/types (Task 3), view + route + nav + theme relocation + read-only/restart fields + form-field hints (Task 4), user docs (Task 5). Theme intentionally NOT persisted to TOML and `local_config.py`/`.example` untouched — matches the approved spec.
- **Type consistency:** `LocalSettings`/`LocalSettingsPatch` shapes match `read_settings()` output and `write_settings()` patch keys; `apply_maintenance_env` reads `settings["editable"]["maintenance"]` exactly as returned by `read_settings`.
- **Field classification:** editable = `training.*`, `maintenance.*`; restart-required = `ui.public`, `ui.token`; read-only = `ui.host`, `ui.port`, `ui.data_dir` — consistent across store validation, view, and docs.
