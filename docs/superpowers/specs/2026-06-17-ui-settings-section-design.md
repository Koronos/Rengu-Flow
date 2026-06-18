# Settings section — edit `rengu.local.toml` from the UI

**Date:** 2026-06-17
**Status:** Approved (design)

## Goal

Add a **Configuration** section to the UI that edits `rengu.local.toml` directly, and relocate
the existing color-theme control into it. The theme stays exactly as it is today — a per-browser
`localStorage` preference; we only move the widget. The theme is **not** written to the TOML.

## Context

- Backend is `rengu_flow_ui` (FastAPI). `rengu.local.toml` is loaded once at startup by
  `load_local_config()` and pushed into `os.environ` via `apply_local_config_to_environ()`.
  Sections: `[ui]` (host, port, public, data_dir, token), `[maintenance]`, `[training]`,
  `[training.env]`.
- `tomlkit` is already a dependency (the UI parses configs with it) → we can rewrite the TOML
  while preserving comments and formatting. No new dependency.
- `training.*` applies live: each training job is launched in a fresh subprocess that reloads
  the TOML, so edits take effect on the next job without restarting the UI.
- `maintenance.*` and the binding fields (`host/port/public/data_dir/token`) are read by the
  long-lived UI process from `os.environ`, set at startup. `maintenance.*` can be re-applied to
  the environment after a write to take effect live; binding fields are intrinsically
  startup-only.
- Theme today: `useTheme.ts` + `localStorage` (key `rengu-flow-theme`), with an inline script in
  `index.html` applying it pre-paint to avoid a flash. The `<ThemeToggle/>` lives in the sidebar.
  This behavior is left untouched — the only change is where the widget is rendered.
- The `Maintenance` view (route + `/maintenance/*` endpoints + `maintenance.enabled` gate) is the
  closest existing precedent for this feature.

## Field classification

| Field | Editable | Apply semantics |
|---|---|---|
| `training.num_gpus` | yes | live (next job) |
| `training.master_port` | yes | live (next job) |
| `training.extra_args` | yes | live (next job) |
| `training.env` (key/value map) | yes | live (next job) |
| `maintenance.enabled` | yes | live (re-applied to env on write) |
| `maintenance.allow_pip` | yes | live (re-applied to env on write) |
| `ui.public` | yes | **restart required** (badge) |
| `ui.token` | yes | **restart required** (badge) |
| `ui.host` | no | read-only / informational |
| `ui.port` | no | read-only / informational |
| `ui.data_dir` | no | read-only / informational |

## Backend

New module `rengu_flow_ui/settings_store.py`:

- `read_settings() -> dict` — load the TOML with tomlkit; return a structured payload split into
  `editable`, `restartRequired`, `readOnly`, plus metadata (`path`, `exists`). Falls back to
  dataclass defaults when the file is missing.
- `write_settings(patch: dict) -> dict` — merge only the editable + restart-required fields onto
  the existing tomlkit document (non-destructive: untouched keys/comments preserved), validate,
  write atomically, then return the freshly re-read settings.

Validation rules: `num_gpus >= 1`, `master_port`/`port` are ints in `1..65535`, `training.env`
keys are non-empty strings and values are stringified. Validation failures return HTTP 422 with a
field-level message; no partial write.

Two endpoints in `app.py`:

- `GET  {API_PREFIX}/settings` → `read_settings()`.
- `PUT  {API_PREFIX}/settings` → validate + `write_settings()`, then:
  - reload the cached config (`load_local_config()`),
  - re-apply `maintenance.*` to `os.environ` (override, not setdefault) so it takes effect live,
  - respond with the re-read settings.

No changes to `UiConfig` / `local_config.py` — the theme is not a TOML field.

## Theme (frontend)

Unchanged behavior: theme stays a per-browser `localStorage` preference handled entirely by
`useTheme.ts` and the pre-paint inline script in `index.html`. No server round-trip, no API.
The **only** change is that `<ThemeToggle/>` is rendered inside the new Settings view instead of
the sidebar.

## Frontend — view + navigation

- New route `/settings` → `views/SettingsView.vue`; sidebar entry (gear icon, near Docs /
  Maintenance) in both `App.vue` nav blocks; breadcrumb label "Configuration".
- The view reuses existing form primitives (`ConfigFormSectionCard`, key-value field for
  `training.env`) with cards per section: **Appearance** (theme widget only — no TOML), **Training**,
  **Training env**, **Maintenance**, **Server** (read-only host/port/data_dir + restart-required
  public/token).
- `restartRequired` fields show a "restart to apply" badge; read-only fields are disabled/display.
- `<ThemeToggle/>` is **removed from the sidebar** (both nav blocks) and rendered only inside the
  Appearance card. Its `useTheme` wiring is unchanged.
- API client: add `getSettings()` / `updateSettings(patch)` to `api.ts`.

## Tests

- Backend: `settings_store` round-trip preserves comments and unrelated keys; merge is
  non-destructive; validation rejects bad port / `num_gpus`; `GET`/`PUT` endpoints return the
  expected shape; `maintenance` env re-applied after `PUT`.
- Frontend: a `SettingsView` smoke test (theme widget renders + form fields bind). No new theme
  tests — `useTheme` is unchanged.

## Out of scope

- Editing `host`/`port`/`data_dir` from the UI (read-only).
- Live restart of the server from the UI.
- Persisting the theme to the TOML / making it server-side — theme stays per-browser
  (`localStorage`); only the widget moves.
