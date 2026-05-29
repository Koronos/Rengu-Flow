# Maintenance (development)

The web UI includes an optional **Maintenance** page for local development: recreate the SQLite library, inspect git submodules, and copy (or optionally run) dependency install commands for the base stack, UI, and Cosmos Predict2 extras.

Maintenance is **off by default** so a shared or production control server cannot wipe your config library by accident.

## Enable maintenance

Set on the machine that runs `./rengu ui serve` (for example in `start-ui.sh` before export):

| Variable | Values | Effect |
|----------|--------|--------|
| `RENGUFLOW_MAINTENANCE` | `1`, `true`, `yes`, `on` | Enables `/api/v1/maintenance/*` and shows **Maintenance** in the sidebar |
| `RENGUFLOW_MAINTENANCE_ALLOW_PIP` | `1`, `true`, … | Allows **Run** on dependency profiles (executes `pip` in the server’s Python); otherwise only **Copy** |

Restart the control server after changing these variables.

Open [http://127.0.0.1:8765/maintenance](http://127.0.0.1:8765/maintenance) when enabled.

## What each action does

### Recreate database

Deletes `jobs.db` under your UI data directory (`RENGU_FLOW_UI_DATA`, default `.rengu-flow-ui/`) and creates empty tables for training configs, datasets, and jobs. **All saved configs, datasets, and job history are lost.** Staging files and job logs under the same folder are not removed.

You must type `RESET` in the confirmation dialog. The CLI equivalent is:

```bash
./rengu ui reset-db
```

### Git submodules

Runs `git submodule update --init --recursive` in the repository root. **rengu-flow** normally has no `.gitmodules` (Cosmos code is vendored in-tree); this is mainly useful if you use a diffusion-pipe-style clone with submodules. The page still reports submodule status when `.gitmodules` exists.

### Dependencies

Profiles map to optional dependencies in `pyproject.toml`:

| Profile | Install target |
|---------|----------------|
| `base` | `pip install -e .` |
| `ui` | `pip install -e ".[ui]"` |
| `cosmos_predict2` | `pip install -e ".[cosmos_predict2]"` |

Cosmos training needs the **cosmos_predict2** extra in the **same environment** you use for `rengu-flow` / DeepSpeed jobs—not only the UI server venv.

Prefer running copied commands in your training virtualenv. Server-side **Run** is optional and can take a long time; it installs into the Python process that hosts the UI.

## API (when enabled)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/maintenance/enabled` | Always available; `{ "enabled": bool }` |
| GET | `/api/v1/maintenance/status` | DB path, tables, submodule/git info, dependency commands |
| POST | `/api/v1/maintenance/database/reset` | Body `{ "confirmation": "RESET" }` or `?confirm=true` |
| POST | `/api/v1/maintenance/submodules/update` | Submodule init/update |
| POST | `/api/v1/maintenance/deps/install` | Body `{ "profile": "cosmos_predict2", "execute": false }` |
