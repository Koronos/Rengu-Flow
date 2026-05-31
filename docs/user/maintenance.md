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

Deletes `jobs.db` under your UI data directory (`RENGU_FLOW_UI_DATA`, default `data/`) and creates empty tables for training configs, datasets, and jobs. **All saved configs, datasets, and job history are lost.** Staging files and job logs under the same folder are not removed.

You must type `RESET` in the confirmation dialog. The CLI equivalent is:

```bash
./rengu ui reset-db
```

### Back up / restore the library (migration)

Configs and datasets are portable as TOML files, so you can back them up or carry them across
a schema change (the DB is disposable; these files are the durable copy):

```bash
rengu-flow-ui export-library ./library-backup   # writes configs/<id>.toml + datasets/<id>.toml
rengu-flow-ui import-library ./library-backup    # restores rows under their original ids
rengu-flow-ui import-library ./library-backup --overwrite  # replace existing ids
```

Each exported file is a valid training/dataset TOML with an extra `[__rengu_index]` table
(id, name, timestamps) that the trainer ignores. Import skips files it does not recognize and,
without `--overwrite`, ids that already exist. Run history (jobs) is not part of this export —
finished run folders are re-added with **Import run** instead.

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

## How dependencies are installed (additive, on-demand)

rengu-flow's installer is **additive and never destructive**: it only ever *adds* what's missing and never removes packages you already have. This matters because users install **custom optimizers/schedulers** (referenced by qualified path in TOML) and other packages by hand — those must survive every install/update.

- **Additive sync.** Every `uv sync` rengu-flow runs uses `--inexact`, so syncing one profile (e.g. `ui`) never removes other extras (`cosmos_predict2`, `optim`, `dev`) or your hand-installed packages.
- **On-demand.** Optional extras are installed only when their modules aren't importable yet. Starting a Cosmos training auto-installs the `cosmos_predict2` extra; launching the UI ensures the `ui` extra; etc.
- **Git/VCS packages.** Backends uv can't express as a pyproject extra are installed additively with `uv pip install` (register them in `rengu_flow/install/profiles.py::PROFILE_GIT_REQUIREMENTS`).
- **Self-healing.** Installed profiles are recorded in `.rengu-flow/installed-profiles.json` (gitignored). On UI launch (and `rengu update`) rengu-flow re-ensures them additively, so an environment that lost packages is restored automatically.

> **Caveat — don't launch with `uv run`.** `uv run <cmd>` performs an *exact* sync to the project's base dependencies **before** running, which removes every optional extra and any hand-installed package (this ignores `--inexact`). Launch via **`./rengu …`**, the venv binary **`.venv/bin/rengu …`**, or **`uv run --no-sync rengu …`**. The bundled `./rengu` and `./start-ui.sh` launchers already avoid this; self-healing (above) also recovers if it happens.

## API (when enabled)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/maintenance/enabled` | Always available; `{ "enabled": bool }` |
| GET | `/api/v1/maintenance/status` | DB path, tables, submodule/git info, dependency commands |
| POST | `/api/v1/maintenance/database/reset` | Body `{ "confirmation": "RESET" }` or `?confirm=true` |
| POST | `/api/v1/maintenance/submodules/update` | Submodule init/update |
| POST | `/api/v1/maintenance/deps/install` | Body `{ "profile": "cosmos_predict2", "execute": false }` |
