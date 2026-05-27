# Web UI (user guide)

Renga Flow includes an **optional** local web interface to manage training configs, start and stop jobs, send [signal files](signal-files.md), and inspect progress. It runs on the same machine as training and does not change the training process unless you enable `enable_status_file` in config (see below).

## Quick start

From the repository root:

```bash
./start-ui.sh
```

### Development mode (hot reload)

While editing the UI or Python API, use dev mode so you do not rebuild or restart manually:

```bash
./start-ui-dev.sh
```

Run it **from a terminal** in the repo root (not by double-clicking the file — the window would close when the script exits). If something fails, the script waits for Enter so you can read the error.

This starts (Linux or WSL — same environment as training):


| Process | URL | Behavior |
|---------|-----|----------|
| Vite dev server | [http://127.0.0.1:5173](http://127.0.0.1:5173) | Vue HMR — save a `.vue` file and the browser updates |
| API (`renga-flow-ui`) | [http://127.0.0.1:8765](http://127.0.0.1:8765) | `uvicorn --reload` on `renga_flow_ui/` |

Open the **5173** URL in the browser (Vite proxies `/api` to the API). Press **Ctrl+C** in the terminal to stop both.

Use `./start-ui.sh` when you want production-like serving from `ui/web/dist/` (no separate Vite process).

Run from a Linux environment or WSL (training dependencies are not supported on native Windows).

**Important:** leave the terminal window open while the UI runs. Closing the window stops the server.

The web client uses **Element Plus** (Vue 3) with a responsive layout: on phones, navigation opens in a side drawer and tables/forms stack for monitoring jobs, runs, and signals on the go.

This script:

1. Installs Python dependencies (`renga-flow` with the `[ui]` extra) via **uv** (if `uv.lock` is present) or **pip** in `.venv`
2. Builds the web frontend with **npm** when `ui/web/dist/` is missing (or pass `--rebuild-web`)
3. Starts the control server (default [http://127.0.0.1:8765](http://127.0.0.1:8765)) and opens your browser once `/api/v1/health` responds

### Script options

| Flag | Description |
|------|-------------|
| `--no-open` | Do not open a browser tab |
| `--rebuild-web` | Force `npm ci` and `npm run build` in `ui/web/` |

### Settings (`start-ui.sh`)

Edit the config block near the top of [`start-ui.sh`](../../start-ui.sh):

| Setting | Default in script | Description |
|---------|-------------------|-------------|
| `RENGA_FLOW_UI_HOST` | `127.0.0.1` | Bind address |
| `RENGA_FLOW_UI_PORT` | `8765` | HTTP port |
| `RENGA_FLOW_UI_DATA` | `$ROOT/.renga-flow-ui` | Config library, job DB, logs (gitignored) |
| `RENGA_FLOW_UI_TOKEN` | (commented out) | If set, API requests need `X-Renga-Flow-Token` |

The script exports those variables for the server. To drive them from the shell environment instead, change the block (see the comment in the script).

The script does **not** install CUDA, PyTorch, or DeepSpeed. Use your existing training environment for GPU jobs.

### Where configs are stored

By default the UI keeps its library under **`.renga-flow-ui/`** at the repository root (gitignored):

- `jobs.db` — SQLite: training configs, dataset configs, and job queue/history
- `staging/` — per-job `train.toml` (+ dataset copy) materialized when you launch training
- `logs/` — subprocess stdout from jobs started in the UI

Configs and datasets are stored as **TOML text in the database** (not as separate `.toml` files). Use **Export TOML** or drag-drop import when you need files on disk. Training still receives real `.toml` paths in `staging/` and copies them into the run folder.

In Docker, mount that folder (adjust `RENGA_FLOW_UI_DATA` in `start-ui.sh` to the mount path inside the container).

## What you can do in the UI

| Area | Description |
|------|-------------|
| **Docs** | In-app guide index (`docs/user/*.md`) from the **Docs** nav item |
| **Training** | **Form** editor (all major TOML sections: model, adapter/network, optimizer, scheduler, training, checkpoints, eval, preview, monitoring) plus raw **TOML** tab; lists registered models, adapters, and optimizers from the framework |
| **Datasets** | Library of dataset TOMLs: multiple `[[directory]]` folders per file, per-folder **Scan**, live **Dataset preview** (folder stats plus thumbnail gallery), **Compose** to merge library datasets into one file (OneTrainer-style packs); in-app links to [dataset config](dataset-config.md) docs |
| **Train** | Queue runs after choosing a config in the **Training** library (edit/validate there first); tab **Runs on disk** lists output folders; **Import script run** registers an existing `output/…` folder from terminal training |
| **Config form** | Required fields first; visual dataset picker; click the **i** icon to open in-app help (loads `docs/**/*.md` from the repo) |
| **Runs** | List folders under `output_dir`, view metrics, send signals to active runs |
| **TensorBoard** | **Open TensorBoard** on the run detail or Output runs page — starts TensorBoard via `uv` (no extra pip install); compares all runs under the same `output_dir` |
| **Signals** | Same files as [signal files](signal-files.md): `save`, `save_quit`, `export_model`, `export_model_quit`, `preview` |
| **Host bar** | Top bar shows live CPU/RAM/GPU load, temperatures, and VRAM; click for per-core CPU, sensors, swap, and full GPU details (via `nvidia-smi` when available) |

### Suggested workflow

1. **Datasets** (optional) — build or import dataset TOMLs with your image folders.
2. **Training** — create or import a training config; set `dataset = ...` via the dataset picker; validate.
3. **Train** — click **Choose config in library** to open Training, edit if needed, then **Use for training job**; set GPUs/resume and queue or start.

To **continue a run** with new settings (e.g. more epochs), open the job or filesystem run detail and click **Continue training…**. That loads the TOML from the run folder into **Training**; after editing, queue a continuation job. Training resumes in the same output folder and updates the TOML snapshot there.

If you already trained from the terminal, use **Import script run** on the **Train** page: pick a folder under your `output_dir` (or paste an absolute path). The UI links that run for TensorBoard metrics and [signal files](signal-files.md), and can copy the run’s `*.toml` files into the config/dataset library.

The app opens on **Configs** by default. The Jobs page does not include an inline config dropdown so you always review the full config in the library before running.

### Form fields vs validation

The config **form only shows options that apply to the model type** you selected (for example SDXL checkpoint paths vs Cosmos main/VAE/text files). **Validate** still checks the full TOML rules used by training:

- Required weights for that model must be set (or filled via the form).
- Some models accept **alternate keys in raw TOML only** (e.g. Cosmos can use `t5_path` instead of `llm_path` even though the form defaults to Qwen3).
- Options such as **block swap** only apply to models that support them; the form hides them for others, and validation errors if they are set anyway.

If Validate fails on a field you do not see, switch to the **TOML** tab or change **Model type** — you may have leftover keys from another model.

## Resume vs new run

- **Resume**: In Jobs, set **Resume folder** to an existing run directory (e.g. `output/20250217_14-30-00`) and start. The UI passes `--resume_from_checkpoint` to the trainer. Prefer the TOML snapshot copied into that run folder over an edited library config.
- **New run**: Leave resume empty; a new timestamped folder is created under `output_dir`.

## Optional status file (low overhead)

In your training TOML:

```toml
[monitoring]
enable_status_file = true
```

When enabled, rank 0 writes `status.json` in the run directory every `logging_steps` for faster progress display in the UI. Default is `false` (no extra trainer work).

## Advanced: run server only

If dependencies are already installed, export the same variables as in `start-ui.sh` (or rely on the Python fallback path `<repo>/.renga-flow-ui`):

```bash
pip install -e ".[ui]"
export RENGA_FLOW_UI_DATA="$(pwd)/.renga-flow-ui"
renga-flow-ui serve --host 127.0.0.1 --port 8765
```

Serve the built SPA from `ui/web/dist/` (run `./start-ui.sh` once to build, or `cd ui/web && npm ci && npm run build`).

## TensorBoard from the UI

On **Output runs** or a **run/job detail** page, click **Open TensorBoard**. The UI runs:

```bash
uv run --no-project --with 'tensorboard>=2.14' tensorboard --logdir=<output_dir>
```

So you do **not** need `tensorboard` in the project venv unless `.venv/bin/tensorboard` exists (then it is preferred); otherwise you need [uv](https://docs.astral.sh/uv/) on `PATH`. Use **Stop TensorBoard** on the same page to shut down the local server. The log directory is the parent folder (e.g. `output/`), not a single run folder, so run names appear in the TensorBoard sidebar. Logs from a failed start are written to `{RENGA_FLOW_UI_DATA}/logs/tensorboard.log`.
