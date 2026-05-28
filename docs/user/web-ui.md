# Web UI (user guide)

Renga Flow includes an **optional** local web interface to manage training configs, start and stop jobs, send [signal files](signal-files.md), and inspect progress. It runs on the same machine as training and does not change the training process unless you enable `enable_status_file` in config (see below).

## Quick start

From the repository root:

```bash
./renga init ui    # first time: local config + .venv with [ui] extra
./renga ui start
```

Or double-click **`start-ui.sh`** in the repo root — it runs `uv sync --extra ui` (creates `.venv` if needed) and starts the UI. You need **[uv](https://docs.astral.sh/uv/)** on `PATH`; you do **not** need a separate system `python3` or `source .venv/bin/activate`.

Run from Linux or WSL. See [CLI guide](cli.md) for all commands.

**Important:** leave the terminal open while the UI runs. Closing the window stops the server.

`renga ui start`:

1. Runs `uv venv` (if needed) and `uv sync --extra ui` unless `--skip-sync`
2. Builds `ui/web/dist/` with npm when missing (`--rebuild-web` to force)
3. Starts the API (default [http://127.0.0.1:8765](http://127.0.0.1:8765)) and opens the browser when `/api/v1/health` responds

| Flag | Description |
|------|-------------|
| `--no-open` | Do not open a browser tab |
| `--rebuild-web` | Force `npm ci` and `npm run build` in `ui/web/` |
| `--skip-sync` | Skip `uv sync` (launcher scripts use this after `uv sync --extra ui`) |

### Optional dependencies (Cosmos, LyCORIS, optimizers)

When you **start training** from the UI or run `renga train` / `renga validate` / `renga cache`, Renga reads your training TOML and runs `uv sync` for any missing extras (for example **Cosmos Predict2** when `[model] type = "cosmos_predict2"`). You do not need maintenance mode or a separate install step.

### Settings (`renga.local.toml`)

Copy `renga.local.toml.example` to `renga.local.toml` (or run `./renga init`). Edit the `[ui]` section:

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `127.0.0.1` | Bind address |
| `port` | `8765` | HTTP port |
| `data_dir` | `.renga-flow-ui` | Config library, job DB, logs (gitignored) |
| `token` | (optional) | If set, API requests need `X-Renga-Flow-Token` |

The script does **not** install CUDA, PyTorch, or DeepSpeed. Use your existing training environment for GPU jobs.

### Where configs are stored

By default the UI keeps its library under **`.renga-flow-ui/`** at the repository root (gitignored):

- `jobs.db` — SQLite: training configs, dataset configs, and job queue/history
- `staging/` — per-job `train.toml` (+ dataset copy) materialized when you launch training
- `logs/` — subprocess stdout from jobs started in the UI

Configs and datasets are stored as **TOML text in the database** (not as separate `.toml` files). Use **Export TOML** or drag-drop import when you need files on disk. Training still receives real `.toml` paths in `staging/` and copies them into the run folder.

In Docker, mount that folder (set `data_dir` in `renga.local.toml` to the mount path inside the container).

## What you can do in the UI

| Area | Description |
|------|-------------|
| **Docs** | In-app guide index (`docs/user/*.md`) from the **Docs** nav item |
| **Training** | **Form** editor (model, adapter, training, **Previews** tab with list+modal for `preview.prompts`, eval, monitoring) plus raw **TOML** tab; lists registered models, adapters, and optimizers from the framework |
| **Datasets** | Library of dataset TOMLs: multiple `[[directory]]` folders per file, per-folder **Scan**, live **Dataset preview** (folder stats plus thumbnail gallery), **Compose** to merge library datasets into one file (OneTrainer-style packs); **Augmentation** tab for global and per-folder diversity settings (presets and strategy overrides loaded from the training catalog); in-app links to [dataset config](dataset-config.md) docs |
| **Train** | Queue runs after choosing a config in the **Training** library (edit/validate there first); tab **Runs on disk** lists output folders; **Import script run** registers an existing `output/…` folder from terminal training |
| **Config form** | Required fields first; visual dataset picker; **Previews** tab separates global settings from per-prompt rows (Add/Edit/Duplicate/Remove); click the **i** icon to open in-app help (loads `docs/**/*.md` from the repo) |
| **Runs** | List folders under `output_dir`, view metrics, send signals to active runs |
| **TensorBoard** | **Open TensorBoard** on the run detail or Output runs page — starts TensorBoard via `uv` (no extra pip install); compares all runs under the same `output_dir` |
| **Signals** | Same files as [signal files](signal-files.md): `save`, `save_quit`, `export_model`, `export_model_quit`, `preview` |
| **Host bar** | Top bar shows live CPU/RAM/GPU load, temperatures, and VRAM; click for per-core CPU, sensors, swap, and full GPU details (via `nvidia-smi` when available) |
| **Maintenance** | Optional dev page: recreate `jobs.db`, submodule update, dependency install commands — see [Maintenance](maintenance.md) (`RENGAFLOW_MAINTENANCE=1`) |

### Suggested workflow

1. **Datasets** (optional) — build or import dataset TOMLs with your image folders.
2. **Training** — create or import a training config; set `dataset = ...` via the dataset picker; validate.
3. **Train** — click **Choose config in library** to open Training, edit if needed, then **Use for training job**; set GPUs/resume and queue or start.

**New config** seeds an SDXL LoRA template (dataset path, checkpoint path, optimizer, scheduler, epochs, batch size, `output_dir`). It does **not** include `synthetic_num_batches` or other debug-only smoke settings — point `dataset` at your dataset TOML and set `model.checkpoint_path` before you validate or train. Use **Import TOML…** on the config list if you already have a file on disk.

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
- **New run**: Leave resume empty; a new folder is created under `output_dir`. By default the folder name is a UTC timestamp only (e.g. `20250217_14-30-00`). Set optional **`run_name`** in the training config (Training → Form, **Run name**) to prefix the folder and TensorBoard sidebar entry, e.g. `my_experiment_20250217_14-30-00`. This is not the dataset library name.

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **`run_name`** | Human-readable label for output folders and TensorBoard | Non-empty string; letters, digits, `.`, `_`, `-` (no `/` or `\`); max 80 characters | Omitted → timestamp-only folder |

## Optional status file (low overhead)

In your training TOML:

```toml
[monitoring]
enable_status_file = true
```

When enabled, rank 0 writes `status.json` in the run directory every `logging_steps` for faster progress display in the UI. Default is `false` (no extra trainer work).

## Advanced: run server only

If dependencies are already installed:

```bash
./renga ui build
./renga ui serve --host 127.0.0.1 --port 8765
```

Developer mode: `./renga ui dev` (Vite HMR on port 5173, API on 8765 with Python auto-reload). Use `--no-reload-api` for a faster API-only startup.

## TensorBoard from the UI

On **Output runs** or a **run/job detail** page, click **Open TensorBoard**. The UI runs:

```bash
uv run --no-project --with 'tensorboard>=2.14' tensorboard --logdir=<output_dir>
```

So you do **not** need `tensorboard` in the project venv unless `.venv/bin/tensorboard` exists (then it is preferred); otherwise you need [uv](https://docs.astral.sh/uv/) on `PATH`. Use **Stop TensorBoard** on the same page to shut down the local server. The log directory is the parent folder (e.g. `output/`), not a single run folder, so run names appear in the TensorBoard sidebar. Logs from a failed start are written to `{RENGA_FLOW_UI_DATA}/logs/tensorboard.log`.
