# Web UI (user guide)

Rengu Flow includes an **optional** local web interface to manage training configs, start and stop jobs, send [signal files](signal-files.md), and inspect progress. It runs on the same machine as training and does not change the training process unless you enable `monitoring.enable_status_file` in config (see below).

## Quick start

From the repository root:

```bash
./rengu init ui    # first time: local config + .venv with [ui] extra
./rengu ui start
```

Or double-click **`start-ui.sh`** in the repo root — it runs `uv sync --extra ui` (creates `.venv` if needed) and starts the UI. You need **[uv](https://docs.astral.sh/uv/)** on `PATH`; you do **not** need a separate system `python3` or `source .venv/bin/activate`.

Run from Linux or WSL. See [CLI guide](cli.md) for all commands.

**Important:** leave the terminal open while the UI runs. Closing the window stops the server.

`rengu ui start`:

1. Runs `uv venv` (if needed) and `uv sync --extra ui` unless `--skip-sync`
2. Builds `ui/web/dist/` with npm when missing (`--rebuild-web` to force)
3. Starts the API (default [http://127.0.0.1:8765](http://127.0.0.1:8765)) and opens the browser when `/api/v1/health` responds

| Flag | Description |
|------|-------------|
| `--no-open` | Do not open a browser tab |
| `--rebuild-web` | Force `npm ci` and `npm run build` in `ui/web/` |
| `--skip-sync` | Skip `uv sync` (launcher scripts use this after `uv sync --extra ui`) |

### Optional dependencies (Cosmos, LyCORIS, optimizers)

Core training dependencies (PyTorch, **torchvision**, DeepSpeed) are always installed by the base sync, so **SDXL trains with only the `[ui]` extra** — no model-specific extra required.

When you **start training** from the UI or run `rengu train` / `rengu validate` / `rengu cache`, Rengu Flow reads your training TOML and runs `uv sync` for any *additional* model-specific extras it needs (for example the **Cosmos Predict2** text stack when `[model] type = "cosmos_predict2"`, **LyCORIS** for LoKr, or the **optim** extra for extended optimizers). You do not need maintenance mode or a separate install step.

### Settings (`rengu.local.toml`)

Copy `rengu.local.toml.example` to `rengu.local.toml` (or run `./rengu init`). Edit the `[ui]` section:

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `127.0.0.1` | Bind address |
| `port` | `8765` | HTTP port |
| `data_dir` | `data` | Config library, job DB, logs (gitignored) |
| `token` | (optional) | If set, API requests need `X-Rengu-Flow-Token` |

The script does **not** install CUDA, PyTorch, or DeepSpeed. Use your existing training environment for GPU jobs.

### Where configs are stored

By default the UI keeps its library under **`data/`** at the repository root (a non-hidden, gitignored folder):

- `jobs.db` — SQLite: training configs, dataset configs, and job queue/history
- `staging/` — per-job `train.toml` (+ dataset copy) materialized when you launch training
- `logs/` — subprocess stdout from jobs started in the UI

Configs and datasets are stored as **TOML text in the database** (not as separate `.toml` files). Use **Export TOML** or drag-drop import when you need files on disk. Training still receives real `.toml` paths in `staging/` and copies them into the run folder.

In Docker, mount that folder (set `data_dir` in `rengu.local.toml` to the mount path inside the container).

## Navigation

| Nav label | Route | Purpose |
|-----------|-------|---------|
| **Docs** | `/docs` | In-app index of `docs/user/*.md` |
| **Datasets** | `/datasets` | Dataset TOML library (folders, augmentation, compose) |
| **Configs** | `/configs` | Training config TOML library (form + raw TOML) |
| **Runs** | `/runs` | Launch jobs, queue, live monitor, output folders, run detail |

**Maintenance** (`/maintenance`) appears only when `RENGUFLOW_MAINTENANCE=1` — see [Maintenance](maintenance.md).

There is no separate **Signals** page: signal buttons appear on **run detail** and **job detail** while a job is running (or during disk-export wait).

## What you can do in the UI

| Area | Description |
|------|-------------|
| **Docs** | User guides from the repository; field **i** icons open the same markdown in a drawer |
| **Datasets** | Library of dataset TOMLs: `[[directory]]` folders, **Scan**, thumbnail gallery, **Compose** merged datasets; **Augmentation** tab (MVP presets) — [dataset config](dataset-config.md), [augmentation](dataset-augmentation.md) |
| **Configs** | Training TOML library: form editor (model, adapter, training, previews, eval, monitoring) and **TOML** tab; **Validate**, **Save**, duplicate/delete on the list |
| **Runs** | **Launch** training from a library config (GPUs, optional resume folder); job queue; **Runs on disk** tab; **Import script run** for terminal-trained folders |
| **Run / job detail** | Metrics, TensorBoard link, signal buttons, continue training, checkpoint/export docs |
| **TensorBoard** | Started from run detail — `uv run tensorboard --logdir=<output_dir>` |
| **Host bar** | CPU/RAM/GPU (via `nvidia-smi` when available) |

### Suggested workflow

1. **Datasets** (optional) — build or import dataset TOMLs with your image folders.
2. **Configs** — create or import a training config; set `dataset` via the dataset picker; **Validate**.
3. **Runs** — pick a config from the library (**Use for training job**), set GPUs and optional resume folder, then start or queue.

**New config** seeds an SDXL LoRA template. Use **Import TOML…** on the config list if you already have a file on disk.

To **continue a run**, open job or filesystem run detail → **Continue training…** → edits in **Configs** → queue continuation. Training resumes in the same output folder.

For terminal-trained runs, use **Import script run** on **Runs** to link the folder for metrics and signals.

### Form fields vs validation

The config **form shows fields for the selected model type** (SDXL vs Cosmos paths, adapter options, previews). **Validate** applies the same rules as the CLI trainer.

- Required weights must be set.
- Some Cosmos keys exist only in raw TOML (e.g. `t5_path` instead of `llm_path`).
- **Training block swap** (`blocks_to_swap`) appears when the model supports it (SDXL, Cosmos). Works for both adapter and **full-model** training — full-model also needs `optimizer.gradient_release`. The **Block-swap prefetch** (`block_swap_prefetch`) toggle appears only once `blocks_to_swap > 0` **and** `optimizer.gradient_release` is set — the case where the backend can actually overlap transfers (adapter runs keep trainable params resident and ignore it). It is an opt-in overlap that helps on larger GPUs but is counterproductive on tight 8 GB WSL2 (off by default). See [VRAM optimization](../developer/vram-optimization.md) and [Shared training techniques](../developer/training-techniques.md). Cosmos preview may also use `preview.preview_blocks_to_swap`.

If Validate fails on a hidden field, check the **TOML** tab for leftover keys from another model type.

## Resume vs new run

On **Runs**, when launching a job:

| Field | Purpose |
|-------|---------|
| **Resume folder** | Existing run directory under `output_dir` (e.g. `20250217_14-30-00`). Passes `--resume_from_checkpoint` to the trainer. Prefer the TOML snapshot in that folder over an edited library config. |
| *(empty)* | New timestamped folder under `output_dir`. Optional **`run_name`** in config is appended after the date (e.g. `20250217_14-30-00_my_experiment`). |

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **`run_name`** | Label for output folders and TensorBoard | Letters, digits, `.`, `_`, `-`; max 80 chars; no `/` or `\` | Omitted → timestamp-only folder |

## Optional status file (low overhead)

In your training TOML:

```toml
[monitoring]
enable_status_file = true
```

When enabled, rank 0 writes `status.json` in the run directory every `logging_steps` for faster progress in the UI. Default is `false`.

## Advanced: run server only

```bash
./rengu ui build
./rengu ui serve --host 127.0.0.1 --port 8765
```

Developer mode: `./rengu ui dev` (Vite on port 5173, API on 8765). Flags: `--no-open`, `--dev-port`, `--skip-sync`, `--no-reload-api` — see [CLI](cli.md).

## TensorBoard from the UI

On run detail, click **Open TensorBoard**. The UI runs TensorBoard with `--logdir` set to your training **`output_dir`** (parent of run folders), so all runs appear in the sidebar. Logs: `{data_dir}/logs/tensorboard.log`.
