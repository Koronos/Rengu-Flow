# Rengu CLI (`rengu`)

Linux-only command-line interface for setup, training, and the web UI. Run from the repository root as `./rengu` (requires [uv](https://docs.astral.sh/uv/) on `PATH`; uv creates `.venv` and installs Python — no separate system `python3` needed).

## First run

```bash
./rengu init          # rengu.local.toml + uv sync (base)
./rengu init ui       # also install the [ui] extra
./rengu ui start      # build frontend if needed and open the control panel
```

`rengu install` is deprecated; use `rengu init` instead.

## Local config vs training config

| File | Purpose |
|------|---------|
| **`rengu.local.toml`** (copy from `rengu.local.toml.example`, gitignored) | Machine settings: UI host/port, training launcher defaults, optional subprocess env vars |
| **Training TOML** (`my_train.toml`, library configs, `examples/`) | Model paths under `[model]`, dataset, adapter, optimizer, `pipeline_stages`, etc. |

Checkpoint paths belong in the **training TOML**, not in `rengu.local.toml`.

## Commands

| Command | Description |
|---------|-------------|
| `rengu init [profiles…]` | Create `rengu.local.toml`, UI data dir, `uv sync` (profiles: `base`, `ui`, `cosmos`, `optim`, `lycoris`, `dev`, `all`) |
| `rengu init --only-config` | Local TOML + dirs only; skip `uv sync` |
| `rengu update [profiles…]` | Re-sync from `uv.lock` (`--all-extras` for every documented extra) |
| `rengu train --config PATH` | Launch DeepSpeed (see flags below) |
| `rengu validate --config PATH` | Validate training config and exit |
| `rengu cache --config PATH` | Run dataset cache only (`--cache_only` on trainer) |
| `rengu dump-dataset PATH` | Inspect dataset TOML |
| `rengu ui start` | `uv sync --extra ui`, build `ui/web/dist`, serve API, open browser |
| `rengu ui serve` | API only (`--host`, `--port`, `--reload`) |
| `rengu ui dev` | API with reload + Vite dev server |
| `rengu ui build` | `npm run build` in `ui/web` |
| `rengu ui reset-db` | Reset UI SQLite library |

Legacy: `rengu --config foo.toml` (without `train`) still works.

### `rengu train` flags

| Flag | Purpose | Default |
|------|---------|---------|
| `--config PATH` | Training TOML (required) | — |
| `--num-gpus N` | Override GPU count for this run | `[training].num_gpus` in `rengu.local.toml`, else `1` |
| `--master-port PORT` | DeepSpeed master port | `[training].master_port` or `29500` |
| `--resume-from-checkpoint` | Resume from `latest` in the run directory | off |
| trailing args after `--` | Passed to `rengu_flow.main` (trainer flags) | `[training].extra_args` from local TOML |

Example with trainer flags:

```bash
./rengu train --config my.toml -- --regenerate_cache
```

### `rengu cache`

Same as `train` but the launcher appends `--cache_only`. Extra args after `--` are forwarded to the trainer.

### `rengu init` profiles

| Profile | Extra installed |
|---------|-----------------|
| `base` | Core training stack (default) |
| `ui` | Web control plane |
| `cosmos` / `cosmos_predict2` | Cosmos Predict2 / Anima |
| `optim` | Extended optimizers |
| `lycoris` | LoKr (LyCORIS) |
| `dev` | Dev tools |
| `all` | All documented extras |

### `rengu ui` flags

**`ui start`:** `--no-open`, `--rebuild-web`, `--skip-sync` (see [Web UI](web-ui.md)).

**`ui dev`:** `--no-open`, `--dev-port` (default `5173`), `--skip-sync`, `--no-reload-api`.

**`ui serve`:** `--host`, `--port`, `--reload`.

### `[training]` in `rengu.local.toml`

| Key | Purpose | Default |
|-----|---------|---------|
| `num_gpus` | Default for `rengu train` | `1` |
| `master_port` | DeepSpeed master port | `29500` |
| `extra_args` | Default trainer argv list | `[]` |

### Trainer flags (`rengu_flow.main`)

Passed via `rengu train --config X -- FLAG …` or `extra_args` in local TOML:

| Flag | Purpose |
|------|---------|
| `--resume_from_checkpoint` | Resume from run `latest` |
| `--cache_only` | Build dataset cache and exit |
| `--regenerate_cache` / `--regenerate_text_cache` | Force cache rebuild |
| `--trust_cache` | Skip cache validation |
| `--validate-only` | Load config, validate, exit |
| `--dump_dataset` | Dump dataset layout and exit |
| `--reset_dataloader` / `--reset_optimizer` / `--reset_optimizer_params` | Partial resume controls |
| `--local_rank` | Set by DeepSpeed launcher |
| `--master_port` | DeepSpeed port (also on `rengu train`) |
| `--i_know_what_i_am_doing` | Bypass some safety checks |

Legacy one-shot: `rengu --config foo.toml --cache_only` (same as `rengu cache --config foo.toml`).

Before training, validate, or cache, Rengu inspects the training TOML and runs `uv sync` for any missing optional extras (e.g. **cosmos** when `[model] type = "cosmos_predict2"`).

## Training environment (`[training.env]`)

Optional subprocess environment for `rengu train` and UI jobs. **Default: empty** — nothing is set unless you add keys.

Keys are **exact** environment variable names; values are strings:

```toml
[training.env]
NCCL_P2P_DISABLE = "1"
NCCL_IB_DISABLE = "1"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
RENGU_TUNING_TF32_APPLY = "1"
```

> **WSL2 note.** Do **not** set `PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"` on WSL: its
> `cuMemMap` path is unsupported under WSL2/WDDM and crashes cuDNN convolutions (SDXL's UNet) with
> `CUDA driver error: device not ready`. Rengu detects WSL and forces `expandable_segments:False`
> automatically (plus low-fragmentation defaults), so you can leave this key unset there.

### `RENGU_TUNING_TF32_APPLY`

Rengu-flow-specific flag (not a standard CUDA variable). When set to `"1"`, training enables TF32 matmul/cuDNN and `cudnn.benchmark` on Ampere/Ada GPUs (e.g. RTX 30xx/40xx). Leave unset on older GPUs or if you do not want this behavior.

See also [Web UI](web-ui.md) for UI-related keys under `[ui]` in the same file.
