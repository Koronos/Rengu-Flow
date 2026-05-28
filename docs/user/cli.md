# Renga CLI (`renga`)

Linux-only command-line interface for setup, training, and the web UI. Run from the repository root as `./renga` (requires [uv](https://docs.astral.sh/uv/) on `PATH`; uv creates `.venv` and installs Python — no separate system `python3` needed).

## First run

```bash
./renga init          # renga.local.toml + uv sync (base)
./renga init ui       # also install the [ui] extra
./renga ui start      # build frontend if needed and open the control panel
```

`renga install` is deprecated; use `renga init` instead.

## Local config vs training config

| File | Purpose |
|------|---------|
| **`renga.local.toml`** (copy from `renga.local.toml.example`, gitignored) | Machine settings: UI host/port, training launcher defaults, optional subprocess env vars |
| **Training TOML** (`my_train.toml`, library configs, `examples/`) | Model paths under `[model]`, dataset, adapter, optimizer, `pipeline_stages`, etc. |

Checkpoint paths belong in the **training TOML**, not in `renga.local.toml`.

## Commands

| Command | Description |
|---------|-------------|
| `renga init [profiles…]` | Create `renga.local.toml`, UI data dir, `uv sync` (profiles: `base`, `ui`, `cosmos`, `optim`, `lycoris`, `dev`, `all`) |
| `renga init --only-config` | Local TOML + dirs only; skip `uv sync` |
| `renga update [profiles…]` | Re-sync from `uv.lock` (`--all-extras` for every documented extra) |
| `renga train --config PATH` | Launch DeepSpeed (uses `[training]` in local TOML) |
| `renga validate --config PATH` | Validate training config and exit |
| `renga cache --config PATH` | Run `--cache_only` |
| `renga dump-dataset PATH` | Inspect dataset TOML |
| `renga ui start` | `uv sync --extra ui`, build `ui/web/dist`, serve API, open browser |
| `renga ui serve` | API only (`--host`, `--port`, `--reload`) |
| `renga ui dev` | API with reload + Vite dev server |
| `renga ui build` | `npm run build` in `ui/web` |
| `renga ui reset-db` | Reset UI SQLite library |

Legacy: `renga --config foo.toml` (without `train`) still works.

Before training, validate, or cache, Renga inspects the training TOML and runs `uv sync` for any missing optional extras (e.g. **cosmos** when `[model] type = "cosmos_predict2"`).

## Training environment (`[training.env]`)

Optional subprocess environment for `renga train` and UI jobs. **Default: empty** — nothing is set unless you add keys.

Keys are **exact** environment variable names; values are strings:

```toml
[training.env]
NCCL_P2P_DISABLE = "1"
NCCL_IB_DISABLE = "1"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
RENGA_TUNING_TF32_APPLY = "1"
```

### `RENGA_TUNING_TF32_APPLY`

Renga-flow-specific flag (not a standard CUDA variable). When set to `"1"`, training enables TF32 matmul/cuDNN and `cudnn.benchmark` on Ampere/Ada GPUs (e.g. RTX 30xx/40xx). Leave unset on older GPUs or if you do not want this behavior.

See also [Web UI](web-ui.md) for UI-related keys under `[ui]` in the same file.
