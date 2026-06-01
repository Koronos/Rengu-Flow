# Rengu Flow

> **Preliminary release (v0.1.x)** — Rengu Flow is under active development. APIs, config keys, CLI commands, and documentation may change in breaking or non-breaking ways between releases. Pin versions and re-read the docs when upgrading.

A **TOML-driven training framework** for diffusion models. You define a config file (model, adapter, optimizer, dataset, etc.) and run training with DeepSpeed. The framework is modular: it supports multiple model types (e.g. SDXL), adapters (LoRA, LoKr), optimizers, and schedulers, all selected and configured via TOML.

## Overview

- **Config-first**: One main TOML file points to a dataset TOML and sets model, adapter, optimizer, LR scheduler, and training options.
- **Modular**: Models, adapters, optimizers, and schedulers are pluggable; you can extend the framework with new ones.
- **DeepSpeed**: Training runs with DeepSpeed (single or multi-GPU). Checkpoints and adapter saves follow the config (e.g. by epoch or step).
- **External control**: Signal files in the run directory (e.g. `save`, `save_quit`) let you trigger checkpoints or a clean exit without an API.

## Requirements

- **OS**: Linux (required for the `rengu` CLI and recommended for DeepSpeed/CUDA).
- **uv**: Required for `./rengu`, `./start-ui.sh`, and documented setup ([CLI guide](docs/user/cli.md)). uv installs Python ≥ 3.10 into `.venv` when needed.
- **GPU**: NVIDIA GPU with CUDA for training; see [pyproject.toml](pyproject.toml) for Python dependencies (PyTorch, DeepSpeed, etc.).

## Models and adapters

| Model | Adapters | Description |
|-------|----------|-------------|
| **SDXL** (`sdxl`) | LoRA, LoKr | Stable Diffusion XL; train LoRA or LoKr on the transformer. |
| **Cosmos Predict2** (`cosmos_predict2`) | LoRA, LoKr, full finetune | Cosmos Predict2 DiT + Qwen VAE + Qwen3/T5. **Anima** checkpoints are this architecture — train them with `type = "cosmos_predict2"` (Anima is the checkpoint branding, not a config type). |

Set `[model] type = "sdxl"` or `type = "cosmos_predict2"` and `[adapter] type = "lora"` or `type = "lokr"`. Omit `[adapter]` for full-model finetune. See [SDXL training](docs/user/training-sdxl-lora-lokr.md) or [Cosmos Predict2 / Anima](docs/user/training-cosmos-predict2-lora-lokr-finetune.md). Optional deps: `pip install -e ".[cosmos_predict2]"` for the latter.

## Installation

From the project root (Linux):

```bash
./rengu init          # rengu.local.toml + uv sync
./rengu init ui       # optional web UI dependencies
```

Advanced: run `uv sync` yourself, then `.venv/bin/rengu`. The `./rengu` wrapper runs `uv sync` on first use. See [CLI guide](docs/user/cli.md).

Optional extras: `./rengu init cosmos`, `./rengu init lycoris`, etc.

## Quick start

1. **Setup:** `./rengu init` (and `./rengu init ui` if you use the web UI).

2. **Local settings (optional).** Copy `rengu.local.toml.example` to `rengu.local.toml` (created automatically by `rengu init`) for UI host/port and training launcher defaults. Model checkpoint paths go in the **training TOML**, not here.

3. **Prepare a training config:**

   ```bash
   cp examples/minimal_config_lora_sdxl.toml my_train.toml
   ```

   Edit `my_train.toml`: set `dataset` and `[model]` paths (e.g. `checkpoint_path` for SDXL).

4. **Run training:**

   ```bash
   ./rengu train --config my_train.toml
   ```

   Or with DeepSpeed directly: `deepspeed --num_gpus=1 -m rengu_flow.main --config my_train.toml`

   Validate only: `./rengu validate --config my_train.toml`

## Web UI (optional)

```bash
./rengu ui start
```

See [Web UI user guide](docs/user/web-ui.md) and [CLI guide](docs/user/cli.md).

## Documentation

- [User guide](docs/user/) — Training, config, and signal files for external control.
- [Developer guide](docs/developer/) — Extending the framework (APIs, adding models and signals). Includes [testing](docs/developer/testing.md) (how to run and extend the test suite).
- [Architecture](docs/developer/architecture.md) — Design goals and execution flow.
- [Implementation backlog](docs/BACKLOG.md) — Planned / deferred features.
- [Third-party notices](THIRD_PARTY_NOTICES.md) — Upstream licenses (diffusion-pipe, NVIDIA, etc.).

See [docs/README.md](docs/README.md) for the full index.

## License

Rengu Flow is distributed under the **GNU General Public License v3.0 or later** ([LICENSE](LICENSE)). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for incorporated components.
