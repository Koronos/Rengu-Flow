# Renga Flow

A **TOML-driven training framework** for diffusion models. You define a config file (model, adapter, optimizer, dataset, etc.) and run training with DeepSpeed. The framework is modular: it supports multiple model types (e.g. SDXL), adapters (LoRA, LoKr), optimizers, and schedulers, all selected and configured via TOML.

## Overview

- **Config-first**: One main TOML file points to a dataset TOML and sets model, adapter, optimizer, LR scheduler, and training options.
- **Modular**: Models, adapters, optimizers, and schedulers are pluggable; you can extend the framework with new ones.
- **DeepSpeed**: Training runs with DeepSpeed (single or multi-GPU). Checkpoints and adapter saves follow the config (e.g. by epoch or step).
- **External control**: Signal files in the run directory (e.g. `save`, `save_quit`) let you trigger checkpoints or a clean exit without an API.

## Requirements

- **Python** ≥ 3.10
- **OS**: Linux (recommended for DeepSpeed and CUDA). May work on other platforms with a compatible PyTorch/DeepSpeed setup.
- **GPU**: NVIDIA GPU with CUDA for training; see [pyproject.toml](pyproject.toml) for Python dependencies (PyTorch, DeepSpeed, etc.).

## Models and adapters

| Model | Adapters | Description |
|-------|----------|-------------|
| **SDXL** (`sdxl`) | LoRA, LoKr | Stable Diffusion XL; train LoRA or LoKr on the transformer. |
| **Cosmos Predict2** (`cosmos_predict2`) | LoRA, LoKr, full finetune | Cosmos Predict2 DiT + Qwen VAE + Qwen3/T5 (checkpoints often branded as Anima). |

Set `[model] type = "sdxl"` or `type = "cosmos_predict2"` and `[adapter] type = "lora"` or `type = "lokr"`. Omit `[adapter]` for full-model finetune. See [SDXL training](docs/user/training-sdxl-lora-lokr.md) or [Cosmos Predict2 / Anima](docs/user/training-cosmos-predict2-lora-lokr-finetune.md). Optional deps: `pip install -e ".[cosmos_predict2]"` for the latter.

## Installation

From the project root:

```bash
pip install -e .
```

For development you can use **uv**: `uv sync` then `uv run python -m renga_flow.main ...`.

Optional: LoRA/Lycoris-style adapters can use the `lycoris` extra: `pip install -e ".[lycoris]"`.

## Quick start

1. **Install** (see above).

2. **Prepare a config.** Copy an example and set your paths:

   ```bash
   cp examples/minimal_config_lora_sdxl.toml my_train.toml
   ```

   Edit `my_train.toml`: set `dataset` to your dataset TOML path, and under `[model]` set `checkpoint_path` to your SDXL checkpoint (e.g. a `.safetensors` file).

3. **Run training** with DeepSpeed:

   ```bash
   deepspeed --num_gpus=1 -m renga_flow.main --config my_train.toml
   ```

   For a single GPU you can use `--num_gpus=1`. Use more GPUs or a launcher (e.g. `torchrun`) for multi-GPU. Output and checkpoints go to `output_dir` (default `output`); each run gets a timestamped subfolder.

To only validate the config without training (e.g. if DeepSpeed is not set up), run:

```bash
python -m renga_flow.main --config my_train.toml
```

The script will load and validate the config then exit.

## Web UI (optional)

Local control panel for configs, jobs, logs, and signal files:

```bash
./start-ui.sh          # Linux / Git Bash — keep the terminal open
start-ui.bat           # Windows CMD — double-click or run from cmd
```

See [Web UI user guide](docs/user/web-ui.md). Training without the UI is unchanged (`pip install -e .` only).

## Documentation

- [User guide](docs/user/) — Training, config, and signal files for external control.
- [Developer guide](docs/developer/) — Extending the framework (APIs, adding models and signals). Includes [testing](docs/developer/testing.md) (how to run and extend the test suite).

See [docs/README.md](docs/README.md) for the full index.
