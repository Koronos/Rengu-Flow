<p align="center">
  <img src="ui/web/public/icon.svg" alt="Rengu Flow" width="132" height="132" />
</p>

<h1 align="center">Rengu Flow</h1>

<p align="center">
  <a href="LICENSE"><img alt="License: GPL-3.0-or-later" src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg"></a>
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-3776ab.svg?logo=python&logoColor=white">
  <img alt="PyTorch 2.12 (CUDA 13)" src="https://img.shields.io/badge/PyTorch-2.12%20%C2%B7%20CUDA%2013-ee4c2c.svg?logo=pytorch&logoColor=white">
  <img alt="DeepSpeed 0.19" src="https://img.shields.io/badge/DeepSpeed-0.19-1a73e8.svg">
  <img alt="Packaged with uv" src="https://img.shields.io/badge/packaging-uv-de5fe9.svg">
  <img alt="Status: preliminary v0.4.x" src="https://img.shields.io/badge/status-preliminary%20v0.4.x-f59e0b.svg">
</p>

> **Preliminary release (v0.4.x)** — Rengu Flow is under active development. APIs, config keys, CLI commands, and documentation may change in breaking or non-breaking ways between releases. Pin versions and re-read the docs when upgrading.

A **TOML-driven training framework** for diffusion models. You describe a run in a config file — model, adapter, optimizer, LR scheduler, dataset, and training options — and Rengu Flow launches it with [DeepSpeed](https://www.deepspeed.ai/). Everything is modular and registry-based: models, adapters, optimizers, and schedulers are selected and configured entirely from TOML, with an optional local web UI on top.

## Why Rengu Flow

- **Config-first** — One main TOML points to a dataset TOML and sets model, adapter, optimizer, scheduler, and training options. No code changes to start a run.
- **Modular & registry-based** — Models, adapters, optimizers, and schedulers are pluggable; extend the framework by registering new ones (see [architecture](docs/developer/architecture.md)).
- **VRAM-aware** — Block swap (CPU offload of transformer blocks), activation checkpointing and offload, optional quantization, gradient release, and OOM-skip let large models train on a single consumer GPU.
- **Self-managing dependencies** — The `rengu` CLI drives [uv](https://docs.astral.sh/uv/): it creates the venv, installs Python, and pulls only the optional extras a given config needs (e.g. Cosmos deps when `[model] type = "cosmos_predict2"`).
- **External control** — Drop signal files in the run directory (`save`, `save_quit`, `export_model`, `preview_now`, …) to checkpoint, export, preview, or exit cleanly — no API or restart required.
- **Local web UI** — Optional control panel to build configs and datasets, launch and queue runs, send signals, and watch live progress and previews.

## Features

- **Adapters & full finetune** — LoRA, LoKr (vendored, ComfyUI/Forge-compatible saves), the full **LyCORIS** algorithm family, and adapter-free full-model finetuning.
- **Datasets** — Directory datasets with aspect-ratio buckets, multi-resolution [resolution schedules](docs/user/dataset-config.md), a disk **cache (v2)** for latents and text embeddings, tag dropout / caption variants, and opt-in [augmentation presets](docs/user/dataset-augmentation.md).
- **Dataset Studio (`rengu prep`)** — Auto-tagging, captioning, watermark cleanup, and a bulk tag editor for preparing training data ([guide](docs/user/dataset-prep.md)).
- **Training loop** — Periodic [evaluation](docs/user/training-loop-and-eval.md) on held-out datasets, [image previews](docs/user/previews.md) during training, min-SNR / debiased loss weighting, EMA, and `torch.compile`.
- **Checkpointing & export** — Resume checkpoints vs. inference exports, retention limits, scheduled saves, and optional async export ([guide](docs/user/checkpoint-and-save.md)).
- **Experiment tracking** — A single sink fans out to a local manifest, TensorBoard, and (opt-in) Weights & Biases.
- **Optimizers & schedulers** — Built-in names, fully-qualified import paths, and vendored/extended optimizers (e.g. Prodigy, Automagic, K-Optimizers) selected from TOML.

## Supported models and adapters

| Model | Type | Adapters | Notes |
|-------|------|----------|-------|
| **Stable Diffusion XL** | `sdxl` | LoRA, LoKr, LyCORIS, full finetune | Optional UNet-only via `freeze_text_encoders`. |
| **Cosmos Predict2 / Anima** | `cosmos_predict2` (alias `anima`) | LoRA, LoKr, LyCORIS, full finetune | DiT + Wan VAE + Qwen3/T5. **Anima** checkpoints are this architecture; `type = "anima"` is accepted as a legacy alias. Needs the `cosmos` extra. |

**Adapter selection.** Set `[adapter] type = "lora"` / `"lokr"` / a `lycoris_*` type; omit `[adapter]` entirely for full-model finetune. The LyCORIS family (requires the `lycoris` extra) covers: `lycoris_locon`, `lycoris_loha`, `lycoris_lokr`, `lycoris_dylora`, `lycoris_glora`, `lycoris_diag_oft`, `lycoris_boft` (DoRA is the `dora_wd` toggle on locon/loha/lokr, not a separate type). See [SDXL training](docs/user/training-sdxl-lora-lokr.md), [Cosmos Predict2 / Anima](docs/user/training-cosmos-predict2-lora-lokr-finetune.md), and [full-model finetuning](docs/user/full-model-training-sdxl.md).

## Requirements

**You install these (system level):**

- **OS** — **Linux/WSL2** (multi-GPU, the `deepspeed` engine) **or native Windows** (single-GPU, the `accelerate` engine — no DeepSpeed required). See the [native Windows install guide](docs/user/windows-install.md) and the [WSL workflow](docs/developer/wsl-windows-workflow.md).
- **NVIDIA GPU + driver** — a CUDA-capable GPU with a driver recent enough for **CUDA 13.x** (check the "CUDA Version" reported by `nvidia-smi`). Download: **[NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)**.
- **CUDA Toolkit 13.x** — provides **`nvcc`**, which **DeepSpeed** uses to JIT-compile its C++/CUDA ops on the `deepspeed` engine (Linux/WSL). Its major version must match the PyTorch build (CUDA 13). The native-Windows `accelerate` engine does **not** use DeepSpeed, so `nvcc` is optional there (only `torch.compile` benefits from a toolchain). Point `CUDA_HOME` at the toolkit if it is not auto-detected. Download: **[CUDA Toolkit 13.x](https://developer.nvidia.com/cuda-downloads)**.
- **uv** — required on `PATH` for `./rengu` (and `rengu.cmd` on Windows). uv creates `.venv` and installs a compatible **Python (3.10–3.13)** automatically; no separate system `python3` needed. Install: **[uv](https://docs.astral.sh/uv/getting-started/installation/)**.
- **(native Windows, optional) Visual Studio Build Tools** — only needed if you use `torch.compile`. Download: **[VS Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)** (select "Desktop development with C++").
- **(web UI, optional) Node.js + pnpm** — only needed to **build** the web UI frontend (`rengu ui` / `rengu ui build`); the API itself is pure Python. The frontend uses **Vite 8**, which requires **Node.js ≥ 20.19 or ≥ 22.12** — the **22 LTS** line is recommended (older Node, e.g. 18, fails the build). Builds prefer **[pnpm](https://pnpm.io/installation)** (npm is used as a fallback). With **[Volta](https://volta.sh/)**, set it per project: `volta pin node@22 pnpm`. Download: **[Node.js](https://nodejs.org/)**.

**Installed automatically (by `rengu init` / `uv sync`):**

- **PyTorch 2.12 + CUDA 13.0** (`torch==2.12.0+cu130`), torchvision 0.27, and DeepSpeed 0.19, from the [PyTorch `cu130` index](https://download.pytorch.org/whl/cu130).
- The **CUDA 13 runtime stack — cuDNN, cuBLAS, NCCL, cuFFT, cuRAND, …** — ships *inside* those PyTorch wheels (as `nvidia-*-cu13` packages). You do **not** install cuDNN or the runtime libraries separately.
- All remaining Python deps (diffusers, PEFT, safetensors, …). Exact versions are pinned in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

> **Tested stack** (May 2026, WSL2 + NVIDIA): Python 3.13, torch 2.12.0+cu130, torchvision 0.27.0+cu130, deepspeed 0.19.0 — verified end-to-end on an 8 GB RTX 3000 Ada (SDXL + Cosmos LoRA and SDXL full-finetune smokes).

## Installation

From the repository root (Linux):

```bash
./rengu init          # create rengu.local.toml + uv sync (base training stack)
./rengu init ui       # also install the web UI extra
```

The `./rengu` wrapper runs `uv sync` on first use, so the venv is built automatically. Install optional extras by listing **profiles**:

```bash
./rengu init cosmos lycoris   # Cosmos Predict2 + LyCORIS adapters
./rengu init all              # every documented extra
```

| Profile | Installs |
|---------|----------|
| `base` | Core training stack (default) |
| `ui` | Local web control panel |
| `cosmos` / `cosmos_predict2` | Cosmos Predict2 / Anima |
| `lycoris` | LyCORIS adapter family (incl. LoKr backend) |
| `optim` | Extended optimizers |
| `kaon` | K-Optimizers (git-pinned: Adakaon, AdaMuon, KProdigy, …) |
| `prep` | Dataset Studio (taggers, captioners, watermark cleanup) |
| `dev` | Test/dev tools |
| `all` | All of the above |

`./rengu init --only-config` writes `rengu.local.toml` and directories without syncing. Advanced users can run `uv sync` themselves and call `.venv/bin/rengu` directly. Before any `train`/`validate`/`cache`, Rengu Flow inspects the config and auto-installs any missing extras it needs.

## Updating

```bash
./rengu update          # stable channel: fast-forward main, re-sync, rebuild UI if present
./rengu update --beta   # beta channel: switch to the develop branch and update it
```

`rengu update` pulls the latest project code, re-syncs dependencies from the lockfile, and recompiles the web UI if it was built locally. It also refreshes the optional profiles you already installed (so git-pinned extras like `kaon` move to their new commit pin); profiles you never installed are left alone.

**Release channels.** `rengu update` tracks `main` (stable); `rengu update --beta` switches the checkout to the `develop` branch (beta) and updates it. Switching back is just `rengu update`. While on the beta channel, `rengu version` and the web UI show a **beta** marker — the channel is derived from the git branch, not the version number. Useful flags: `--all-extras` (every documented extra), `--no-pull` (skip the git pull), and `--force` (discard local *tracked* code changes and hard-reset when a fast-forward is blocked — never touches untracked/ignored files, so your UI data dir and `jobs.db` are safe). Check your version with `./rengu version`.

## Quick start

1. **Set up** the environment:

   ```bash
   ./rengu init
   ```

2. **Local settings (optional).** `rengu init` creates `rengu.local.toml` (gitignored) for machine settings — UI host/port, default GPU count, master port, and subprocess env vars. Model checkpoint paths go in the **training TOML**, not here. See [`rengu.local.toml.example`](rengu.local.toml.example).

3. **Create a training config** from an example:

   ```bash
   cp examples/minimal_config_lora_sdxl.toml my_train.toml
   ```

   Edit `my_train.toml`: set the `dataset` path and `[model]` paths (e.g. `checkpoint_path` for SDXL).

4. **Train:**

   ```bash
   ./rengu train --config my_train.toml
   ```

   - Validate without training: `./rengu validate --config my_train.toml`
   - Build the dataset cache only: `./rengu cache --config my_train.toml`
   - Resume from the latest checkpoint: `./rengu train --config my_train.toml --resume-from-checkpoint`
   - Run DeepSpeed directly: `deepspeed --num_gpus=1 -m rengu_flow.main --config my_train.toml`

See the [CLI guide](docs/user/cli.md) for every command, flag, and `rengu.local.toml` key.

## Command-line interface

| Command | Description |
|---------|-------------|
| `rengu init [profiles…]` | Create `rengu.local.toml` + UI data dir, `uv sync` the chosen profiles |
| `rengu update [profiles…]` | Pull, re-sync from `uv.lock`, refresh installed extras, rebuild UI |
| `rengu version` | Print Rengu Flow version, git commit, and installed kaon version |
| `rengu train --config PATH` | Launch a DeepSpeed training run (`--num-gpus`, `--master-port`, `--resume-from-checkpoint`) |
| `rengu validate --config PATH` | Validate a training config and exit |
| `rengu cache --config PATH` | Build the dataset cache and exit |
| `rengu dump-dataset PATH` | Inspect a dataset TOML |
| `rengu prep <tag\|caption\|clean\|models>` | Dataset Studio: tagging, captioning, watermark cleanup, model list/download |
| `rengu ui [start\|serve\|dev\|build\|reset-db]` | Run or build the local web UI |

Trailing args after `--` are forwarded to the trainer (e.g. `./rengu train --config my.toml -- --regenerate_cache`). Full flag reference: [CLI guide](docs/user/cli.md).

## Web UI

```bash
./rengu ui start
```

Builds the frontend if needed, serves the API, and opens a browser. The UI lets you edit training configs and datasets, launch and queue runs, send [signal files](docs/user/signal-files.md), and watch live progress and previews. Live progress is parsed from the job's stdout — no extra config required. See the [Web UI user guide](docs/user/web-ui.md).

## Controlling a run

Training reacts to **signal files** dropped in the run directory (also exposed as buttons in the UI):

| Signal | Effect |
|--------|--------|
| `save` / `save_quit` | Checkpoint now (and exit). |
| `export_model` / `export_model_quit` | Export inference weights now (and exit). |
| `preview_now` | Render the configured preview prompts on the next step. |
| `continue` / `quit` | Resume after an export-wait pause / exit without saving. |

Details and the full list: [signal files](docs/user/signal-files.md).

## Documentation

- **[User guide](docs/user/)** — Training each model, dataset config and prep, optimizers, previews, checkpoints, signal files, the CLI, and the web UI. Start at the [docs index](docs/README.md).
- **[Developer guide](docs/developer/)** — [Architecture](docs/developer/architecture.md), the [model pipeline contract](docs/developer/model-pipeline-contract.md), [adding optimizers/schedulers](docs/developer/adding-optimizers-and-schedulers.md), [networks/adapters](docs/developer/networks.md), [VRAM optimization](docs/developer/vram-optimization.md), and [testing](docs/developer/testing.md).
- **[Implementation backlog](docs/BACKLOG.md)** — Planned / deferred work.

## Status and known limitations

Rengu Flow is a **preliminary release** under active development; treat config keys and CLI flags as subject to change between versions. A few specific notes:

- **Native Windows is single-GPU only.** It runs on the `accelerate` engine (plain PyTorch, no DeepSpeed). **Multi-GPU**, **`optimizer.gradient_release`**, **`blocks_to_swap`**, and **`pipeline_stages > 1`** require the `deepspeed` engine and are **Linux/WSL2 only** (they patch DeepSpeed's pipeline engine; Windows has no NCCL). Everything else — caching, gradient accumulation, per-resolution micro-batches, resolution schedules, `activation_offload`, `activation_checkpointing`, `torch.compile`, EMA, eval/previews — works on both. See the [native Windows install guide](docs/user/windows-install.md). On WSL2, do not set `PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"` — Rengu Flow detects WSL and applies safe defaults automatically (see the [CLI guide](docs/user/cli.md#training-environment-trainingenv)).
- **Built-in models** are SDXL and Cosmos Predict2 / Anima. Other architectures (e.g. Flux) are not yet registered — see [backlog](docs/BACKLOG.md).
- **Cosmos `load_and_fuse_adapter` is intentionally unsupported**; load adapter weights instead.

## Third-party components

Rengu Flow incorporates and adapts work from several projects. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full notices and licenses:

- **[diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)** (GPL-3.0) — design and portions of the training flow; vendored optimizers under [`rengu_flow/vendor/diffusion_pipe_optimizers/`](rengu_flow/vendor/diffusion_pipe_optimizers/).
- **NVIDIA Cosmos Predict2** (Apache-2.0) — DiT and LLM-adapter modeling code.
- **Alibaba Wan VAE** — used by the Cosmos pipeline.
- **AI Toolkit / Ostris** (MIT) — the Automagic optimizer.

## License

Rengu Flow is distributed under the **GNU General Public License v3.0 or later** ([LICENSE](LICENSE)). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for incorporated components and their licenses.
