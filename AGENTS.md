# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Claude Code reads `AGENTS.md` as an equivalent to `CLAUDE.md`. This project uses `AGENTS.md`.

## What this is

Rengu Flow is a **TOML-driven, registry-based** training framework for diffusion models (SDXL, Cosmos Predict2 — "Anima" checkpoints are Cosmos Predict2). It reimplements ideas from [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) in-repo (no runtime dependency on it). Training runs through **DeepSpeed**. An optional FastAPI + Vue web UI sits on top.

## Environment

- **Python managed by `uv`** (never pip/venv directly). The `./rengu` wrapper runs `uv sync --inexact` on first use, then execs `.venv/bin/rengu`.
- **Linux/WSL2 only** for actual training (DeepSpeed/CUDA). Do **all** work inside WSL — shell, git, `uv`, training. Editing files over `\\wsl.localhost\...` from a Windows editor is fine, but don't *run* git/python from the Windows side, and create git worktrees from inside WSL (a Windows-created worktree gets a UNC gitdir that WSL git can't open). Measure VRAM by torch `cuda_peak`, not `nvidia-smi`. Full pitfalls (incl. agents wrapping `wsl bash -lc`): `docs/developer/wsl-windows-workflow.md`; also `[[rengu-flow-test-workflow]]` memory.
- Tested stack (May 2026): Python 3.13, torch 2.12.0+cu130, deepspeed 0.19.0. torch/torchvision are pinned and come from the `pytorch-cu130` uv index — don't bump them casually (DeepSpeed 0.19 needs torch ≥ 2.11 to build its CUDA ops; see the comment in `pyproject.toml`).

## Commands

```bash
./rengu init                 # rengu.local.toml + uv sync (base deps)
./rengu init ui              # + web UI extras   (also: cosmos, lycoris, optim, dev)
./rengu train --config x.toml
./rengu validate --config x.toml      # config validation only, no GPU
./rengu ui start             # launch web UI
deepspeed --num_gpus=1 -m rengu_flow.main --config x.toml   # train without the wrapper

# Tests (run inside WSL). No GPU needed — CPU + mocks, tiny batches.
# pytest is in the `dev` extra, so pass --extra dev (add --extra ui/cosmos_predict2/lycoris for those suites).
uv run --extra dev pytest
uv run --extra dev pytest tests/test_dataset_config.py            # single file
uv run --extra dev pytest tests/test_install.py::test_name        # single test
uv run --extra dev pytest -k augmentation -x

# GPU smoke tests (optional, local, needs real checkpoints via .env)
scripts/run_model_smoke.sh sdxl|sdxl_lokr|cosmos|cosmos_lokr
```

There is **no linter/formatter/typechecker configured** for the Python side. Don't invent a `ruff`/`black` step. The Vue frontend in `ui/web/` uses `npm` (`npm run build`, `npm run dev`, `npm run test` via Vitest, `vue-tsc` for types) — separate from Python.

## Architecture

**Single resolution path** for every pluggable component: read `type` from the config section → look it up in a registry by string name → construct with that section's kwargs. Registration is **explicit** (decorators/registries in `rengu_flow/registry/`), never filesystem auto-discovery. To add a model/optimizer/scheduler, register it; don't add `import`-time magic.

Fixed execution phases (`rengu_flow/main.py`): config+dataset load → validate → distributed/DeepSpeed init → resolve components → pre-train (`DatasetManager.cache()`, load weights, configure adapter) → build pipeline from `model.to_layers()` → train loop (`train_batch` → logging, eval, `Saver`, signal files, previews) → shutdown. Adapters have **no registry yet** — they're branches in the pipeline + `rengu_flow/networks/*`.

Core packages under `rengu_flow/`: `cli/` (subcommand dispatch), `data/` (dataset/loader/cache + `DatasetManager`), `model/` (sdxl, cosmos_predict2 pipelines), `networks/` (LoRA/LoKr), `optim/`, `registry/`, `training/` (loop, EMA, block-swap, OOM-skip), `install/` (profile-based installer), `platform_compat.py`.

### Non-obvious boundaries — respect these

- **Core training/CLI must NOT depend on the UI's `jobs.db`.** `rengu_flow_ui/` (FastAPI + SQLite job registry + Vue) is a decoupled control plane that talks to training only via FastAPI endpoints and **signal files** (`save`, `save_quit`, `export_model`, `preview`, etc. — files dropped in the run output dir, checked each iteration). The cache `meta.db` (cache_v2 sqlite index) is a *separate* local thing that core training legitimately owns and is thread-safe. See `[[rengu-flow-db-boundary]]` memory.
- **WSL CUDA allocator:** never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it crashes under WSL2/WDDM. `rengu_flow.platform_compat.configure_cuda_allocator` neutralizes it automatically; don't undo that. See `[[rengu-flow-gpu-training-8gb]]` memory.
- **Config is the contract:** training parameters live in TOML, not hard-coded defaults. Model checkpoint paths go in the **training TOML**; `rengu.local.toml` (git-ignored) is only for UI host/port and launcher defaults.

### Test conventions

- The `@pytest.mark.no_ui_db` marker skips the autouse UI-sqlite fixture — use it for tests that don't touch the UI store (config/training-only).
- Prefer `@pytest.mark.parametrize` over many near-duplicate tests. Shared fixtures in `tests/conftest.py`.

## Docs

`docs/developer/` (architecture, testing, adding-optimizers-and-schedulers, networks, signal-files, vram-optimization, wsl-windows-workflow) and `docs/user/`. `docs/BACKLOG.md` tracks planned/deferred work (e.g. adapter registry, callback/hook registries). Follow `docs/developer/documentation-conventions.md` when editing docs.
