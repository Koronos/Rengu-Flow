# Training SDXL LoRA and LoKr (user guide)

This guide explains how to train SDXL with **LoRA** or **LoKr** (LyCORIS) adapters using Rengu Flow. No implementation details; task-oriented.

## What you need

- A TOML config file with `[model]` (type `sdxl`), `[optimizer]`, and `dataset`.
- An SDXL **base model** path in `model.checkpoint_path` — usually one large `.safetensors` file (not a LoRA).
- For LoKr, no extra install is needed: SDXL uses the built-in (vendored) LoKr backend, which
  integrates with the DeepSpeed pipeline (adapter weights live on each `nn.Linear`, so they are
  placed on the GPU and trained). The LyCORIS package is not required for SDXL LoKr.

## Config: LoRA

Add an `[adapter]` section with `type = "lora"` and a rank (or `dim`):

```toml
[adapter]
type = "lora"
rank = 16
# optional: dim = 16 (alias for rank, Kohya-style), alpha = 16, dropout = 0.0, dtype = "bfloat16"
```

- **rank**: LoRA rank (e.g. 8, 16, 32). Required unless you use `dim`.
- **dim**: Alias for `rank` (Kohya-style). You must set either `rank` or `dim`.
- **alpha**: Optional. Scaling factor; effective scale is `alpha / rank`. Default: same as rank (scale 1.0). Increase for stronger adapter effect, decrease for weaker.
- **dropout**: Optional; default 0.0.
- **dtype**: Optional; defaults to the model dtype.

Example minimal LoRA config: see `examples/minimal_config_lora_sdxl.toml`.

## Config: LoKr

Add an `[adapter]` section with `type = "lokr"` and a rank (or `dim`):

```toml
[adapter]
type = "lokr"
rank = 16
# optional: dim = 16 (alias for rank), alpha = 16, factor = -1, decompose_both = false, full_matrix = false, dtype = "bfloat16"
```

- **rank**: LoKr rank. Required unless you use `dim`.
- **dim**: Alias for `rank` (Kohya-style). You must set either `rank` or `dim`.
- **alpha**: Optional. Scaling factor; effective scale is `alpha / rank`. Default: same as rank (scale 1.0). Increase for stronger adapter effect, decrease for weaker.
- **factor**: Factorization hint; use -1 for automatic. Optional.
- **decompose_both**: Decompose both Kronecker factors. Optional; default false.
- **full_matrix**: Use full matrices instead of low-rank for the second factor. Optional; default false.
- **dtype**: Optional; defaults to the model dtype.

Example minimal LoKr config: see `examples/minimal_config_lokr_sdxl.toml`.

## Loss function (top-level config)

Optional keys in the **main** training TOML (not in `[adapter]`):

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **huber_delta** | Huber loss (PyTorch standard) | positive float | MSE if unset |
| **smooth_l1_beta** | Smooth L1 loss | positive float | MSE if unset |
| **pseudo_huber_c** | Legacy pseudo-Huber from diffusion-pipe configs | positive float | prefer `huber_delta` for new configs |

Only one of these should be set; if none are set, training uses MSE.

## Performance: torch.compile (top-level config)

Optional top-level keys (not in `[adapter]`) that apply `torch.compile` to the whole pipeline model (UNet):

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **compile** | Enable `pipeline_model.compile()` | `true` / `false` | `false` |
| **compile_mode** | Inductor mode → `torch.compile(mode=...)` | `"reduce-overhead"` (CUDA-graph based, lowest per-step overhead — best for fixed-shape steps, costs a little extra VRAM for graph pools), `"max-autotune"`, `"max-autotune-no-cudagraphs"` | `"default"` if unset |
| **compile_dynamic** | `torch.compile(dynamic=True)` for varying input shapes | `true` / `false` | `false` |

The first steps pay a one-time Inductor/CUDA-graph **warmup** and run slower while graphs build; steady-state steps afterward are faster. Worthwhile when the run is long enough to amortize the slow early steps — short smokes mix warmup into the average and are not representative. `reduce-overhead` can further reduce step time over the default mode on fixed-shape steps; benchmark it for your setup. See [Shared training techniques — torch.compile](../developer/training-techniques.md#torchcompile).

```toml
compile = true
# compile_mode = "reduce-overhead"   # optional: CUDA-graph mode, lowest per-step overhead
# compile_dynamic = true             # optional: only if input shapes vary between steps
```

## Saving adapters

- **By epoch**: Set `save_every_n_epochs = 1` (or another number) in your config. Adapters are written to the run directory as `epoch1`, `epoch2`, etc.
- **By step**: Set `save_every_n_steps` to save every N steps (e.g. `save_every_n_steps = 500`). Adapters are written as `step500`, `step1000`, etc.
- At least one of `save_every_n_epochs` or `save_every_n_steps` is recommended (defaults set `save_every_n_epochs = 1`).
- **Signal files**: `save` / `save_quit` for resume checkpoints; `export_model` / `export_model_quit` to export weights on demand; `preview` for TensorBoard sample images. See [Signal files](signal-files.md), [Checkpoints, model export, and retention](checkpoint-and-save.md), and [Training previews](previews.md).

## Where outputs go

- **Run directory**: Under `output_dir` (default `output`), e.g. `output/20250217_14-30-00`, or `output/my_experiment_20250217_14-30-00` if you set optional `run_name` in the training config.
- **Adapter saves**: Subfolders named `epoch<N>` or `step<N>` inside the run directory. Each contains:
  - **LoRA**: `lora.safetensors` (Kohya format).
  - **LoKr**: `adapter_model.safetensors` (LyCORIS/Comfy format).
- These files can be used in ComfyUI, Forge, or other inference UIs that support LoRA/LyCORIS.

## Real images and cache

For a folder of images (not synthetic data):

1. Point the main config at a [dataset TOML](dataset-config.md) with at least one `[[directory]]`.
2. Set **`model.cache_text_embeddings = true`** (default) so captions are encoded once during cache.
3. Run cache only, then training:

```bash
deepspeed --num_gpus=1 --module rengu_flow.main --config my.toml --cache_only
deepspeed --num_gpus=1 --module rengu_flow.main --config my.toml
```

Cache is stored under each directory’s `cache/sdxl/`. Use `--regenerate_cache` after changing images or captions; `--trust_cache` when nothing changed.

**Smoke example (12 CC0 images, 30 steps):** Copy `.env.example` → `.env` and set `RENGU_SDXL_CHECKPOINT_PATH`. Then `scripts/run_model_smoke.sh sdxl` (fixtures + cache + train; cleans `output/` and fixture caches afterward). Configs: `tests/fixtures/smoke/train_sdxl.toml` and `tests/fixtures/smoke/dataset_cc0.toml`.

## How to run training

1. Install: `pip install -e .` (or `pip install rengu-flow[lycoris]` for optional LyCORIS backend).
2. Run with DeepSpeed, e.g.:  
   `deepspeed --num_gpus=1 --module rengu_flow.main --config examples/minimal_config_lora_sdxl.toml`  
   For real data, use a config with `dataset = "..."` and no `synthetic_num_batches`.
3. Check the log for `Run dir: ...` to see where checkpoints and adapters are saved.

## Resuming and loading an existing adapter

- **Resume training**: Use `--resume_from_checkpoint` (or set `resume_from_checkpoint = true` in config) to resume from the latest run directory.
- **Start from an existing LoRA/LoKr file**: Set `init_from_existing` in the adapter section to the path of a directory containing a `.safetensors` file. The script will load those weights before training.
