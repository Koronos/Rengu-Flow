# Training SDXL LoRA and LoKr (user guide)

This guide explains how to train SDXL with **LoRA**, **LoKr**, or any of the **LyCORIS** network adapters using Rengu Flow. No implementation details; task-oriented.

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
# optional: dim = 16 (alias for rank, Kohya-style), dropout = 0.0, dtype = "bfloat16"
```

- **rank**: LoRA rank (e.g. 8, 16, 32). Required unless you use `dim`.
- **dim**: Alias for `rank` (Kohya-style). You must set either `rank` or `dim`.
- **alpha**: Not configurable — rengu always sets `alpha = rank` (scale 1.0) so exports load at the intended strength; setting it in the TOML is rejected.
- **dropout**: Optional; default 0.0.
- **dtype**: Optional; defaults to the model dtype.

Example minimal LoRA config: see `examples/minimal_config_lora_sdxl.toml`.

## Config: LoKr

Add an `[adapter]` section with `type = "lokr"` and a rank (or `dim`):

```toml
[adapter]
type = "lokr"
rank = 16
# optional: dim = 16 (alias for rank), factor = -1, decompose_both = false, full_matrix = false, dtype = "bfloat16"
```

- **rank**: LoKr rank. Required unless you use `dim`.
- **dim**: Alias for `rank` (Kohya-style). You must set either `rank` or `dim`.
- **alpha**: Not configurable — rengu always sets `alpha = rank` (scale 1.0); setting it in the TOML is rejected.
- **factor**: Factorization hint; use -1 for automatic. Optional.
- **decompose_both**: Decompose both Kronecker factors. Optional; default false.
- **full_matrix**: Use full matrices instead of low-rank for the second factor. Optional; default false.
- **dtype**: Optional; defaults to the model dtype.

Example minimal LoKr config: see `examples/minimal_config_lokr_vendored.toml`.

## LyCORIS networks

The LyCORIS package exposes eight adapter algorithms beyond plain LoRA. All eight export a kohya-flat `adapter_model.safetensors` file (not `lora.safetensors` — that name is only produced by the built-in `lora` type) with `lora_unet_*` / `lora_te1_*` / `lora_te2_*` key prefixes — the format kohya-key-compatible loaders (ComfyUI and friends) expect.

When `adapter.type` starts with `lycoris_`, rengu installs the `lycoris` dependency profile automatically before training (the same profile the LoKr types use).

> **`lokr` vs `lycoris_lokr`:** both are Kronecker LoKr with the same export format. The built-in `lokr` is rengu's own lightweight implementation (`factor`/`decompose_both`/`full_matrix`); `lycoris_lokr` is the LyCORIS-library version with extra knobs (dropout, Tucker, `use_scalar`, DoRA via `dora_wd`, module targeting). On SDXL pick whichever fits — they train the same base. (On Cosmos there is one more difference: only the built-in `lokr` works on a quantized base — see that model's guide.)

### Practical recipe

```toml
[adapter]
type = "lycoris_loha"
rank = 16
# Output: adapter_model.safetensors (kohya-flat keys, ComfyUI-loadable)
```

Every type accepts `rank` (or its alias `dim`), `dtype`, `dropout`, `rank_dropout`, `module_dropout`, and `train_conv` (except DyLoRA, which does not support `train_conv`). As with LoRA/LoKr, `alpha` is not configurable: rengu sets `alpha = rank` automatically and rejects an explicit value.

### Type reference

| TOML `type` | Label | What it is | Extra fields (beyond common) |
|---|---|---|---|
| `lycoris_locon` | LyCORIS · LoCon | Standard LoRA math through the LyCORIS backend | `use_tucker` (default `false`), `use_scalar` (default `false`), `dora_wd` (default `false`), `wd_on_output` (default `true`) |
| `lycoris_loha` | LyCORIS · LoHa | Hadamard-product factorization of the weight delta | `use_tucker` (default `false`), `use_scalar` (default `false`), `dora_wd` (default `false`), `wd_on_output` (default `true`) |
| `lycoris_lokr` | LyCORIS · LoKr | Kronecker-product factorization | `use_tucker`, `use_scalar`, `dora_wd`, `wd_on_output`, `factor` (default `-1`, automatic), `full_matrix` (default `false`), `decompose_both` (default `false`), `unbalanced_factorization` (default `false`) |
| `lycoris_dylora` | LyCORIS · DyLoRA | Nested-rank training; the saved file can be truncated to any multiple of `block_size` after training | `block_size` (default `4`). `rank` must be divisible by `block_size`. `train_conv` is not supported, and the run needs `activation_checkpointing = false` (the random sub-rank per forward breaks checkpoint recompute). |
| `lycoris_glora` | LyCORIS · GLoRA | Adds input-side adaptation via `a1`/`a2` + `b1`/`b2` factor pairs | Dropout family only (`dropout`, `rank_dropout`, `module_dropout`) |
| `lycoris_diag_oft` | LyCORIS · Diag-OFT | Diagonal orthogonal rotation; `rank` sets the block split per layer instead of a low-rank dimension | `constraint` (default `0.0`), `rescaled` (default `false`). The exported `.alpha` stores the `constraint` value, not a rank. |
| `lycoris_boft` | LyCORIS · BOFT | Butterfly orthogonal rotation | `constraint` (default `0.0`), `rescaled` (default `false`). The exported `.alpha` stores the `constraint` value, not a rank. BOFT needs every adapted layer width to split as (even m ≤ rank) × power-of-two; SDXL widths carry a factor of 5, so `rank = 10` is the smallest that fits — smaller ranks fail at startup with "impossible to decompose". Its staged weight rebuild is also the most VRAM-hungry type: on a 16 GB card add `blocks_to_swap` (e.g. 8) or it OOMs even at 512px. |

### DoRA (weight decomposition)

DoRA is not a separate type — it is the `dora_wd` toggle (default `false`) available on
`lycoris_locon`, `lycoris_loha`, and `lycoris_lokr`. Turning it on splits each adapted
weight into a trained per-channel magnitude (exported as `dora_scale`) and the base
algorithm's delta as direction; it often tracks full finetuning better at low rank, at a
per-step compute cost. `wd_on_output` (default `true`) picks the magnitude axis (output
channel vs input column) and only matters when `dora_wd` is on. To train "plain LoRA +
DoRA", use `lycoris_locon` with `dora_wd = true`.

### Shared LyCORIS options

All `lycoris_*` types also accept:

- **`train_norm`** (default `false`): additionally trains the existing
  LayerNorm/GroupNorm weights, exported as `w_norm`/`b_norm` keys (ComfyUI loads
  them). Useful when global tone or contrast refuses to shift. SDXL only — the
  Cosmos DiT has no trainable norm weights, and the run fails fast if requested.
- **`rs_lora`** (default `false`, `lycoris_locon` only):
  rank-stabilized scaling `alpha / sqrt(rank)` instead of `alpha / rank`, keeping
  the update magnitude steady at high ranks (32+). The exported per-module alpha
  becomes `alpha * sqrt(rank)` so any loader reproduces the trained strength.
- **`target_include` / `target_exclude`** (default: all modules): glob patterns
  matched against each module's full dotted path; include is applied first, then
  exclude. The run fails at startup if nothing matches. Example — attention-only
  LoHa on the UNet:

```toml
[adapter]
type = "lycoris_loha"
rank = 16
target_include = ["unet.*attn*"]
```

### Key format and compatibility

Exported files use kohya-flat key prefixes:

- UNet modules: `lora_unet_<module_path>`
- Text encoder 1: `lora_te1_<module_path>`
- Text encoder 2: `lora_te2_<module_path>`

For all types except Diag-OFT and BOFT, the per-module `.alpha` tensor holds the rank value (enabling standard `alpha / rank` scaling in inference loaders). For Diag-OFT and BOFT it holds the `constraint` value.

DyLoRA exports `lora_up.weight` / `lora_down.weight` (the same key family as LoCon), so a DyLoRA file loads as a regular LoRA at full rank in any LyCORIS-compatible loader. To use a truncated rank, slice the file in multiples of `block_size` before loading.

### Not exposed

Two LyCORIS algorithms are not available:

- **`full`** — the upstream `FullModule.apply_to` deletes the weight that its own `org_forward` still needs. Rengu also has native full fine-tuning via its own mechanism.
- **`ia3`** — registered in the LyCORIS config SDK but absent from the `network_module_dict` wrapper registry, so it cannot be attached.

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
| **compile_mode** | Inductor mode → `torch.compile(mode=...)` | unset → `"default"` (the validated choice); `"max-autotune-no-cudagraphs"` (much longer per-shape warmup, marginal gain). The CUDA-graph modes `"reduce-overhead"` and `"max-autotune"` **crash on the first step** under DeepSpeed's per-layer compile — do not use them. | `"default"` if unset |
| **compile_dynamic** | `torch.compile(dynamic=True)` for varying input shapes | `true` / `false` | `false` |

The first steps pay a one-time Inductor **warmup** and run slower while graphs build; steady-state steps afterward are faster. Worthwhile when the run is long enough to amortize the slow early steps — short smokes mix warmup into the average and are not representative. Leave `compile_mode` unset (`"default"`); the CUDA-graph modes are incompatible with the pipeline compile (see below). See [Shared training techniques — torch.compile](../developer/training-techniques.md#torchcompile).

```toml
compile = true
# compile_dynamic = true             # optional: only if input shapes vary between steps
```

## Saving adapters

- **By epoch**: Set `save_every_n_epochs = 1` (or another number) in your config. Adapters are written to the run directory as `epoch1`, `epoch2`, etc.
- **By step**: Set `save_every_n_steps` to save every N steps (e.g. `save_every_n_steps = 500`). Adapters are written as `step500`, `step1000`, etc.
- At least one of `save_every_n_epochs` or `save_every_n_steps` is recommended (defaults set `save_every_n_epochs = 1`).
- **Signal files**: `save` / `save_quit` for resume checkpoints; `export_model` / `export_model_quit` to export weights on demand; `preview` for TensorBoard sample images. See [Signal files](signal-files.md), [Checkpoints, model export, and retention](checkpoint-and-save.md), and [Training previews](previews.md).

## Where outputs go

- **Run directory**: Under `output_dir` (default `output`), e.g. `output/20250217_14-30-00`, or `output/20250217_14-30-00_my_experiment` if you set optional `run_name` in the training config.
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
rengu cache --config my.toml
rengu train --config my.toml
```

Cache is stored under each directory’s `cache/sdxl/`. Use `--regenerate_cache` after changing images or captions; `--trust_cache` when nothing changed.

**Smoke example (12 CC0 images, 30 steps):** Copy `.env.example` → `.env` and set `RENGU_SDXL_CHECKPOINT_PATH`. Then `scripts/run_model_smoke.sh sdxl` (fixtures + cache + train; cleans `output/` and fixture caches afterward). Configs: `tests/fixtures/smoke/train_sdxl.toml` and `tests/fixtures/smoke/dataset_cc0.toml`.

## How to run training

1. Install: `pip install -e .` (or `pip install rengu-flow[lycoris]` for optional LyCORIS backend).
2. Run training:  
   `rengu train --config examples/minimal_config_lora_sdxl.toml`  
   For real data, use a config with `dataset = "..."` and no `synthetic_num_batches`.
3. Check the log for `Run dir: ...` to see where checkpoints and adapters are saved.

## Resuming and loading an existing adapter

- **Resume training**: Use `--resume_from_checkpoint` (or set `resume_from_checkpoint = true` in config) to resume from the latest run directory.
- **Start from an existing LoRA/LoKr file**: Set `init_from_existing` in the adapter section to the path of a directory containing a `.safetensors` file. The script will load those weights before training.
