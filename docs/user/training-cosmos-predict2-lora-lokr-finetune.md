# Training Cosmos Predict2 (Anima checkpoints)

This guide covers **austere end-to-end training** for checkpoints marketed as **Anima** (branding only): the architecture is **Cosmos Predict2 DiT** with **Qwen image VAE** and **Qwen3 + T5** text conditioning (`llm_path` in config). In TOML always use:

- `type = "cosmos_predict2"`

Install the optional extra:

```bash
pip install -e ".[cosmos_predict2]"
```

## Model files (which `.safetensors` is which?)

Cosmos / Anima training needs **three separate weight files** on disk (plus optional extras). They are **not interchangeable** — each path must point at the right download.

| Config key | What you are pointing at | Typical example |
|------------|--------------------------|-----------------|
| **`transformer_path`** | **Main image model** — the large checkpoint you LoRA/finetune | `anima-preview.safetensors` |
| **`vae_path`** | **Image VAE** — converts pixels ↔ latents for the dataset cache | `qwen_image_vae.safetensors` |
| **`llm_path`** | **Text encoder (Qwen3)** — turns captions into conditioning | `qwen_3_06b_base.safetensors` or a folder |
| **`t5_path`** | **Text encoder (T5)** — older setups only; use **instead of** `llm_path`, not as a fourth duplicate | One `.safetensors` when your bundle is T5-based |
| **`llm_adapter_path`** | Optional small adapter on the text stack | Leave empty unless your pack includes it |

In the web UI these appear as **Main model**, **Image VAE**, and **Text encoder — Qwen3** (or T5). The TOML keys stay `transformer_path`, `vae_path`, etc.

### Precision (`dtype` vs optional overrides)

| Key | Scope | When to set |
|-----|--------|-------------|
| **`dtype`** | **Required.** VAE, text encoder (Qwen3/T5), adapters, and “stable” DiT layers (embedders, norms, 1D params). Bulk DiT blocks too if `transformer_dtype` is omitted. | Always — usually `bfloat16`. |
| **`transformer_dtype`** | Only how **`transformer_path`** weights are loaded into the DiT (most transformer blocks). Defaults to `dtype`. | Rarely — VRAM or load issues with the main checkpoint. |
| **`diffusion_model_dtype`** | DiT **forward** autocast dtype (via `cuda_autocast`). If set, also defaults `transformer_dtype` when that is omitted. | Same names as `dtype` (e.g. `bfloat16`). | Omitted (uses `dtype`) |
| **`cache_text_embeddings`** | Caption cache during `--cache_only`. | Keep `true` (default). |

Tokenizer configs ship inside the package (`assets/qwen3_06b`, `assets/t5_old`) — you do not path those in TOML.

### Minimal `[model]` example

```toml
[model]
type = "cosmos_predict2"
dtype = "bfloat16"
transformer_path = "path/to/anima-preview.safetensors"
vae_path = "path/to/qwen_image_vae.safetensors"
llm_path = "path/to/qwen_3_06b_base.safetensors"
cache_text_embeddings = true
```

## Modes

### LoRA

```toml
[adapter]
type = "lora"
rank = 16
```

Example: `examples/minimal_config_cosmos_predict2_lora.toml`.

### LoKr

```toml
[adapter]
type = "lokr"
rank = 6
factor = -1
```

`alpha` is derived from `rank` (do not set `alpha` in TOML). Saves use Comfy-style keys: `diffusion_model.*` and per-module `.alpha`.

Example: `examples/minimal_config_cosmos_predict2_lokr.toml`.

#### Why are there two LoKr types? (`lokr` vs `lycoris_lokr`)

They share the same math (Kronecker factorization) and the same export format, so a
trained file from either loads the same way in ComfyUI. They differ in backend and
trade-offs:

| | `lokr` (built-in) | `lycoris_lokr` (LyCORIS library) |
|---|---|---|
| Backend | rengu's own implementation, params injected onto each `nn.Linear` | `lycoris-lora` `LokrModule` via the shared attach seam |
| Extra knobs | `factor`, `decompose_both`, `full_matrix` | all of those **plus** `dropout`/`rank_dropout`/`module_dropout`, `use_tucker`, `use_scalar`, `dora_wd` (DoRA on top), `unbalanced_factorization`, and `target_include`/`target_exclude` |
| Quantized base (`transformer_fp8_matmul` / `transformer_4bit`) | **Supported** — quantization-aware (routes the base matmul through the quantized `base_linear`, adds the Kronecker delta on top) | **Not supported** — the LyCORIS backend matches targets by exact class name `Linear`, so it skips the quantized linears (`Fp8MatmulLinear` / `Linear4bit`) entirely; config validation rejects the combination |

**Rule of thumb:** use the built-in **`lokr`** for the canonical/quantized-base
recipe; reach for **`lycoris_lokr`** (or another `lycoris_*` type) on an
unquantized base when you want the extra knobs (DoRA, dropout, Tucker, module
targeting).

#### A note on VRAM (why the LyCORIS types can need more than the built-in LoKr)

The built-in `lokr` adds a tiny Kronecker delta per layer and is the lightweight
baseline. Some `lycoris_*` types are heavier by design and need extra VRAM levers,
which is why their fixtures differ from the plain `lokr` one:

- **Diag-OFT / BOFT** rebuild the full weight matrix each step (orthogonal rotation),
  so they are the most memory-hungry — add `blocks_to_swap` on 16 GB cards.
- **DyLoRA** cannot use activation checkpointing (random sub-rank per forward), and
  the DiT's full activations do not fit 16 GB at 512px even with block swap — train
  it at a lower resolution (e.g. 256px) plus `blocks_to_swap`.
- The plain `lokr`/`lycoris_lokr`/`lycoris_loha`/`lycoris_locon` types keep the same
  light footprint as the original LoKr.

### LyCORIS networks

All seven LyCORIS algorithms are available for the DiT (same library backend as SDXL —
see the type reference in `training-sdxl-lora-lokr.md` for what each one is):
`lycoris_locon`, `lycoris_loha`, `lycoris_lokr`, `lycoris_dylora`,
`lycoris_glora`, `lycoris_diag_oft`, `lycoris_boft`. DoRA is the `dora_wd` toggle on
locon/loha/lokr, not a separate type.

```toml
[adapter]
type = "lycoris_loha"   # any of the seven types above
rank = 8
```

- Targets the same Linears as `lora`/`lokr` (every Linear inside the DiT blocks).
- `alpha` is derived from `rank` (do not set `alpha` in TOML). Saves use the same
  Comfy-style keys as cosmos LoKr: `diffusion_model.<module path>.<weight>` plus a
  per-module `.alpha`.
- Per-type extras match the SDXL table (`dora_wd`, LoKr's `factor`/`full_matrix`/…),
  including `rs_lora` (locon/dora) and `target_include`/`target_exclude` module
  globs (paths look like `blocks.0.self_attn.q_proj`).

Two types carry runtime constraints on the DiT (verified on the 2B / 2048-channel
Anima checkpoint):

- **`lycoris_dylora`** requires `activation_checkpointing = false` (its random
  sub-rank per forward breaks checkpoint recompute); pair with `blocks_to_swap` to
  recover the VRAM that disabling checkpointing costs.
- **`lycoris_diag_oft` / `lycoris_boft`** rebuild full weight matrices each step, so
  they are the most VRAM-hungry — add `blocks_to_swap` on 16 GB cards. BOFT also
  needs `rank` large enough to factorize every layer width: the DiT has widths with
  a factor of 5, so use `rank = 16` (smaller ranks fail at startup with "impossible
  to decompose").
- **`train_norm`** is *not* available here: the Cosmos DiT has no affine norm
  weights, and requesting it fails at startup.
- **Quantized base (`transformer_fp8_matmul` / `transformer_4bit`):** not supported
  with `lycoris_*`. The LyCORIS backend matches targets by exact class name
  (`Linear`), so it silently skips the quantized linears (`Fp8MatmulLinear` /
  `Linear4bit`) and would adapt only the unquantized minority — config validation
  rejects the combination. Only the built-in `lokr` is quantization-aware (it routes
  through the quantized `base_linear` and adds the Kronecker delta on top), so use
  `adapter.type = "lokr"` when training on a quantized base.
- DyLoRA and the OFT family are exposed for SDXL only: DyLoRA conflicts with
  `activation_checkpointing` (standard in cosmos configs), and Diag-OFT/BOFT's staged
  weight rebuild does not fit the DiT on 16 GB cards.

### Full finetune

Omit the `[adapter]` section. All DiT parameters with `requires_grad` are trained; use `save_model` export (not `adapter_model.safetensors`).

Example: `examples/minimal_config_cosmos_predict2_finetune.toml`.

## Learning rates

Optional per-block LRs in `[model]`:

- `self_attn_lr`, `cross_attn_lr`, `mlp_lr`, `mod_lr`, `llm_adapter_lr`

The LLM adapter submodule is **frozen by default** (`llm_adapter_lr = 0`) — it has outsized
influence on conditioning and degrades easily. Set `llm_adapter_lr` to a positive value only
if you intentionally want to train it.

## Dataset and cache

Point `dataset` at a TOML with `frame_buckets = [1]` for images (see `examples/minimal_cosmos_predict2_dataset.toml`). Run cache before training:

```bash
rengu train --config my.toml --cache_only
```

With `cache_text_embeddings = true` (default), text embeddings are cached once; VAE latents are cached per resolution bucket. Disk cache uses **`cache_format = "v2"`** by default (see [Training loop](training-loop-and-eval.md#pipeline-cache-and-debug-options)).

## Performance and VRAM (Anima / Cosmos)

Guidance for **real runs (typically ≥1000 steps)** on **LoKR** with ~16 GB VRAM (e.g. RTX 4080). Install deps with `pip install -e ".[cosmos_predict2]"` or `uv sync --extra cosmos_predict2` (see `pyproject.toml`).

Short tuning smokes (30 steps) are only **previews** for CI and quick regressions. They mix in `torch.compile` warmup and are **not** representative of per-step time on long jobs — ignore smoke averages for `compile`; judge steady-state iter time after warmup on your own run.

### Recommended for long training

| Setting | Recommendation |
|---------|----------------|
| **`cache_text_embeddings = true`** | Run `--cache_only` once; training should not re-encode captions every step. |
| **`activation_checkpointing = true`** | Required for typical VRAM on 16 GB; `false` caused **OOM** in tuning (~16 GB peak). |
| **`reentrant_activation_checkpointing = true`** | Default for `cosmos_predict2` when AC is on (`rengu_flow/config/defaults.py`); modest steady-state gain vs `false`. |
| **`compile = true`** | Enables **`pipeline_model.compile()`** — `torch.compile` on the whole pipeline model (diffusion-pipe parity). After Inductor warmup, steady steps were ~**0.51 s** vs ~**0.68–0.70 s** without compile on the same LoKR setup — worthwhile when the run is long enough to amortize slower early steps. Leave **`compile_mode`** unset (default mode is the validated one). ⚠️ Do **not** set `"reduce-overhead"` or `"max-autotune"`: both crash on the first step with a CUDAGraphs "output overwritten" error (torch 2.12 + DeepSpeed per-layer compile; measured on single-res and multi-res). `"max-autotune-no-cudagraphs"` runs but pays minutes of extra warmup per shape for marginal gain. **Multi-res / AR buckets need no extra flag**: the trainer enumerates the dataset's size buckets and compiles one static graph per shape, so every bucket runs at single-res compiled speed (leave `compile_dynamic` unset; see [Shared training techniques — torch.compile](../developer/training-techniques.md#torchcompile)). |
| **`blocks_to_swap`** | Offload DiT blocks (`transformer.blocks`) to CPU and stream them on demand when VRAM is tight (`pipeline_stages = 1`). Works for **both adapters and full finetune** (full finetune additionally requires `optimizer.gradient_release = true`). Start around half the block count and tune; on very small cards swap most of them. See [VRAM optimization](../developer/vram-optimization.md). |
| **`cache_dedup_text_embeddings = true`** | Speeds `--cache_only` when many images share the same caption (tag-heavy sets). |
| **`micro_batch_size_per_gpu`** | Set from VRAM; use **`gradient_accumulation_steps`** for effective batch without OOM. |

### Faster checkpointing — `"auto"` (compile-driven)

`activation_checkpointing` accepts more than `true`/`false`:

| Value | What it does | When |
|-------|--------------|------|
| `true` | **Full** checkpointing — recompute every block. Lowest VRAM. | **Default. Use on small/tight GPUs or without compile.** |
| `"auto"` | **Compiler-driven AC** — Inductor's memory-budget partitioner picks the optimal save/recompute split per compiled graph. Quality-neutral (exact recompute). Dial with `activation_memory_budget`. | **Best option whenever `compile = true`.** |

(The old `"selective"` (SAC) and `"unsloth"` modes were retired — `"auto"` measured faster AND lighter than SAC, and unsloth traded +2.6% step time for −0.5 GB. Legacy configs fall back to `true` with a warning; see `docs/EXPERIMENTS_GRAVEYARD.md`.)

**Measured @1024 LoKr, batch 1, compile=true (RTX 4080, steady state):**

| setting | iter time | vs full | peak VRAM |
|---|---|---|---|
| `true` (full) | 0.974 s | — | 5.76 GB |
| retired `"selective"` (SAC) | 0.932 s | −4.3% | 6.56 GB |
| `"auto"`, budget **0.1** | 0.881 s | **−9.5%** | **6.37 GB** (beat SAC on both axes) |
| `"auto"`, budget **0.3** (default) | 0.822 s | **−15.7%** | 8.99 GB |
| `"auto"`, budget **0.5** | 0.774 s | **−20.6%** | 11.32 GB (speed plateau — 0.8 gains nothing) |
| `false` | OOM | — | >15.5 GB |

`"auto"` requires `compile = true` (the partitioner lives in the compiled joint graph) and composes with the per-shape static compile — multi-res schedules get the same gains. **The budget is per-shape**: the configured `activation_memory_budget` applies to the *largest* bucket (the VRAM-binding one) and smaller buckets scale it up automatically toward 1.0 (little/no recompute) at unchanged peak — so small resolutions in a multi-res schedule run at full speed instead of paying the large bucket's recompute rate. Each shape's announce line shows the budget it compiled with. Per-step losses match `true` to ~1e-5 (normal bf16 kernel-order noise): no precision cost. SDXL benefits even more: LoKr @512 measured **0.311 s vs 0.486 s** with full checkpointing (−36%) at +0.1 GB.

**Full finetune (2B DiT @512, `adamw8bit` + `blocks_to_swap = 14` + `gradient_release`):** activations are not the binding constraint (optimizer states + grads are — plain finetune OOMs even with 8-bit Adam and full checkpointing), so block swap stays necessary; but `"auto"` composes on top: swap + `true` (no compile) **1.836 s / 6.81 GB** vs swap + `compile` + `"auto"` (budget 0.1) **1.560 s / 9.98 GB** (−15%). ⚠️ `true` + `compile` + block swap crashes with a `CheckpointError` (non-reentrant recompute metadata mismatch) — with block swap + compile use `"auto"` (it has no checkpoint wrapper), or drop `compile`.

> ⚠️ **Mind the budget on low-VRAM cards.** Higher `activation_memory_budget` keeps more activations resident; if you OOM, lower it (0.1 still beat SAC on both axes) or fall back to `activation_checkpointing = true`. At higher resolution (e.g. 1536) re-check that your budget still fits.

- **`activation_checkpoint_interval`** — checkpoint every N blocks (default `1`, only applies to `true`). Measured neutral on Cosmos; leave at `1`.

### Compile on-disk cache (static shapes only)

**`compile_disk_cache`** (default `"auto"`) persists `torch.compile`'s Inductor/Triton kernels to disk so a re-run skips recompilation. `"auto"` enables it **only when `compile_dynamic` is off** — because dynamic shapes (which multi-resolution + aspect-ratio bucketing require) never reproduce the cache key, so the cache is a no-op there. With static (fixed-shape) training it saves ~30 s of compile per run.

> By default the cache lives in **`<cache_root>/compile`** (next to your dataset caches, following a custom `cache_root`). It must be on an ext4-style filesystem (255-char filenames); on an **encrypted home** (~143-char limit) it auto-disables with a warning — point **`compile_cache_dir`** at an ext4 path. When compile is on, the trainer also prints a one-line heads-up that the first step compiles (and may take ~1–4 min) so a long first step doesn't look like a hang.

**What re-keys the cache (forces a full recompile)** — the cache key hashes the compile-relevant config and every tensor shape in the graph, so changing any of these makes previous entries unusable: `activation_checkpointing` mode (`true` ↔ `"auto"`), **`activation_memory_budget`** (any value change), `compile_mode`, adapter **rank/factor/full_matrix** (parameter shapes live inside the graphs), `micro_batch_size_per_gpu`, the resolution/AR bucket set (new shapes add entries; old ones stay valid), and torch upgrades. **Optimizer settings do NOT re-key it** (type/lr/betas/momentum_dtype — the optimizer step is not compiled). Note a disk-cache **hit** still costs a few seconds per shape on the first step (dynamo always re-traces in-process; the cache skips Inductor codegen/autotune — ~3-8 s vs ~30 s+ cold). At the end of each run the trainer prints `[compile-cache] fxgraph disk cache: X hits / Y misses` — all-misses on a config you ran before means something above changed.

### If it doesn't fit: the VRAM ladder

When a run OOMs, follow the model-agnostic **[VRAM ladder](training-loop-and-eval.md#if-it-doesnt-fit-the-vram-ladder)** in the shared training guide — ordered, composable steps from cheapest (text-embedding cache, checkpointing, budget) to heaviest (memory-efficient optimizer states — kaon — and block swap). Cosmos data points: full AC takes 1024 LoKr from OOM (>15.5 GB) to 5.76 GB; block swap + `gradient_release` + `compile` + `"auto"`(0.1) fits the 2B **full finetune** on 16 GB (1.56 s/step @512, 9.98 GB peak).

### Very low VRAM (≈8 GB)

The DiT is large, so on an 8 GB card lean on block swap of `transformer.blocks` plus the shared
low-VRAM stack. Measured on an 8 GB RTX 3000 Ada (WSL2), 3-step smoke at the CC0 dataset:

- **Cosmos LoRA + `blocks_to_swap` (most blocks)**: ~**1.6 GB** peak — block swap streams the frozen
  DiT, only the adapter + a couple of blocks stay resident.
- **Cosmos full finetune + block swap**: omit `[adapter]`, set `optimizer.gradient_release = true`
  (required — each block's optimizer step runs in the backward while it is resident), a frugal
  optimizer (Adafactor, or `genericoptim` with `cpu_offload`), `cache_text_embeddings = true`
  (keeps the Qwen3/T5 text encoder out of the training graph) and `activation_checkpointing = true`.

```toml
blocks_to_swap = 24            # of ~28 DiT blocks; keep a few resident. pipeline_stages = 1.
activation_checkpointing = true

[model]
type = "cosmos_predict2"
dtype = "bfloat16"
cache_text_embeddings = true
# transformer_dtype = "float8_e4m3fn"   # optional: load DiT blocks in fp8 to cut load/VRAM further

[optimizer]
type = "transformers.optimization.Adafactor"
lr = 1.0e-5
scale_parameter = false
relative_step = false
warmup_init = false
gradient_release = true        # required for full-finetune block swap
```

The same lever interactions and trade-offs (and the WSL2 allocator caveat) are documented once, model-agnostically, in [VRAM optimization](../developer/vram-optimization.md) — block swap there is described for SDXL's UNet but the mechanism (`HookBlockSwapOffloader`) and the recipe are identical for Cosmos's DiT.

### Do not use (Cosmos)

| Setting | Why |
|---------|-----|
| **`activation_checkpointing = false`** | OOM on ~16 GB adapter training. |

### Optional / low impact

| Setting | Notes |
|---------|--------|
| **`optimizer.type = 'adamw8bitkahan'`** | Needs bitsandbytes + CUDA on `LD_LIBRARY_PATH`; little benefit observed vs `adamw` on Anima LoKR. |
| **`optimizer.gradient_release = true`** | Only with `pipeline_stages = 1`. |
| **`genericoptim` + `compile`** | Slower than `adamw` + `compile` in previews — stick to `adamw` unless you need GenericOptim. |

### Example TOML (throughput-minded LoKR, long runs)

```toml
activation_checkpointing = true
reentrant_activation_checkpointing = true
compile = true
# compile_mode: leave unset — "reduce-overhead"/"max-autotune" crash (CUDAGraphs + DeepSpeed)
# compile_dynamic = true             # only for dozens of distinct shapes; slower steady state
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 1
```

## Validate config

```bash
python -m rengu_flow.main --config my.toml --validate-only
```

## Manual GPU smoke (not automated)

1. Install `.[cosmos_predict2]` and DeepSpeed with CUDA.
2. Copy `.env.example` → `.env` and set `RENGU_COSMOS_TRANSFORMER_PATH`, `RENGU_COSMOS_VAE_PATH`, `RENGU_COSMOS_LLM_PATH`.
3. `scripts/run_model_smoke.sh cosmos` — vendors `tests/fixtures/smoke_cc0/` if needed, then `--cache_only` and **30** training steps (`tests/fixtures/smoke/train_cosmos_predict2.toml`). The script removes `output/` and dataset caches after the run to save disk (`KEEP_SMOKE_ARTIFACTS=1` to keep them). For training **signal files** and **genericoptim resume**, use `scripts/smoke_training_signals.sh` (see [signal files](signal-files.md)).
4. Confirm `adapter_model.safetensors` under the run directory.

Optional: `[train.oom_skip]` for single-GPU OOM resilience — see [Training loop and eval](training-loop-and-eval.md) and `examples/config_oom_skip.toml`.

Out of scope for this austere path: **`load_and_fuse_adapter`** (use `load_adapter_weights` only), ComfyUI submodule. **Block swap** during training is supported for both adapters and full finetune (full finetune also needs `optimizer.gradient_release = true`), with `pipeline_stages = 1` — see [training loop](training-loop-and-eval.md#block-swap) and [VRAM optimization](../developer/vram-optimization.md). Dataset **augmentation MVP** is supported — see [dataset augmentation](dataset-augmentation.md).

**Training previews** are supported via `[preview]` and the `preview_now` signal file when `pipeline_stages = 1` — see [Training previews](previews.md). For **Anima**, a practical default is `num_inference_steps = 20`, `guidance_scale = 4`, `width`/`height = 512` on 16 GB GPUs.
