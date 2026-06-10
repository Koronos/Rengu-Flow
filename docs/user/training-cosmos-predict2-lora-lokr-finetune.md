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
deepspeed --num_gpus=1 --module rengu_flow.main --config my.toml --cache_only
```

With `cache_text_embeddings = true` (default), text embeddings are cached once; VAE latents are cached per resolution bucket. Disk cache uses **`cache_format = "v2"`** by default (see [Training loop](training-loop-and-eval.md#deepspeed-pipeline-and-debug-options)).

## Performance and VRAM (Anima / Cosmos)

Guidance for **real runs (typically ≥1000 steps)** on **LoKR** with ~16 GB VRAM (e.g. RTX 4080). Install deps with `pip install -e ".[cosmos_predict2]"` or `uv sync --extra cosmos_predict2` (see `pyproject.toml`).

Short tuning smokes (30 steps) are only **previews** for CI and quick regressions. They mix in `torch.compile` warmup and are **not** representative of per-step time on long jobs — ignore smoke averages for `compile`; judge steady-state iter time after warmup on your own run.

### Recommended for long training

| Setting | Recommendation |
|---------|----------------|
| **`cache_text_embeddings = true`** | Run `--cache_only` once; training should not re-encode captions every step. |
| **`activation_checkpointing = true`** | Required for typical VRAM on 16 GB; `false` caused **OOM** in tuning (~16 GB peak). |
| **`reentrant_activation_checkpointing = true`** | Default for `cosmos_predict2` when AC is on (`rengu_flow/config/defaults.py`); modest steady-state gain vs `false`. |
| **`compile = true`** | Enables **`pipeline_model.compile()`** — `torch.compile` on the whole pipeline model (diffusion-pipe parity). After Inductor warmup, steady steps were ~**0.51 s** vs ~**0.68–0.70 s** without compile on the same LoKR setup — worthwhile when the run is long enough to amortize slower early steps. Optional: **`compile_mode = "reduce-overhead"`** (CUDA-graph based, lowest per-step overhead — best for fixed-shape steps; can further cut step time over the default mode, benchmark per setup; other modes: `"max-autotune"`, `"max-autotune-no-cudagraphs"`) and **`compile_dynamic = true`** (only if input shapes vary between steps). See [Shared training techniques — torch.compile](../developer/training-techniques.md#torchcompile). |
| **`blocks_to_swap`** | Offload DiT blocks (`transformer.blocks`) to CPU and stream them on demand when VRAM is tight (`pipeline_stages = 1`). Works for **both adapters and full finetune** (full finetune additionally requires `optimizer.gradient_release = true`). Start around half the block count and tune; on very small cards swap most of them. See [VRAM optimization](../developer/vram-optimization.md). |
| **`cache_dedup_text_embeddings = true`** | Speeds `--cache_only` when many images share the same caption (tag-heavy sets). |
| **`micro_batch_size_per_gpu`** | Set from VRAM; use **`gradient_accumulation_steps`** for effective batch without OOM. |

### Faster checkpointing (SAC) — for GPUs with VRAM headroom

`activation_checkpointing` accepts more than `true`/`false`:

| Value | What it does | When |
|-------|--------------|------|
| `true` | **Full** checkpointing — recompute every block. Lowest VRAM. | **Default. Use on small/tight GPUs.** |
| `"selective"` | **Selective Activation Checkpointing (SAC)** — *keep* the expensive attention activations, recompute only the cheaper ops. Quality-neutral (same math). | **16 GB+ with headroom**, to trim the recompute tax. |
| `"unsloth"` | Alternative checkpoint kernel for supported models. | If standard AC is tight. |

SAC is **opt-in** and **uses MORE VRAM than full checkpointing** (it stores what it would otherwise recompute). Measured on Anima/Cosmos LoKR at **1024px, batch 2**: full ckpt **1.82 s @ 7.6 GB** vs SAC **1.74 s @ 9.5 GB** — ~**4 %** faster at the resolution that dominates a multi-res schedule, still well within 16 GB. It works *with* `compile_dynamic = true` (orthogonal to compile).

> ⚠️ **Not for low-VRAM cards.** SAC needs free VRAM; on a small GPU it can OOM. The trainer prints a warning when it's enabled (extra-loud on <12 GB). If you OOM, set `activation_checkpointing = true` or shrink `selective_checkpoint_save_ops`. At higher resolution (e.g. 1536) re-check that it still fits before relying on it.

- **`selective_checkpoint_save_ops`** — the VRAM/speed dial for SAC. Comma-separated extra aten ops to keep (e.g. `"mm,addmm,bmm"`); empty keeps attention only (lightest). Adding matmuls gave *no* extra speed at 1024 (torch.compile's partitioner already handles them), so attention-only is the sweet spot.
- **`activation_checkpoint_interval`** — checkpoint every N blocks (default `1`). Measured neutral on Cosmos; leave at `1`.

### Compile on-disk cache (static shapes only)

**`compile_disk_cache`** (default `"auto"`) persists `torch.compile`'s Inductor/Triton kernels to disk so a re-run skips recompilation. `"auto"` enables it **only when `compile_dynamic` is off** — because dynamic shapes (which multi-resolution + aspect-ratio bucketing require) never reproduce the cache key, so the cache is a no-op there. With static (fixed-shape) training it saves ~30 s of compile per run.

> By default the cache lives in **`<cache_root>/compile`** (next to your dataset caches, following a custom `cache_root`). It must be on an ext4-style filesystem (255-char filenames); on an **encrypted home** (~143-char limit) it auto-disables with a warning — point **`compile_cache_dir`** at an ext4 path. When compile is on, the trainer also prints a one-line heads-up that the first step compiles (and may take ~1–4 min) so a long first step doesn't look like a hang.

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
| **`activation_checkpointing = "unsloth"`** | Alternative VRAM tradeoff if standard AC is tight. |
| **`optimizer.type = 'adamw8bitkahan'`** | Needs bitsandbytes + CUDA on `LD_LIBRARY_PATH`; little benefit observed vs `adamw` on Anima LoKR. |
| **`optimizer.gradient_release = true`** | Only with `pipeline_stages = 1`. |
| **`genericoptim` + `compile`** | Slower than `adamw` + `compile` in previews — stick to `adamw` unless you need GenericOptim. |

### Example TOML (throughput-minded LoKR, long runs)

```toml
activation_checkpointing = true
reentrant_activation_checkpointing = true
compile = true
# compile_mode = "reduce-overhead"   # optional: CUDA-graph mode, lowest per-step overhead
# compile_dynamic = true             # optional: only if input shapes vary between steps
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

**Training previews** are supported via `[preview]` and the `preview` signal file when `pipeline_stages = 1` — see [Training previews](previews.md). For **Anima**, a practical default is `num_inference_steps = 20`, `guidance_scale = 4`, `width`/`height = 512` on 16 GB GPUs.
