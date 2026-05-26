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
| **`diffusion_model_dtype`** | Intended for forward-pass math vs storage dtype. | **Leave unset** — parsed in config but **not used** by training yet. |
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

Set `llm_adapter_lr = 0` to freeze the LLM adapter submodule when present.

## Dataset and cache

Point `dataset` at a TOML with `frame_buckets = [1]` for images (see `examples/minimal_cosmos_predict2_dataset.toml`). Run cache before training:

```bash
deepspeed --num_gpus=1 -m renga_flow.main --config my.toml --cache_only
```

With `cache_text_embeddings = true` (default), text embeddings are cached once; VAE latents are cached per resolution bucket.

## Validate config

```bash
python -m renga_flow.main --config my.toml --validate-only
```

## Manual GPU smoke (not automated)

1. Install `.[cosmos_predict2]` and DeepSpeed with CUDA.
2. Set real paths for transformer, VAE, and LLM in an example TOML.
3. `--cache_only` on a small image folder.
4. Train 1 epoch with LoRA; confirm `adapter_model.safetensors` under the run directory.
5. Optional: repeat with LoKr and full finetune.

Out of scope for this austere path: block swap, training previews, augmentation presets, OOM skip, ComfyUI submodule.
