# Training Cosmos Predict2 (Anima checkpoints)

This guide covers **austere end-to-end training** for checkpoints marketed as **Anima**: the architecture is **Cosmos Predict2 DiT** with **Qwen image VAE** and **Qwen3 + T5** text conditioning (`llm_path` in config). In TOML you can use either:

- `type = "cosmos_predict2"` (canonical), or
- `type = "anima"` (alias — same pipeline when `llm_path` is set).

Install the optional extra:

```bash
pip install -e ".[cosmos_predict2]"
```

## Paths

Under `[model]` you need:

| Key | Purpose |
|-----|---------|
| `transformer_path` | DiT weights (e.g. Anima preview `.safetensors`) |
| `vae_path` | Qwen image VAE weights |
| `llm_path` | Qwen3 0.6B (directory or single `.safetensors`) |
| `t5_path` | Alternative: T5-only text stack (no Qwen3) |

Tokenizer configs ship inside the package (`assets/qwen3_06b`, `assets/t5_old`).

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
