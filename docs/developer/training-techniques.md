# Shared training techniques (`rengu_flow/training/`)

Cross-model VRAM, speed, and quality helpers. Pipeline models only supply **which modules** to swap or model-specific loss inputs.

## Modules

| Module | Purpose | Config keys |
|--------|---------|-------------|
| [`block_swap.py`](../../rengu_flow/training/block_swap.py) | `BlockSwapOffloader` / `NoopOffloader` | `blocks_to_swap`, `disable_block_swap_for_eval`, `disable_block_swap_for_preview` |
| [`loss_weighting.py`](../../rengu_flow/training/loss_weighting.py) | min-SNR, debiased estimation | `model.min_snr_gamma`, `model.debiased_estimation_loss` (SDXL) |
| [`ema.py`](../../rengu_flow/training/ema.py) | CPU EMA shadow weights | `ema_decay` |
| [`optimizer_hooks.py`](../../rengu_flow/training/optimizer_hooks.py) | Fused optimizer validation | `optimizer.fused_backward`, `optimizer.fused_optimizer_groups` |
| [`quantized_load.py`](../../rengu_flow/training/quantized_load.py) | fp8 load dtype helpers | `model.transformer_dtype`, `model.diffusion_model_dtype` (Cosmos) |

## Block swap

- **Base:** [`BasePipeline.enable_block_swap`](../../rengu_flow/model/base.py) calls `get_block_swap_modules()`.
- **Cosmos:** `transformer.blocks` — [`TransformerLayer`](../../rengu_flow/model/cosmos_predict2/layers.py).
- **SDXL:** UNet `down_blocks` / `mid_block` / `up_blocks` via hook-based `HookBlockSwapOffloader`
  ([`SDXLPipeline.enable_block_swap`](../../rengu_flow/model/sdxl.py)) — works for adapters AND
  full-model (full-model additionally requires `optimizer.gradient_release`).
- **Preview (Cosmos):** `preview.preview_blocks_to_swap` uses the same `BlockSwapOffloader`.
- Requires `pipeline_stages = 1`. DeepSpeed places the model on the GPU; `main.py` then calls
  `prepare_block_swap_training()` after `deepspeed.initialize` to push swappable blocks to CPU.

## Cache TE dedup

`cache_dedup_text_embeddings = true` deduplicates text-encoder GPU work by caption SHA-256 during [`DatasetManager.cache`](../../rengu_flow/data/manager.py) (`_cache_fn` in [`manager.py`](../../rengu_flow/data/manager.py)). Opt-in; best for tag-heavy datasets where many files share captions.

## EMA

[`TrainingEMA`](../../rengu_flow/training/ema.py) is constructed when top-level `ema_decay` is set. [`main.py`](../../rengu_flow/main.py) calls `update()` after each successful `train_batch` (CPU shadow tensors). No automatic export yet.

## Forward / load dtype (Cosmos)

- **`model.diffusion_model_dtype`** — sets `rengu_flow.utils.common.AUTOCAST_DTYPE` for DiT forward (`main.py` after model load).
- **`model.transformer_dtype`** — bulk DiT checkpoint load dtype in [`CosmosPredict2Pipeline.load_diffusion_model`](../../rengu_flow/model/cosmos_predict2/pipeline.py).
- If `diffusion_model_dtype` is set and `transformer_dtype` is omitted, [`defaults.py`](../../rengu_flow/config/defaults.py) copies it to `transformer_dtype`.

## Fused optimizer hooks

[`optimizer_hooks.validate_fused_optimizer_config`](../../rengu_flow/training/optimizer_hooks.py) rejects `optimizer.fused_backward` / `optimizer.fused_optimizer_groups` when `gradient_accumulation_steps > 1`. Full Kohya-style fused backward is not wired yet; use existing **`optimizer.gradient_release`** for per-parameter steps on single-GPU pipeline runs.
