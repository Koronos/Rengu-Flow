# Shared training techniques (`rengu_flow/training/`)

Cross-model VRAM, speed, and quality helpers. Pipeline models only supply **which modules** to swap or model-specific loss inputs.

## Modules

| Module | Purpose | Config keys |
|--------|---------|-------------|
| [`block_swap.py`](../../rengu_flow/training/block_swap.py) | `HookBlockSwapOffloader` (training) / `BlockSwapOffloader` (Cosmos preview) / `NoopOffloader` | `blocks_to_swap`, `disable_block_swap_for_eval`, `disable_block_swap_for_preview` |
| [`loss_weighting.py`](../../rengu_flow/training/loss_weighting.py) | min-SNR, debiased estimation | `model.min_snr_gamma`, `model.debiased_estimation_loss` (SDXL) |
| [`ema.py`](../../rengu_flow/training/ema.py) | CPU EMA shadow weights | `ema_decay` |
| [`optimizer_hooks.py`](../../rengu_flow/training/optimizer_hooks.py) | Fused optimizer validation | `optimizer.fused_backward`, `optimizer.fused_optimizer_groups` |
| [`quantized_load.py`](../../rengu_flow/training/quantized_load.py) | fp8 load dtype helpers | `model.transformer_dtype`, `model.diffusion_model_dtype` (Cosmos) |
| [`main.py`](../../rengu_flow/main.py) | `torch.compile` on the pipeline model | `compile`, `compile_mode`, `compile_dynamic` |

## Block swap

- **Base:** [`BasePipeline.enable_block_swap`](../../rengu_flow/model/base.py) builds the hook-based
  `HookBlockSwapOffloader` from `get_block_swap_modules()`; neither SDXL nor Cosmos overrides it, so
  both use this offloader for **training**.
- **Cosmos:** `transformer.blocks` — [`TransformerLayer`](../../rengu_flow/model/cosmos_predict2/layers.py)
  (its `wait_for_block` / `submit_move_blocks_forward` calls are no-ops under the hook offloader).
- **SDXL:** UNet `down_blocks` / `mid_block` / `up_blocks` — supplies `get_block_swap_modules()` /
  `_block_swap_root_modules()`; works for adapters AND full-model (full-model additionally requires
  `optimizer.gradient_release`).
- **Preview (Cosmos):** `preview.preview_blocks_to_swap` uses the layer-driven `BlockSwapOffloader`.
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

## torch.compile

Top-level `compile = true` makes [`main.py`](../../rengu_flow/main.py) call `pipeline_model.compile(**compile_kwargs)` — `torch.nn.Module.compile`, i.e. `torch.compile` applied to the whole DeepSpeed pipeline model (UNet/DiT), not an optimizer-step compile. Two optional keys shape the call:

| Key | Maps to | Values |
|-----|---------|--------|
| `compile_mode` | `torch.compile(mode=...)` | unset → `"default"` (the validated choice); `"max-autotune-no-cudagraphs"` (much longer per-shape warmup, marginal gain on AC-heavy steps). **CUDA-graph modes crash**: `"reduce-overhead"` and `"max-autotune"` fail on the first step (torch 2.12, Cosmos) with "accessing tensor output of CUDAGraphs that has been overwritten" — DeepSpeed's per-layer compile makes each layer's output a graph-owned tensor that pipeline buffers retain across replays. Measured on single-res AND multi-res (2026-06). |
| `compile_dynamic` | `torch.compile(dynamic=True)` | usually leave unset — see shape handling below. `true` trades per-shape warmup for slower generic kernels |

**Shape handling (multi-res / AR buckets).** When a real dataset is loaded, [`training/compile_plan.py`](../../rengu_flow/training/compile_plan.py) counts the distinct size buckets (every latent shape the model will see is enumerable up front) and plans the compile around them: it forces `dynamic=False` so each shape gets its own **static** graph — the same kernels a single-res run at that resolution would use — and raises `torch._dynamo`'s recompile budgets (`cache_size_limit`, default 8 per code object) to fit every shape, so bucket counts above 8 no longer silently fall back to eager. Without this, torch's automatic-dynamic-shapes converts the model to slower dynamic kernels on the second distinct shape. Each shape compiles once (first step on it); the Inductor disk cache persists all per-shape kernels, so later runs skip the warmup. Set `compile_dynamic = true` only if per-shape warmup is unacceptable and slower steady-state steps are fine.

**Compiler-driven activation checkpointing.** `activation_checkpointing = "auto"` ([`training/activation_budget.py`](../../rengu_flow/training/activation_budget.py)) drops the manual checkpoint wrappers entirely and sets `torch._functorch.config.activation_memory_budget` so Inductor's min-cut partitioner chooses the save/recompute split per compiled joint graph. Requires `compile = true`; tune with `activation_memory_budget` (0.0-1.0, default 0.3). Dominates SAC on both axes at low budgets (measured @1024 LoKr: budget 0.1 → −9.5% step time at less VRAM than SAC; 0.5 → −20.6%, plateau).

The first steps pay a one-time Inductor/CUDA-graph **warmup** (graphs build), then steady-state steps are faster — worthwhile when the run is long enough to amortize the slow early steps. Short smokes mix warmup into the mean and are not representative; judge steady-state iter time after warmup. Measured figures live in the [Cosmos LoKR user doc](../user/training-cosmos-predict2-lora-lokr-finetune.md) (steady steps ~0.51 s vs ~0.68–0.70 s without compile).

## Fused optimizer hooks

[`optimizer_hooks.validate_fused_optimizer_config`](../../rengu_flow/training/optimizer_hooks.py) rejects `optimizer.fused_backward` / `optimizer.fused_optimizer_groups` when `gradient_accumulation_steps > 1`. Full Kohya-style fused backward is not wired yet; use existing **`optimizer.gradient_release`** for per-parameter steps on single-GPU pipeline runs.
