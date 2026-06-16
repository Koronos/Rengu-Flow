# Training previews (developer guide)

User-facing options: `docs/user/previews.md`.

## Module

**`rengu_flow.utils.preview`**

| Function | Role |
|----------|------|
| `get_preview_config(config)` | Returns `config["preview"]` dict or `{}`. |
| `previews_configured(config)` | True if prompts exist and `enabled` is not false. |
| `normalize_preview_prompts(preview_cfg)` | `(tag, prompt)` list for logging. |
| `should_run_previews(...)` | Schedule + `forced` from signal. |
| `run_previews(model, config, sink, step, ...)` | Distributed barriers; inference on rank 0 only. Images logged via the tracking `sink`. |
| `_run_sdxl_previews(...)` | Calls `StableDiffusionXLPipeline.__call__` with `output_type="pil"`. |
| `_run_cosmos_previews(...)` | `prepare_preview_memory` → `CosmosPredict2Pipeline.generate_preview_image` → TensorBoard. |

## Cosmos Predict2

**`rengu_flow.model.cosmos_predict2.preview_sampling`**

- `build_timestep_schedule` — same `shift` / `flux_shift` as `prepare_inputs` in training.
- `encode_preview_prompt` — tokenize + `compute_text_embeddings` (T5 ids for `llm_adapter`).
- `euler_sample_latents` — rectified flow: `x_t = (1-t)·x0 + t·noise`, integrate `t: 1 → 0` with `x -= dt * v_pred`; optional CFG (`guidance_scale`, default `4.0` for Cosmos/Anima).
- `decode_latents_to_pil` — `WanVAE.decode`, frame `T=0`.

**VRAM (Tier A):** `prepare_preview_memory` / `restore_after_preview` on the pipeline — text encoder to CPU when `preview_offload_text_encoder` (default true), `transformer.eval()`, `empty_cuda_cache` + `cuda.synchronize` in `run_previews` finally.

**VRAM (Tier B):** `preview_blocks_to_swap` → shared `BlockSwapOffloader` in `rengu_flow/training/block_swap.py`; manual block loop in preview sampling. Training uses the same offloader when `blocks_to_swap` is set.

Requires **`pipeline_stages == 1`**; otherwise `run_previews` prints a skip message on the main process.

## Training loop integration

**`rengu_flow.main._run_training`**

1. Optional `preview_before_first_step` before the loop (step / x-axis `0`).
2. After each step, `saver.process_step` returns `(checkpointed, saved, signals)`.
3. If `should_run_previews(..., forced=signals.should_preview)`, call `run_previews`.

X-axis for `add_image` matches scalars: `examples` when `x_axis_examples`, else `step`.

## Block swap and optimizer

Same pattern as **`evaluate()`**: `prepare_block_swap_inference(disable_block_swap)`, `empty_cuda_cache`, optional `optimizer.eval()`, then restore training mode and `prepare_block_swap_training`.

- Eval uses top-level **`disable_block_swap_for_eval`** (passed from `main.py` into `evaluate()`).
- Previews use **`disable_block_swap_for_preview`** (passed into `run_previews()`). When omitted, behavior matches eval’s default in the preview module.

## Image logging (tracking sink)

Preview images are logged through the tracking `sink` (`rengu_track`), which fans out to the
configured backends (e.g. TensorBoard):

```python
sink.image(f"preview/{name}", chw_float_0_1, global_step)
```

CHW float tensor in `[0, 1]` from PIL via `_pil_to_chw_float`.

## Signals

**`rengu_flow.utils.signal_files`**: `SIGNAL_PREVIEW = "preview_now"` → `SignalResult.should_preview`.

Extend `SignalResult` and `broadcast_object_list` length when adding signals.

## Extending to other models

Add a branch in `run_previews` (or a `generate_preview` method on `ModelPipelineProtocol`) for new `model.type` values. SDXL uses the loaded diffusers pipeline on `model._pipeline`. Cosmos uses `generate_preview_image` on `CosmosPredict2Pipeline`.

## Tests

- `tests/test_preview.py` — config helpers, `preview` signal, Cosmos dispatch mock (no GPU).
- `tests/test_cosmos_preview_sampling.py` — schedule, Euler mock, memory offload, block offloader (CUDA optional).
