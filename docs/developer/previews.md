# Training previews (developer guide)

User-facing options: `docs/user/previews.md`.

## Module

**`renga_flow.utils.preview`**

| Function | Role |
|----------|------|
| `get_preview_config(config)` | Returns `config["preview"]` dict or `{}`. |
| `previews_configured(config)` | True if prompts exist and `enabled` is not false. |
| `normalize_preview_prompts(preview_cfg)` | `(tag, prompt)` list for logging. |
| `should_run_previews(...)` | Schedule + `forced` from signal. |
| `run_previews(model, config, tb_writer, step, ...)` | Distributed barriers; inference on rank 0 only. |
| `_run_sdxl_previews(...)` | Calls `StableDiffusionXLPipeline.__call__` with `output_type="pil"`. |

## Training loop integration

**`renga_flow.main._run_training`**

1. Optional `preview_before_first_step` before the loop (step / x-axis `0`).
2. After each step, `saver.process_step` returns `(checkpointed, saved, signals)`.
3. If `should_run_previews(..., forced=signals.should_preview)`, call `run_previews`.

X-axis for `add_image` matches scalars: `examples` when `x_axis_examples`, else `step`.

## Block swap and optimizer

Same pattern as **`evaluate()`**: `prepare_block_swap_inference`, `empty_cuda_cache`, optional `optimizer.eval()`, then restore training mode and `prepare_block_swap_training`.

## TensorBoard

```python
tb_writer.add_image(f"preview/{name}", chw_float_0_1, global_step)
```

CHW float tensor in `[0, 1]` from PIL via `_pil_to_chw_float`.

## Signals

**`renga_flow.utils.signal_files`**: `SIGNAL_PREVIEW = "preview"` → `SignalResult.should_preview`.

Extend `SignalResult` and `broadcast_object_list` length when adding signals.

## Extending to other models

Add a branch in `run_previews` (or a `generate_preview` method on `ModelPipelineProtocol`) for new `model.type` values. SDXL uses the loaded diffusers pipeline on `model._pipeline`.

## Tests

- `tests/test_preview.py` — config helpers and `preview` signal (no GPU).
