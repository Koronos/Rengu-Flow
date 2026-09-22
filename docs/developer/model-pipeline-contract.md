# Model pipeline contract — implementation status

`rengu_flow.model.base.ModelPipelineProtocol` defines methods the orchestrator and `DatasetManager` expect. Unless noted, **`sdxl`**, **`cosmos_predict2`** and **`krea2`** implement the cache and training hooks below.

Reference: `rengu_flow/model/base.py`, implementations `rengu_flow/model/sdxl.py`, `rengu_flow/model/cosmos_predict2/pipeline.py`, `rengu_flow/model/krea2/pipeline.py`.

| Method | SDXL | Cosmos Predict2 | Krea 2 |
|--------|------|-----------------|--------|
| `load_diffusion_model` | Implemented | Implemented | Implemented (diffusers or original layout) |
| `get_vae` | Implemented | Implemented (Wan VAE) | Implemented (Qwen-Image VAE) |
| `get_text_encoders` | Implemented when `cache_text_embeddings` (default true) | Implemented when `cache_text_embeddings` | Implemented (`cache_text_embeddings` required) |
| `configure_adapter` | Implemented (`lora` / `lokr` / `lycoris_*`) | `DiTPipeline` (`adapter_dit`: `lora` / `lokr` / `lycoris_*`) | `DiTPipeline` |
| `save_adapter` | Implemented | `DiTPipeline` (Comfy `diffusion_model.` keys) | `DiTPipeline` (`transformer.` keys) |
| `load_adapter_weights` | Implemented | `DiTPipeline` | `DiTPipeline` |
| `load_and_fuse_adapter` | Implemented | **Not supported** — raises `NotImplementedError` (documented) | **Not supported** |
| `save_model` | Implemented | Implemented (`net.` prefix) | Implemented (diffusers transformer folder) |
| `get_preprocess_media_file_fn` | Implemented (`PreprocessMediaFile`, 16px round) | Implemented (`PreprocessMediaFile`) | Implemented (image only) |
| `get_call_vae_fn` | Implemented | Implemented | Implemented |
| `get_call_text_encoder_fn` | Implemented — TE1 `prompt_embeds`, TE2 `prompt_embeds_2` + `pooled_prompt_embeds` | Implemented | Implemented (`prompt_embeds` + `text_mask`) |
| `prepare_inputs` | Implemented | Implemented (`dit_common` flow matching) | Implemented (`dit_common` flow matching) |
| `to_layers` | Implemented | Implemented | Implemented (+ TREAD route layers) |
| `model_specific_dataset_config_validation` | Default no-op | Implemented (`frame_buckets` must include `1`) | Default no-op |
| `get_param_groups` | Implemented | Implemented (+ `llm_adapter_lr`) | Default (single group) |
| `get_loss_fn` | Implemented | `DiTPipeline` | `DiTPipeline` |
| `enable_block_swap` | Implemented (`get_block_swap_modules`) | Implemented (`transformer.blocks`) | Implemented (`transformer.transformer_blocks`) |
| `prepare_block_swap_training` | Base (`BlockSwapOffloader`) | Base | Base |
| `prepare_block_swap_inference` | Base | Base | Base |
| `freeze_text_encoders` | Implemented | No-op (frozen in `__init__`) | No-op |

Register new models via `rengu_flow.registry.models.register_model`. Built-in: `sdxl`, `cosmos_predict2` (alias `anima`), `krea2`.

## Shared DiT building blocks (`rengu_flow.model.dit_common`)

Flow-matching DiT pipelines (`cosmos_predict2`, `krea2`, and new DiTs) subclass
`dit_common.DiTPipeline` instead of `BasePipeline`. Only code that is identical across models
lives there; anything needing a per-model branch (quantization scheme, `prepare_preview_memory`,
the Euler/CFG loop, pipe layers) stays in the model package.

- **`DiTPipeline`** — `configure_adapter` / `save_adapter` / `load_adapter_weights` via
  `networks.adapter_dit`, driven by class attributes `adapter_target_modules`,
  `adapter_layer_groups`, `adapter_export_prefix` (`None` = `diffusion_model.`); `get_loss_fn`;
  the preview lifecycle `ensure_vae_for_preview`, `ensure_text_encoder_for_preview`,
  `offload_text_encoder_after_encode`, `restore_after_preview`. Model hooks:
  `_reload_vae_for_preview()`, `_reload_text_encoder_for_preview()` (returns the module),
  `_preview_vae_module()` (default `self.vae`). Helpers for the model's own
  `prepare_preview_memory`: `_suspend_training_block_swap()`, `_make_preview_offloader(...)`,
  `_training_block_swap_active()`.
- **Flow matching** — `sample_timesteps(model_config, bs, device, quantile)` (`logit_normal` /
  `uniform`, `sigmoid_scale`), `shift_timesteps(t, shift, mu)` (fixed `model.shift` wins, else
  exponential shift at `mu`), `calculate_shift(seq_len, base_seq_len, max_seq_len, base_shift,
  max_shift)`, `time_shift(mu, sigma, t)`, `add_flow_noise(latents, t)` → `(noisy, target, t)`.
- **Preview** — `preview_compute_dtype(pipeline)`, `preview_autocast(pipeline)`.

Tests: `tests/test_dit_common.py` (bit-exact against the pre-extraction krea2/cosmos math).

See also [Dataset and cache — model hooks](dataset-and-cache.md#model-hooks-for-cache).
