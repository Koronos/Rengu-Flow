# Model pipeline contract — implementation status

`rengu_flow.model.base.ModelPipelineProtocol` defines methods the orchestrator and `DatasetManager` expect. Unless noted, both **`sdxl`** and **`cosmos_predict2`** implement the cache and training hooks below.

Reference: `rengu_flow/model/base.py`, implementations `rengu_flow/model/sdxl.py`, `rengu_flow/model/cosmos_predict2/pipeline.py`.

| Method | SDXL | Cosmos Predict2 |
|--------|------|-----------------|
| `load_diffusion_model` | Implemented | Implemented |
| `get_vae` | Implemented | Implemented (Wan VAE) |
| `get_text_encoders` | Implemented when `cache_text_embeddings` (default true) | Implemented when `cache_text_embeddings` |
| `configure_adapter` | Implemented (`lora` / `lokr` / `lycoris_*`) | Implemented via `adapter_dit` (`lora` / `lokr` / `lycoris_*`) |
| `save_adapter` | Implemented | Implemented (Comfy keys) |
| `load_adapter_weights` | Implemented | Implemented |
| `load_and_fuse_adapter` | Implemented | **Not supported** — raises `NotImplementedError` (documented) |
| `save_model` | Implemented | Implemented (`net.` prefix) |
| `get_preprocess_media_file_fn` | Implemented (`PreprocessMediaFile`, 16px round) | Implemented (`PreprocessMediaFile`) |
| `get_call_vae_fn` | Implemented | Implemented |
| `get_call_text_encoder_fn` | Implemented — TE1 `prompt_embeds`, TE2 `prompt_embeds_2` + `pooled_prompt_embeds` | Implemented |
| `prepare_inputs` | Implemented | Implemented |
| `to_layers` | Implemented | Implemented |
| `model_specific_dataset_config_validation` | Default no-op | Implemented (`frame_buckets` must include `1`) |
| `get_param_groups` | Implemented | Implemented (+ `llm_adapter_lr`) |
| `get_loss_fn` | Implemented | Implemented |
| `enable_block_swap` | Implemented (`get_block_swap_modules`) | Implemented (`transformer.blocks`) |
| `prepare_block_swap_training` | Base (`BlockSwapOffloader`) | Base |
| `prepare_block_swap_inference` | Base | Base |
| `freeze_text_encoders` | Implemented | No-op (frozen in `__init__`) |

Register new models via `rengu_flow.registry.models.register_model`. Built-in: `sdxl`, `cosmos_predict2`.

See also [Dataset and cache — model hooks](dataset-and-cache.md#model-hooks-for-cache).
