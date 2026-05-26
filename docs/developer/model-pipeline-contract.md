# Model pipeline contract — implementation status

`renga_flow.model.base.ModelPipelineProtocol` defines methods the orchestrator and `DatasetManager` expect. **`[TODO]`** = not implemented on **`SDXLPipeline`** today (or stub only).

Reference: `renga_flow/model/base.py`, implementations `renga_flow/model/sdxl.py`, `renga_flow/model/cosmos_predict2/pipeline.py`.

| Method | SDXL | Cosmos Predict2 |
|--------|------|-----------------|
| `load_diffusion_model` | Implemented | Implemented |
| `get_vae` | Implemented | Implemented (Wan VAE) |
| `get_text_encoders` | **`[TODO]`** — returns `[]` | Implemented when `cache_text_embeddings` |
| `configure_adapter` | Implemented (`lora` / `lokr`) | Implemented via `adapter_dit` |
| `save_adapter` | Implemented | Implemented (Comfy keys) |
| `load_adapter_weights` | Implemented | Implemented |
| `load_and_fuse_adapter` | Implemented | **Not implemented** (raises) |
| `save_model` | Implemented | Implemented (`net.` prefix) |
| `get_preprocess_media_file_fn` | **`[TODO]`** | Implemented (`PreprocessMediaFile`) |
| `get_call_vae_fn` | Implemented | Implemented |
| `get_call_text_encoder_fn` | **`[TODO]`** | Implemented |
| `prepare_inputs` | Implemented | Implemented |
| `to_layers` | Implemented | Implemented |
| `model_specific_dataset_config_validation` | Default no-op | Implemented (`frame_buckets` must include `1`) |
| `get_param_groups` | Implemented | Implemented (+ `llm_adapter_lr`) |
| `get_loss_fn` | Implemented | Implemented |
| `enable_block_swap` | **`[TODO]`** | Not supported (austere) |
| `prepare_block_swap_training` | No-op (base) | No-op |
| `prepare_block_swap_inference` | No-op (base) | No-op |
| `freeze_text_encoders` | Implemented | No-op (frozen in `__init__`) |

Register new models via `renga_flow.registry.models.register_model`. Built-in: `sdxl`, `cosmos_predict2`.

See also [Dataset and cache — model hooks](dataset-and-cache.md#model-hooks-for-cache-sdxl).
