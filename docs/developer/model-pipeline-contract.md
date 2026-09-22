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
- **Preview** — `preview_compute_dtype(pipeline)`, `preview_autocast(pipeline)`;
  `DiTPipeline._prepare_blocks_preview_memory(preview_cfg, blocks_attr="transformer_blocks")` is the
  whole `prepare_preview_memory` for a DiT whose blocks live in `transformer.<blocks_attr>` (krea2,
  qwen_image21): parks the training offloader, then streams the blocks through a preview offloader
  (`preview_blocks_to_swap > 0`) or makes the DiT resident.
- **Frozen-base quantization** (`dit_common.quantize`) — `quantize_frozen_dit(transformer,
  model_config, leaf_names=, skip_substrings=, label=)` applies `model.transformer_fp8_matmul`
  (tensorwise e4m3) / `model.transformer_4bit` (NF4) with the model's own scope (krea2,
  qwen_image21; cosmos keeps its own scheme).
- **Ragged text embeddings** (`dit_common.text`) — `compact_text_embeddings(hidden, mask)` /
  `pad_text_embeddings(embeds, masks)` for per-caption cache rows of any per-token feature shape
  (krea2's `(layers, dim)` stacks, qwen_image21's `(dim,)`).
- **Qwen3-VL text decoder** (`dit_common.qwen3vl`) — `remap_qwen3vl_text_state_dict` (transformers /
  ComfyUI / bare key layouts → `Qwen3VLTextModel` keys, ComfyUI scaled-fp8 dequantized, vision and
  LM-head dropped), `checkpoint_files(path)` (file, sharded folder with index, or folder), and
  `load_qwen3vl_text_model(files, text_config, dtype)` (shard-by-shard, text keys only).
- **Lazy / layer-streamed encoder** (`dit_common.streaming.LazyStreamedEncoder`) — wraps a frozen
  encoder behind the plain `nn.Module` placement API the data layer and preview lifecycle use:
  weights load on the first `.to(<cuda>)` (`.to("cpu")` while unloaded is a no-op, `.to("meta")`
  unloads; a 0-size `meta` placeholder parameter stands in while unloaded), and with
  `offload="auto"|"stream"` the repeated layers stay in pinned host RAM and are copied to the GPU
  per forward by hooks (same-stream copies, immutable masters, no copy-back). qwen_image21 uses it
  for its 17.5 GB Qwen3-VL-8B.

Tests: `tests/test_dit_common.py` (bit-exact against the pre-extraction krea2/cosmos math),
`tests/test_qwen_image21_model.py` (streaming wrapper, Qwen3-VL loader/remap).

### Qwen-Image 2.1 (`rengu_flow/model/qwen_image21/`)

- `dit.py` / `vae.py` — vendored from diffusers `main` 6256aa7 (see `NOTICE.md`).
- `pipeline.py` — `QwenImage21Pipeline`: component paths (`model.<component>_path` over
  `model.diffusers_path/<component>`), lazy streamed text encoder, fp8/NF4 scope
  (`QUANT_LEAF_NAMES` / `QUANT_SKIP_SUBSTRINGS`: per-block linears only), `ADAPTER_LAYER_GROUPS`,
  `EXPORT_PREFIX = "transformer."` (ComfyUI maps `transformer.<diffusers module>` for 2.1, including
  the `gate_layer`/`proj` halves of its fused `img_mlp.gate_up`), training shift constants.
- `text.py` — t2i prompt encoding, a line-for-line port of the reference
  `_get_qwen_prompt_embeds` (tested for equality against a literal copy on a tiny Qwen3-VL).
- `layers.py` — pipe layers over the tensors-only tuple `(hidden, temb, modulation,
  view_as_real(rope), key_valid | 0-size, layout (L, h, w, 0))`; each block re-derives the
  prefill segments `[(0, L, True)]` and target mask from `layout`; `FinalLayer` slices the target
  tokens before the output norm.
- `preview_sampling.py` — reference scheduler sigmas (shift + `shift_terminal`), Euler, prefix KV
  cache (`extract` on step 0, `cached` after), CFG by batching the negative prompt.
- `loading.py` — diffusers folders, ComfyUI DiT/TE single files (fused-MLP split), bundled configs
  under `assets/`.

Not implemented: image-conditioned (edit) training — it needs condition-image VAE latents in the
cache, the ti2i template with vision inputs through the full Qwen3-VL, and per-sample image
blocks in the joint sequence.

See also [Dataset and cache — model hooks](dataset-and-cache.md#model-hooks-for-cache).
