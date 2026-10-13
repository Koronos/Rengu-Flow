# Model pipeline contract — implementation status

`rengu_flow.model.base.ModelPipelineProtocol` defines methods the orchestrator and `DatasetManager` expect. Unless noted, every registered model (**`sdxl`**, **`cosmos_predict2`**, **`krea2`**, **`qwen_image21`**, **`nanosaur2`**) implements the cache and training hooks below. `qwen_image21` follows the Krea 2 column (it is a `DiTPipeline` too; its specifics are in its section below). The flow for adding a model is [adding-a-model.md](adding-a-model.md); the model's descriptor is its `spec.py` ([model-capabilities-and-validation.md](model-capabilities-and-validation.md)).

Reference: `rengu_flow/model/base.py`, implementations `rengu_flow/model/sdxl.py`, `rengu_flow/model/cosmos_predict2/pipeline.py`, `rengu_flow/model/krea2/pipeline.py`, `rengu_flow/model/nanosaur2/pipeline.py`.

| Method | SDXL | Cosmos Predict2 | Krea 2 | Nanosaur2 |
|--------|------|-----------------|--------|-----------|
| `load_diffusion_model` | Implemented | Implemented | Implemented (diffusers or original layout) | Implemented (single release file; architecture read from the weights) |
| `get_vae` | Implemented | Implemented (Wan VAE) | Implemented (Qwen-Image VAE) | Implemented (semantic DINOv2 VAE, f16) |
| `get_text_encoders` | Implemented when `cache_text_embeddings` (default true) | Implemented when `cache_text_embeddings` | Implemented (`cache_text_embeddings` required) | Implemented (`cache_text_embeddings` required; lazy Gemma3-270M) |
| `configure_adapter` | Implemented (`lora` / `lokr` / `lycoris_*`) | `DiTPipeline` (`adapter_dit`: `lora` / `lokr` / `lycoris_*`) | `DiTPipeline` | `DiTPipeline` (reference LoRA scope) |
| `save_adapter` | Implemented | `DiTPipeline` (Comfy `diffusion_model.` keys) | `DiTPipeline` (`transformer.` keys) | `DiTPipeline` (`diffusion_model.` keys + `.alpha`) |
| `load_adapter_weights` | Implemented | `DiTPipeline` | `DiTPipeline` | `DiTPipeline` |
| `load_and_fuse_adapter` | Implemented | **Not supported** — raises `NotImplementedError` (documented) | **Not supported** | **Not supported** |
| `save_model` | Implemented | Implemented (`net.` prefix) | Implemented (diffusers transformer folder) | Implemented (`nanosaur2_diffusion_model.safetensors`, release layout) |
| `get_preprocess_media_file_fn` | Implemented (`PreprocessMediaFile`, 16px round) | Implemented (`PreprocessMediaFile`) | Implemented (image only) | Implemented (image only, 16 px round) |
| `get_call_vae_fn` | Implemented | Implemented | Implemented | Implemented |
| `get_call_text_encoder_fn` | Implemented — TE1 `prompt_embeds`, TE2 `prompt_embeds_2` + `pooled_prompt_embeds` | Implemented | Implemented (`prompt_embeds` + `text_mask`) | Implemented (`prompt_embeds` + `token_weights`) |
| `prepare_inputs` | Implemented | Implemented (`dit_common` flow matching) | Implemented (`dit_common` flow matching) | Implemented (reference recipe, path-drop flags) |
| `to_layers` | Implemented | Implemented | Implemented (+ TREAD route layers) | Implemented (+ SPRINT sparse-window layers) |
| `model_specific_dataset_config_validation` | Default no-op | Implemented (`frame_buckets` must include `1`) | Default no-op | Default no-op (`default_uncond_fraction = 0.1` for the training dataset, `data.dataset.apply_model_training_dataset_defaults`) |
| `get_param_groups` | Implemented | Implemented (+ `llm_adapter_lr`) | Default (single group) | Default (single group) |
| `get_loss_fn` | Implemented | `DiTPipeline` | `DiTPipeline` | `DiTPipeline` (reference weight in the mask) |
| `enable_block_swap` | Implemented (`get_block_swap_modules`) | Implemented (`transformer.blocks`) | Implemented (`transformer.transformer_blocks`) | **Not supported** (670M fits) |
| `prepare_block_swap_training` | Base (`BlockSwapOffloader`) | Base | Base | — |
| `prepare_block_swap_inference` | Base | Base | Base | — |
| `freeze_text_encoders` | Implemented | No-op (frozen in `__init__`) | No-op | No-op |

A model is registered in two places, both explicit: the pipeline class with `@register_model("<type>")` (`rengu_flow.registry.models`; the class is imported lazily from the spec's `pipeline = "<module>:<Class>"`), and its `SPEC` module in `rengu_flow/registry/builtin_models.py`. Built-in: `sdxl`, `cosmos_predict2` (UI alias `anima`; the validator requires the canonical id), `krea2`, `qwen_image21`, `nanosaur2`. The pipeline class name must carry `name = "<type>"` (it also names the cache dir).

## Shared DiT building blocks (`rengu_flow.model.dit_common`)

Flow-matching DiT pipelines (`cosmos_predict2` partially, `krea2`, `qwen_image21`, `nanosaur2`, and new DiTs) subclass
`dit_common.DiTPipeline` instead of `BasePipeline`. Only code that is identical across models
lives there; anything needing a per-model branch (text recipe, key conversion, shift constants, the Euler/CFG loop, inter-layer tuple) stays in the model package. A template is a default, never a requirement: a model overrides the method when it differs.

`DiTPipeline` is a **template-method class**: it reads the model's spec (`self.spec`, the registered `ModelCapability`) and class attributes, and calls small hooks.

| Template (provided) | Driven by |
|---------------------|-----------|
| `_init_pipeline(config)` — config, block-swap state, text-cache mode (`spec.text_cache = "required"` refuses `cache_text_embeddings = false`, message from `text_cache_required_reason`) | spec, class attributes |
| `_component_path(component)` — `model.<c>_path`, else `<spec.bundle_key>/<c>`; error from `missing_component_message` | `spec.components`, `spec.bundle_key` |
| `load_diffusion_model` — load in `transformer_dtype`, optional frozen-base quantization (`quant_leaf_names`, `quant_skip_substrings`, `quant_label`), trainable params | hook `_load_transformer(dtype)` |
| `get_vae` / `get_text_encoders` / `get_call_vae_fn` / `get_call_text_encoder_fn` / `_latent_stats` / `encode_latents` — posterior mode of a `T = 1` image VAE normalized by the VAE config stats; ragged text rows | hooks `_load_vae()`, `_encode_text(text_encoder, prompts, device)` |
| `get_block_swap_modules` / `_block_swap_root_modules` | `blocks_attr`, `spec.features["block_swap"]` |
| `save_model` — diffusers transformer folder | override when the export layout differs |
| `trim_micro_batch` | opt in with `text_padding_trimmer()` |
| `preview_prompt_embeds` (memoized via `_memoized_preview_embeds`), `ensure_*_for_preview`, `prepare_preview_memory`, `restore_after_preview` | `preview_keep_reloaded_vae`, `preview_unload_freed_text_encoder`, `spec.preview_spec.memory_knobs`, hooks `_reload_vae_for_preview`, `_reload_text_encoder_for_preview`, `_preview_vae_module` |

What a model still writes: `_load_transformer`, `_load_vae`, `_encode_text`, `prepare_inputs`, `to_layers`, `generate_preview_image`, and its class attributes. The full list with purposes is in [adding-a-model.md, Phase 3](adding-a-model.md#phase-3---port-reusing-dit_common).

- **`DiTPipeline`** — `configure_adapter` / `save_adapter` / `load_adapter_weights` via
  `networks.adapter_dit`, driven by class attributes `adapter_target_modules`,
  `adapter_layer_groups`, `adapter_export_prefix` (`None` = `diffusion_model.`),
  `adapter_export_lora_alpha` (also write a `<module>.alpha` per LoRA module); `get_loss_fn`;
  the preview lifecycle `ensure_vae_for_preview`, `ensure_text_encoder_for_preview`,
  `offload_text_encoder_after_encode`, `restore_after_preview`; defaults shared by the DiTs:
  `_init_transformer_params()` (train mode, `original_name`, trainable for full finetune),
  image-only `get_preprocess_media_file_fn`, `load_and_fuse_adapter` (not supported). Model hooks:
  `_reload_vae_for_preview()`, `_reload_text_encoder_for_preview()` (returns the module; a `LazyStreamedEncoder` is returned as
  is by default),
  `_preview_vae_module()` (default `self.vae`). Helpers for the model's own
  `prepare_preview_memory`: `_suspend_training_block_swap()`, `_make_preview_offloader(...)`,
  `_training_block_swap_active()`. Preview prompt embeddings are memoized with
  `_memoized_preview_embeds(keys, preview_cfg, device, encode_missing)`; after a preview,
  `preview_keep_reloaded_vae` keeps a reloaded (small) VAE in host RAM and
  `preview_unload_freed_text_encoder` unloads a text encoder that caching had freed
  (`_unload_text_encoder_freed_by_caching`).
- **Flow matching** — `sample_timesteps(model_config, bs, device, quantile, logit_mean=0.0)`
  (`logit_normal` / `uniform`, `sigmoid_scale`; `logit_mean` centers a model's reference
  logit-normal), `shift_timesteps(t, shift, mu)` (fixed `model.shift` wins, else
  exponential shift at `mu`), `calculate_shift(seq_len, base_seq_len, max_seq_len, base_shift,
  max_shift)`, `time_shift(mu, sigma, t)`, `add_flow_noise(latents, t)` → `(noisy, target, t)`.
- **Preview** — `preview_compute_dtype(pipeline)`, `preview_autocast(pipeline)`,
  `image_to_pil(image)` (decoded `(3, H, W)` in `[-1, 1]` -> RGB PIL);
  `DiTPipeline._prepare_blocks_preview_memory(preview_cfg, blocks_attr="transformer_blocks")` is the
  whole `prepare_preview_memory` for a DiT whose blocks live in `transformer.<blocks_attr>` (krea2,
  qwen_image21): parks the training offloader, then streams the blocks through a preview offloader
  (`preview_blocks_to_swap > 0`) or makes the DiT resident.
- **Local files** (`dit_common.loading`) — `require_exists(path, what)` (config error naming
  `model.<what>`), `strip_comfy_prefix(state_dict)` (`diffusion_model.` re-export prefix),
  `read_safetensors(path, dtype)` (cast tensor by tensor as read; fp8 kept for guards).
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
- `loading.py` — diffusers folders, ComfyUI DiT/TE/VAE single files (DiT fused-MLP split;
  `convert_original_vae_state_dict`: original Wan-style VAE keys → diffusers names, 1-frame conv
  kernels squeezed to 2D), vision tower from the same TE checkpoint (`model.visual.*`), bundled
  configs under `assets/`.

Image-conditioned (edit) training is supported: `[[directory]] control_path` condition images
become cached control latents plus vision-tower prompt embeddings (see
[Image editing](../user/training-qwen-image21.md#image-editing-edit-training)).

### Nanosaur2 (`rengu_flow/model/nanosaur2/`)

- `dit.py` — port of the official ComfyUI node's DiT with the official module names; `forward`
  is composed of `embed` / `run_block` / `sparse_path` / `final`, which the pipe layers reuse.
  `detect_config(state_dict)` reads the architecture from the weights.
- `layers.py` — `InitialLayer` -> blocks -> `SparseStartLayer` / `SparseEndLayer` around the
  SPRINT middle blocks -> `FinalLayer` (velocity, fp32). Path-drop is per row (`sparse_skip`):
  single-stage runs send only the non-skipping rows through the sparse window; pipeline-parallel
  runs keep static shapes with `torch.where`.
- `text.py` — prompt emphasis `(text:weight)` -> per-token `token_weights` (cached ragged like
  `prompt_embeds`; `utils/cache.py` lists the key), BOS-only empty prompt, Gemma3
  `norm(hidden_states[-2])`.
- `vae.py` — DINOv2 encoder (antialiased position resize, semantic patch path) + LDM decoder.
- `pipeline.py` — reference recipe: logit-normal at ln 2, path-drop flags from the caption length
  (BOS only = dropped), the reference x0-space weight folded into the loss mask,
  `default_uncond_fraction = 0.1` (training dataset only; eval datasets are never dropped).
- `preview_sampling.py` — ComfyUI `simple` schedule of the shift-3 discrete flow, Euler,
  `cfg` / `path_drop` / `alternate` guidance.
- Parity against the node with the real weights: `scripts/parity_nanosaur2.py`.

See also [Dataset and cache — model hooks](dataset-and-cache.md#model-hooks-for-cache).
