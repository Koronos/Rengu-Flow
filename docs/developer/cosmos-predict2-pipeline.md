# Cosmos Predict2 pipeline (developer)

## Layout

| Module | Role |
|--------|------|
| `renga_flow/model/cosmos_predict2/pipeline.py` | `CosmosPredict2Pipeline` — orchestration |
| `dit.py` | `MiniTrainDIT` (from diffusion-pipe `cosmos_predict2_modeling.py`) |
| `llm_adapter.py` | LLM adapter blocks |
| `wan_vae.py` | Wan VAE encode/decode |
| `layers.py` | DeepSpeed pipeline layers (`InitialLayer`, `TransformerLayer`, …) |
| `config.py` | `get_dit_config` from checkpoint keys |
| `paths.py` | Bundled tokenizer assets via `importlib.resources` |
| `renga_flow/networks/adapter_dit.py` | LoRA (PEFT) and LoKr save/load (Comfy prefix) |
| `renga_flow/data/preprocess_media.py` | `PreprocessMediaFile` for dataset cache |

Registry: `register_model("cosmos_predict2")`, alias `anima` → same factory.

## Cache hooks

Implemented (not `[TODO]`):

- `get_preprocess_media_file_fn` → `PreprocessMediaFile`
- `get_call_vae_fn` → Wan VAE latents
- `get_call_text_encoder_fn` → Qwen3 hidden states + T5 token ids for adapter path
- `get_text_encoders` → `[text_encoder]` when `cache_text_embeddings` is true

## Adapters

`configure_adapter` delegates to `adapter_dit.configure`. LoKr uses `lokr_sdxl._apply_lokr_vendored` (no ComfyUI). `save_adapter` writes `diffusion_model.*` keys; LoKr injects `.alpha` tensors per module.

`load_and_fuse_adapter` intentionally raises `NotImplementedError`.

## Param groups

`get_param_groups` splits trainable parameters by name (self/cross attn, MLP, adaln, llm_adapter). `lr == 0` sets `requires_grad_(False)` for that bucket.

## Full finetune

When `[adapter]` is absent, `load_diffusion_model` leaves base DiT parameters trainable (`original_name` set for saver). Text encoder stays frozen in `__init__`.

## Reference

Behavior is aligned with diffusion-pipe `models/cosmos_predict2.py` and local diffs on `base.py` (LoKr DiT). See [dependencies-and-upstream.md](dependencies-and-upstream.md).
