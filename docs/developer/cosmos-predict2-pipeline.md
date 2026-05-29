# Cosmos Predict2 pipeline (developer)

## Layout

| Module | Role |
|--------|------|
| `rengu_flow/model/cosmos_predict2/pipeline.py` | `CosmosPredict2Pipeline` — orchestration |
| `dit.py` | `MiniTrainDIT` (from diffusion-pipe `cosmos_predict2_modeling.py`) |
| `llm_adapter.py` | LLM adapter blocks |
| `wan_vae.py` | Wan VAE encode/decode |
| `text.py` | Qwen3/T5 load, `tokenize`, `compute_text_embeddings` |
| `layers.py` | DeepSpeed pipeline layers (`InitialLayer`, `TransformerLayer`, …) |
| `config.py` | `get_dit_config` from checkpoint keys (max 1024) |
| `paths.py` | Bundled tokenizer assets via `importlib.resources` |
| `rengu_flow/networks/adapter_dit.py` | LoRA (PEFT) and LoKr save/load (Comfy prefix) |
| `rengu_flow/data/preprocess_media.py` | `PreprocessMediaFile` for dataset cache |

Registry: `register_model("cosmos_predict2")` only (Anima is a checkpoint branding name in user docs, not a `type` value).

## Cache hooks

Implemented (not `[TODO]`):

- `get_preprocess_media_file_fn` → `PreprocessMediaFile`
- `get_call_vae_fn` → Wan VAE latents
- `get_call_text_encoder_fn` → Qwen3 hidden states + T5 token ids for adapter path
- `get_text_encoders` → `[text_encoder]` when `cache_text_embeddings` is true

## Adapters

`configure_adapter` delegates to `adapter_dit.configure`. LoKr uses `lokr_sdxl._apply_lokr_vendored` (no ComfyUI). `save_adapter` writes `diffusion_model.*` keys; LoKr injects `.alpha` tensors per module.

`load_and_fuse_adapter` intentionally raises `NotImplementedError` (adapter weights stay separate; use `load_adapter_weights` for training). Covered by `tests/test_cosmos_load_and_fuse.py`.

## Param groups

`get_param_groups` splits trainable parameters by name (self/cross attn, MLP, adaln, llm_adapter). `lr == 0` sets `requires_grad_(False)` for that bucket.

## Full finetune

When `[adapter]` is absent, `load_diffusion_model` leaves base DiT parameters trainable (`original_name` set for saver). Text encoder stays frozen in `__init__`.

## Dtype overrides (`[model]`)

Parsed in **`rengu_flow.config.defaults.set_config_defaults`**. Implemented in **`pipeline.py`** as follows:

| Key | Role |
|-----|------|
| `dtype` | Required. VAE, `load_text_stack`, adapter dtype, and DiT layers in `KEEP_IN_HIGH_PRECISION` / 1D / `llm_adapter` at load time. |
| `transformer_dtype` | Optional; defaults to `dtype`. Dtype for most `transformer_path` weights in `load_diffusion_model` (not embedders/norms). |
| `diffusion_model_dtype` | Optional; sets training forward autocast dtype (`main.py` → `AUTOCAST_DTYPE`). Defaults `transformer_dtype` when omitted (`defaults.py`). |

User-facing summary: **`docs/user/training-cosmos-predict2-lora-lokr-finetune.md`** (Precision and **Performance and VRAM** sections).

### Tuning notes (Anima LoKR)

Highlights for operators (see also user doc **Performance and VRAM**):

- **`pipeline_model.compile()`** is wired in `main.py` when `compile = true` (diffusion-pipe parity). Short smokes penalize compile in the mean; on long runs steady iter was ~0.51 s vs ~0.68–0.70 s without compile — see user doc **Performance and VRAM**.
- **`reentrant_activation_checkpointing`** defaults to `true` for `cosmos_predict2` when AC is on and `blocks_to_swap` is unset (`defaults.py`).
- **`enable_block_swap`** uses shared [`rengu_flow/training/block_swap.py`](../training/block_swap.py) on `transformer.blocks` (see [training-techniques.md](training-techniques.md)).
- Text embeddings: prefer **`cache_text_embeddings`** + `--cache_only` so training does not repeat Qwen3 forward passes.

## Dependencies and upstream sources

Full submodule matrix: [dependencies-and-upstream.md](dependencies-and-upstream.md). Installing rengu-flow does **not** require `git submodule update` or a local diffusion-pipe clone.

| diffusion-pipe (train Anima) | rengu-flow module |
|------------------------------|-------------------|
| `models/cosmos_predict2_modeling.py` | `dit.py` |
| `models/llm_adapter.py` | `llm_adapter.py` |
| `models/wan/vae2_1.py` | `wan_vae.py` + `vae.py` |
| `CosmosPredict2Pipeline.__init__` (TE) | `text.py` + `pipeline.py` |
| `PreprocessMediaFile` | `data/preprocess_media.py` |
| `configure_adapter` / LoKr save | `networks/adapter_dit.py` |
| `configs/qwen3_06b`, `t5_old` | `assets/` + `package-data` |

Licenses: `NOTICE.md` in the package (NVIDIA Apache-2.0, Wan VAE header in `wan_vae.py`).

ComfyUI uses the same `diffusion_model.*` / `net.*` weight layout for inference; it is **not** a training dependency.

Reference commits in diffusion-pipe (traceability only): Anima support, high-res `max_img` 1024 (`b0aa4f1`). Local uncommitted diffs on `base.py` / `cosmos_predict2.py` (LoKr DiT + save) are ported to `adapter_dit`.

## Reference

Behavior is aligned with diffusion-pipe `models/cosmos_predict2.py` and local diffs on `base.py` (LoKr DiT).
