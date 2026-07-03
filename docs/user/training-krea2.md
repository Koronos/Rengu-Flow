# Training Krea 2

Krea 2 is an **open-weights 12B-parameter DiT** text-to-image model released by Krea AI
(June 2026). It conditions on a **Qwen3-VL** multimodal text encoder (a stack of tapped
hidden-state layers, not just the last one) and encodes/decodes images through the
**Qwen-Image VAE** (`f8c16`, 8× spatial compression). In TOML always use:

- `type = "krea2"`

Two checkpoints exist and are **not interchangeable**:

- **`krea/Krea-2-Raw`** — the undistilled base. **Train on this one.**
- **`krea/Krea-2-Turbo`** — a distilled few-step checkpoint for fast inference only. It is not
  a valid training base.

## Getting the checkpoint

Krea 2 trains from **per-component local files**, the same pattern as Cosmos/Anima — no full
diffusers folder required. Recommended route: download the three files from
[Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) on Hugging Face (or use the
official `raw.safetensors` for the DiT):

| Component | Config key | File |
|-----------|------------|------|
| DiT | `model.transformer_path` | `diffusion_models/krea2_raw_bf16.safetensors` (or the official `raw.safetensors`) |
| Text encoder | `model.text_encoder_path` | `text_encoders/qwen3vl_4b_bf16.safetensors` |
| Image VAE | `model.vae_path` | `vae/qwen_image_vae.safetensors` — **the same file Cosmos/Anima setups use**; point at your existing copy instead of downloading a second one |

The tokenizer is bundled with rengu — no download or path needed unless you want to override it
with `model.tokenizer_path`. These are the same files ComfyUI loads and the same layout
kohya/musubi-tuner accept, so one download serves every trainer.

Single-file DiT checkpoints in the original Krea key layout (`blocks.N.attn.wq`, `mod.lin`,
`txtfusion...`, used by both the official `raw.safetensors` and ComfyUI's file) are
key-converted automatically — no manual conversion step. Pre-quantized fp8/nvfp4 "scaled"
single files are rejected with a clear error: train from the bf16 file; use
`model.transformer_4bit` / `model.transformer_fp8_matmul` for VRAM instead.

**Full diffusers folder (alternative):** if you already have the diffusers-layout release
(`transformer/`, `vae/`, `text_encoder/`, `tokenizer/` subfolders), point `model.checkpoint_path`
at it instead:

```bash
huggingface-cli download krea/Krea-2-Raw --local-dir /path/to/Krea-2-Raw
```

`checkpoint_path` fills in any component whose `*_path` is left empty
(`<checkpoint_path>/<transformer|vae|text_encoder>`); a `transformer_path` / `vae_path` /
`text_encoder_path` set alongside it always overrides that one component. Either route works —
nothing is ever downloaded automatically, rengu never resolves repo ids.

## `[model]` fields

| Config key | What it is | Required | Default |
|------------|------------|----------|---------|
| **`type`** | Model type. | Yes | — |
| **`dtype`** | Load/compute dtype for the VAE, text encoder, adapters, and (unless overridden) the DiT. | Yes | — |
| **`transformer_path`** | DiT: the official `raw.safetensors` / ComfyUI's `krea2_raw_bf16.safetensors`, or a diffusers `transformer/` folder. Rejects pre-quantized fp8/nvfp4 "scaled" files. | One of `transformer_path` / `checkpoint_path` | — |
| **`vae_path`** | Qwen-Image VAE: `qwen_image_vae.safetensors` (same file Cosmos uses) or a diffusers `vae/` folder. | One of `vae_path` / `checkpoint_path` | — |
| **`text_encoder_path`** | Qwen3-VL: `qwen3vl_4b_bf16.safetensors` or a transformers `text_encoder/` folder. | One of `text_encoder_path` / `checkpoint_path` | — |
| **`checkpoint_path`** | Full diffusers-layout folder (`transformer/`, `vae/`, `text_encoder/`); fills any of the three component paths left empty. | No | Unset |
| **`tokenizer_path`** | Folder with tokenizer files. | No | Bundled Qwen3-VL tokenizer (`rengu_flow/model/krea2/assets/qwen3vl_4b`) |
| **`max_sequence_length`** | Prompt token budget before truncation. Lower it to shrink the text-embedding cache; captions longer than this lose their tail. | No | `512` |
| **`transformer_dtype`** | DiT checkpoint load dtype only (VAE/text unaffected). | No | `dtype` |
| **`transformer_4bit`** | Quantize the frozen DiT's linears to 4-bit NF4 (bitsandbytes). Mutually exclusive with `transformer_fp8_matmul`. | No | `false` |
| **`transformer_fp8_matmul`** | Quantize the frozen DiT's linears to fp8 scaled matmul. Mutually exclusive with `transformer_4bit`. | No | `false` |
| **`fp8_matmul_dtype`** | `"e5m2"` or `"e4m3"`, only when `transformer_fp8_matmul = true`. | No | `"e5m2"` |
| **`timestep_sample_method`** | `"logit_normal"` or `"uniform"` timestep sampling for training. | No | `"logit_normal"` |
| **`sigmoid_scale`** | Scales the logit-normal sample before the sigmoid; only used when `timestep_sample_method = "logit_normal"`. | No | `1.0` |
| **`shift`** | Fixed rectified-flow time shift. When set, it **overrides** the default resolution-aware dynamic shift below. | No | Unset (dynamic) |
| **`cache_text_embeddings`** | Always required `true` — the tapped 12-layer Qwen3-VL stack cannot run inside the training graph. Setting it `false` is rejected at startup. | No | `true` |

### Minimal `[model]` example

```toml
[model]
type = "krea2"
dtype = "bfloat16"
transformer_path = "path/to/krea2_raw_bf16.safetensors"
vae_path = "path/to/qwen_image_vae.safetensors"
text_encoder_path = "path/to/qwen3vl_4b_bf16.safetensors"
```

### Timestep shift (training objective)

Training is **rectified flow** with a **velocity target** (`noise - clean_latents`, the
usual flow-matching parameterization). Timesteps are drawn from a logit-normal
distribution and then passed through an **exponential time shift**: by default this shift
is **resolution-aware** — `mu` is computed from the packed image sequence length (patch-2
tokens), interpolating from `0.5` at 256 tokens to `1.15` at 6400 tokens, matching the
reference Krea 2 scheduler config. Set `model.shift` to a fixed number only if you want to
override that per-resolution behavior with a single constant.

## Modes

### LoRA

```toml
[adapter]
type = "lora"
rank = 16
```

Example: `examples/minimal_config_krea2_lora.toml`.

### LoKr

```toml
[adapter]
type = "lokr"
rank = 6
factor = -1
```

`alpha` is derived from `rank` (do not set `alpha` in TOML). Saves use the `transformer.` key
prefix (matching the official Krea 2 LoRA convention) with per-module `.alpha` — different from
Cosmos Predict2's `diffusion_model.*` prefix.

Example: `examples/minimal_config_krea2_lokr.toml`.

The built-in `lokr` is quantization-aware (it routes the base matmul through the quantized
`base_linear` and adds the Kronecker delta on top), so it is the adapter to reach for when
training on top of `transformer_4bit` / `transformer_fp8_matmul`. See the `lokr` vs
`lycoris_lokr` comparison in
[Training Cosmos Predict2](training-cosmos-predict2-lora-lokr-finetune.md#why-are-there-two-lokr-types-lokr-vs-lycoris_lokr)
— the trade-offs are identical for Krea 2.

### LyCORIS networks

All seven LyCORIS algorithms are available for the DiT (same backend as Cosmos Predict2 and
SDXL): `lycoris_locon`, `lycoris_loha`, `lycoris_lokr`, `lycoris_dylora`, `lycoris_glora`,
`lycoris_diag_oft`, `lycoris_boft`. DoRA is the `dora_wd` toggle on locon/loha/lokr, not a
separate type.

```toml
[adapter]
type = "lycoris_loha"   # any of the seven types above
rank = 8
```

The same runtime constraints documented for Cosmos apply here — they follow from the
algorithm's mechanism, not from the checkpoint:

- **`lycoris_dylora`** samples a random sub-rank per forward, which breaks checkpoint
  recompute — requires `activation_checkpointing = false`; pair with `blocks_to_swap` to
  recover VRAM.
- **`lycoris_diag_oft` / `lycoris_boft`** rebuild the full weight matrix every step
  (orthogonal rotation) and are the most VRAM-hungry LyCORIS types — add `blocks_to_swap`
  on 16 GB cards.
- **Quantized base is not supported with `lycoris_*`**: the LyCORIS backend matches targets
  by exact class name `Linear`, so it silently skips the quantized linears
  (`Fp8MatmulLinear` / `Linear4bit`) — config validation rejects `transformer_4bit` /
  `transformer_fp8_matmul` together with any `lycoris_*` adapter. Use `adapter.type = "lokr"`
  (quantization-aware) on a quantized base instead.

### Full finetune

Omit the `[adapter]` section. All DiT parameters with `requires_grad` are trained; export
writes a diffusers-layout transformer folder (`config.json` + `diffusion_pytorch_model.safetensors`)
loadable by `Krea2Transformer2DModel.from_pretrained` / diffusers' `Krea2Pipeline`, not
`adapter_model.safetensors`.

Example: `examples/minimal_config_krea2_finetune.toml`.

### Adapter targets

By default adapters and LyCORIS networks attach to **every `Linear` in the DiT**: the per-block
attention/MLP layers, the text-fusion stack (`Krea2TextFusionBlock`, which collapses the 12
tapped Qwen3-VL layers into one text-conditioning sequence), and the shared `img_in` / `txt_in`
/ time projections and final output linear. This is the model authors' recommended LoRA scope
(their reference configuration is rank 32 / alpha 32).

Narrow that scope with **layer groups** — named, model-defined selections you can combine —
or with raw glob patterns. Both work for `lora`, `lokr`, and every `lycoris_*` type:

```toml
[adapter]
type = "lora"
rank = 32
# Train only the text-conditioning stack (Krea2TextFusion + txt_in projection):
layer_groups = ["text_adapter"]
# or combine several:
# layer_groups = ["text_adapter", "attention"]
```

Krea 2 groups: `text_adapter` (text_fusion + txt_in), `attention`
(per-block attention projections), `feedforward` (per-block SwiGLU), `time_modulation`
(`time_mod_proj`), `image_in_out` (`img_in` + final layer). Cosmos Predict2 defines
`self_attention`, `cross_attention`, and `mlp`.

For anything a named group doesn't cover, `adapter.target_include` / `adapter.target_exclude`
take fnmatch globs against the dotted module path (e.g. `target_include = ["*attn*"]`,
`target_exclude = ["*.to_gate"]`). Groups expand into `target_include`, so the two compose;
patterns that match nothing fail at startup with example module paths.

All adapter exports use the official Krea 2 `transformer.` key prefix over diffusers module
names (`lora` also matches the official `lora_A`/`lora_B` weight names), so official Krea 2
LoRAs can be loaded as a starting point via `adapter.init_from_existing`, and rengu's own
exports load in ComfyUI and diffusers.

## VRAM guidance

The 12B DiT is **~26 GB in bf16** by itself — that alone exceeds a 16 GB or 24 GB card
before any activations, adapter, or optimizer state are counted. Use these as starting
points and adjust with the VRAM ladder below.

| Card | Recommended setup |
|------|--------------------|
| **16 GB** | Adapter training (LoRA/LoKr/LyCORIS), `model.transformer_4bit = true` (or `transformer_fp8_matmul`), plus `blocks_to_swap` (e.g. `20` of 28 blocks). `activation_checkpointing = true`. Full finetune does not fit here. |
| **24 GB** | Adapter training with the frozen base quantized (`transformer_4bit` or `transformer_fp8_matmul`) and little or no block swap; or bf16 base with a moderate `blocks_to_swap`. Full finetune still needs block swap + `optimizer.gradient_release = true`. |
| **48 GB** | Adapter training fits with the base in bf16, no quantization needed. Full finetune fits with `blocks_to_swap` + `optimizer.gradient_release = true` on a single GPU, or comfortably across multiple GPUs without block swap. |

```toml
# 16 GB example: quantized base + block swap
blocks_to_swap = 20
activation_checkpointing = true

[model]
type = "krea2"
dtype = "bfloat16"
checkpoint_path = "path/to/Krea-2-Raw"
transformer_4bit = true

[adapter]
type = "lokr"
rank = 6
factor = -1
```

Full finetune needs either multiple GPUs or heavy `blocks_to_swap` with
`optimizer.gradient_release = true` (each block's optimizer step runs in the backward pass
while that block is resident) — the same lever documented model-agnostically in
[VRAM optimization](../developer/vram-optimization.md). For the general OOM playbook (text-embedding
cache, checkpointing, memory-efficient optimizer states, block swap, in that order), see the
[VRAM ladder](training-loop-and-eval.md#if-it-doesnt-fit-the-vram-ladder).

## Text-embedding cache and `max_sequence_length`

`cache_text_embeddings` is **always `true`** for Krea 2 — the pipeline raises at startup if
it is set `false`. Krea 2 conditions on **12 tapped Qwen3-VL hidden-state layers** per
token (not a single embedding vector), so caching is what keeps that stack out of the
training graph.

Cached embeddings are **compacted to each caption's valid token count** before being
written to disk (padding lanes are dropped), so cache size scales with actual caption
length rather than the fixed prompt budget. `model.max_sequence_length` (default `512`)
caps that budget — lower it to shrink the cache further if your captions are short;
captions longer than the limit lose their tail.

Run the cache pass before training:

```bash
rengu cache --config my.toml
```

## Previews

Krea 2 previews use Euler flow-matching sampling aligned with training. Defaults:
**`num_inference_steps = 28`**, **`guidance_scale = 4.5`** — the Krea CFG convention is
`velocity = cond + guidance_scale × (cond − uncond)` (note this differs from the usual
`uncond + scale × (cond − uncond)` form); setting `guidance_scale = 0` disables CFG (single
forward pass per step). `preview_blocks_to_swap` is supported to stream DiT blocks from CPU
during the preview loop instead of moving the whole model onto the GPU.

```toml
[preview]
num_inference_steps = 28
guidance_scale = 4.5
negative_prompt = ""
width = 1024
height = 1024
prompts = [
  "photo of a red sports car, studio lighting",
]
```

See [Training previews](previews.md) for the shared `[preview]` schema (schedule,
per-prompt tables, signal files).

## Export formats

- **Adapter training** (`lora` / `lokr` / any `lycoris_*`) writes `adapter_model.safetensors`
  with the official Krea 2 `transformer.*` key prefix over diffusers module names (plus
  per-module `.alpha` for `lokr` / `lycoris_*`) — `lora` also uses the official `lora_A`/
  `lora_B` weight names, so it is loadable by ComfyUI and diffusers as-is. This differs from
  Cosmos Predict2's `diffusion_model.*` prefix.
- **Full finetune** writes a diffusers-layout transformer folder — `config.json` (from
  `Krea2Transformer2DModel.save_config`) plus `diffusion_pytorch_model.safetensors` — loadable
  by `Krea2Transformer2DModel.from_pretrained` and by diffusers' `Krea2Pipeline` as the
  `transformer` component.

## Validate config

```bash
python -m rengu_flow.main --config my.toml --validate-only
```
