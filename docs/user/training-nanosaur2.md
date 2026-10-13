# Training Nanosaur2

Nanosaur2-670M is a **small illustration DiT** (670M parameters) by
[well9472](https://huggingface.co/well9472/Nanosaur2-670M), released under the **MIT** license
(not gated). In TOML always use:

- `type = "nanosaur2"`

It is a research-grade model: it does **not** match Anima in character knowledge or fine detail.
Its appeal is size and openness: it was trained from scratch for roughly $600 (VAE: 12 h on one
H100, initialized from DINOv2-B; base model: 11 H100-days with the Dion3 optimizer on 4M
illustration images through 256² → 512² → 1024² buckets; aesthetic tuning: 4 h), and it
**trains in bf16 on an 8 GB card** — LoRA or full finetune — with no quantization and no block
swap.

| Component | Architecture |
|-----------|--------------|
| DiT | 670M parameters, 18 blocks, hidden size 1536, SPRINT (a sparse middle section of the blocks that can be skipped, see [Path-drop](#path-drop-sparse_skip_prob)) |
| VAE | Semantic DINOv2-based, 129M parameters, 64 latent channels, ×16 spatial compression |
| Text encoder | Gemma3-270M, frozen, penultimate hidden layer; its tokenizer is stored inside the encoder file |

## Getting the checkpoint

Nanosaur2 trains from **three per-component local files**, the same pattern as Cosmos/Anima and
Krea 2. Download them from
[well9472/Nanosaur2-670M](https://huggingface.co/well9472/Nanosaur2-670M) (public, no login):

| Component | Config key | `.env` variable | Direct download | Size |
|-----------|------------|-----------------|-----------------|------|
| DiT | `model.transformer_path` | `RENGU_NANOSAUR2_TRANSFORMER_PATH` | [nanosaur2_diffusion_model.safetensors](https://huggingface.co/well9472/Nanosaur2-670M/resolve/main/nanosaur2_diffusion_model.safetensors) | 1.33 GB |
| Image VAE | `model.vae_path` | `RENGU_NANOSAUR2_VAE_PATH` | [nanosaur2_vae.safetensors](https://huggingface.co/well9472/Nanosaur2-670M/resolve/main/nanosaur2_vae.safetensors) | 0.26 GB |
| Text encoder | `model.text_encoder_path` | `RENGU_NANOSAUR2_TEXT_ENCODER_PATH` | [nanosaur2_text_encoder.safetensors](https://huggingface.co/well9472/Nanosaur2-670M/resolve/main/nanosaur2_text_encoder.safetensors) | 0.54 GB |

The `.env` variables only matter for the GPU smoke tests (see [Smoke tests](#smoke-tests)); a
training TOML always carries the paths itself.

> **Do not train on `nanosaur2_dmad_4step_diffusion_model.safetensors`.** That file is the
> distilled 4-step model, made for inference only.

There is no separate tokenizer to download: the SentencePiece model lives inside
`nanosaur2_text_encoder.safetensors`. Nothing is downloaded automatically — rengu never
resolves repo ids.

## Install

Nanosaur2 uses the same extras as the other DiT models, including `sentencepiece` for the
tokenizer:

```bash
./rengu init cosmos
```

## Minimal config

```toml
output_dir = "output"
dataset = "examples/minimal_nanosaur2_dataset.toml"

epochs = 1
micro_batch_size_per_gpu = 1
pipeline_stages = 1
activation_checkpointing = true

[model]
type = "nanosaur2"
dtype = "bfloat16"
transformer_path = "path/to/nanosaur2_diffusion_model.safetensors"
vae_path = "path/to/nanosaur2_vae.safetensors"
text_encoder_path = "path/to/nanosaur2_text_encoder.safetensors"

[adapter]
type = "lora"
rank = 32

[optimizer]
type = "adamw"
lr = 1e-4
```

Examples: `examples/minimal_config_nanosaur2_lora.toml`, `_lokr.toml`, `_finetune.toml`, with the
dataset `examples/minimal_nanosaur2_dataset.toml` (1024 px with aspect-ratio buckets).

Top-level training keys (`gradient_accumulation_steps`, `activation_checkpointing`, ...) must
sit **before** the first `[section]` header; written under `[model]` they are rejected.
`blocks_to_swap` is not offered for this model (it is small enough not to need it), and the
frozen-base quantization knobs (`transformer_4bit`, `transformer_fp8_matmul`) do not exist.

## `[model]` fields

| Config key | What it is | Required | Default |
|------------|------------|----------|---------|
| **`type`** | `"nanosaur2"`. | Yes | — |
| **`dtype`** | Load/compute dtype of the VAE and the DiT (unless `transformer_dtype` is set). Use `"bfloat16"`. | Yes | — |
| **`transformer_path`** | `nanosaur2_diffusion_model.safetensors`. | Yes | — |
| **`vae_path`** | `nanosaur2_vae.safetensors`. | Yes | — |
| **`text_encoder_path`** | `nanosaur2_text_encoder.safetensors`. The file's name, size and mtime key the text-embedding cache: swapping the encoder re-encodes the captions. | Yes | — |
| **`transformer_dtype`** | DiT checkpoint load dtype only (VAE/text unaffected). | No | `dtype` |
| **`sparse_skip_prob`** | Path-drop rate for samples that have a caption, in [0, 1]. See [Path-drop](#path-drop-sparse_skip_prob). | No | `0.05` |
| **`sparse_skip_prob_uncond`** | Path-drop rate for samples whose caption was dropped, in [0, 1]. | No | `0.5` |
| **`timestep_sample_method`** | Advanced. `"logit_normal"` or `"uniform"`. | No | `"logit_normal"` |
| **`sigmoid_scale`** | Advanced. Std of the logit-normal before the sigmoid. | No | `1.0` |
| **`shift`** | Advanced. Fixed timestep shift (> 0). Leave unset: the reference recipe trains without a shift. | No | Unset |
| **`cache_text_embeddings`** | Always `true` — the frozen Gemma3 encoder only runs while caching. Setting it `false` is rejected at startup. | No | `true` |

## Training recipe

The defaults reproduce the model's reference `train_lora.py`; most runs should not touch them.

- **Timesteps:** logit-normal sampling centered at `ln 2` (so noisier timesteps are favored),
  **no shift**.
- **Loss:** the reference loss is computed in x0 space, `((x0_pred − x) / (t + 0.05))²`.
  rengu computes the equivalent velocity MSE weighted by `(t / (t + 0.05))²` (this is a
  weighted v-loss; the weight is folded into the loss mask, so your `loss_type` still applies).
- **Caption dropout: 10 % by default.** The Nanosaur2 *training* dataset gets
  `uncond_fraction = 0.1` unless you set it yourself (at the dataset level or per
  `[[directory]]`; an explicit value always wins). Eval datasets are never dropped. A dropped
  caption is the encoded empty prompt (only the BOS token). See
  [Dataset configuration](dataset-config.md).
- **Optimizer:** `adamw` at `lr = 1e-4` for LoRA (rank 32) matches the smoke fixture; use a much
  lower rate for full finetune (the example uses `adamw8bit` at `5e-6`) and for LoKr
  (the example uses `3e-5`).

### Path-drop (`sparse_skip_prob`)

SPRINT lets a sample skip the **sparse middle blocks** of the DiT. The `path_drop` and
`alternate` preview guidance modes (the reference sampler's recommended setting) use that
skipped pass as the unconditional branch, so the model has to keep *learning* it during
training: each sample skips the sparse blocks with probability **`sparse_skip_prob = 0.05`**
(**`sparse_skip_prob_uncond = 0.5`** for caption-dropped samples). Setting both to `0` is
allowed but not recommended: the model never trains the path-drop pass, and `alternate` /
`path_drop` previews (and inference with that guidance) degrade, since that branch was never
adapted to your data. Keep the reference values unless you only ever sample with plain `cfg`.

## Prompts and captions

- Captions can be **booru tags or natural language**.
- Quality tags: `newest, masterpiece` in the positive prompt; `oldest, low quality` as the
  negative (the preview default).
- **Emphasis** uses `(text:weight)`, e.g. `(fennec ears:1.3)`. The weights are not applied to the
  embedding magnitude: they become a bias on the attention paid to those tokens, so they work
  like attention weighting. They apply to dataset captions and to preview prompts alike.
- Some artist tags need a high emphasis weight to take effect.
- Prompts are truncated at 256 tokens (BOS + the prompt).

## Modes

### LoRA

```toml
[adapter]
type = "lora"
rank = 32
```

`alpha` is always `rank` (setting it is rejected). Example:
`examples/minimal_config_nanosaur2_lora.toml`.

### LoKr and LyCORIS

```toml
[adapter]
type = "lokr"
rank = 8
factor = -1
```

All LyCORIS algorithms (`lycoris_locon`, `lycoris_loha`, `lycoris_lokr`, `lycoris_dylora`,
`lycoris_glora`, `lycoris_diag_oft`, `lycoris_boft`) are available. The DiT only has `Linear`
layers, so conv options (`train_conv`, `use_tucker`) have no effect. Examples:
`examples/minimal_config_nanosaur2_lokr.toml`. The runtime constraints (e.g. `lycoris_dylora`
needs `activation_checkpointing = false`) are the ones described in
[Training Cosmos Predict2](training-cosmos-predict2-lora-lokr-finetune.md).

### Full finetune

Omit `[adapter]`. All DiT parameters are trained; with `optimizer.type = "adamw8bit"` it fits
an 8 GB card at 1024 px. Example: `examples/minimal_config_nanosaur2_finetune.toml`.

### Adapter targets

By default adapters attach to every `Linear` of the image and text **attention** and
**feed-forward** modules (the reference LoRA scope); the shared embedders and adaLN projections
stay out. Narrow the scope with `adapter.layer_groups`:

```toml
[adapter]
type = "lora"
rank = 32
layer_groups = ["cross_attention"]
```

Groups: `attention` (block attention projections), `cross_attention` (the text key/value
projection of those attention layers), `feedforward` (block MLPs) and `text_refine` (the
text-refinement blocks). For anything else use `adapter.target_include` / `target_exclude` with
fnmatch globs against the dotted module path.

## VRAM and speed

Measured on an RTX 3000 Ada (8 GB), 1024², bf16, `activation_checkpointing = true`:

| Setup | Peak VRAM training | Peak VRAM with a 1024² preview | Speed |
|-------|--------------------|--------------------------------|-------|
| LoRA rank 32 | ≈ 2.4 GB | ≈ 4.0 GB | ~1.6 s/it |
| Full finetune, `adamw8bit` | ≈ 4.3 GB | ≈ 5.2 GB | ~2 s/it |

No block swap and no quantization are needed, and neither is supported. For the general OOM
playbook see the [VRAM ladder](training-loop-and-eval.md#if-it-doesnt-fit-the-vram-ladder).

## Text-embedding cache

Captions are encoded once (frozen Gemma3-270M, run in float32) and cached together with their
per-token emphasis weights; run it ahead of training with:

```bash
rengu cache --config my.toml
```

Each caption is cached at its own token length. Replacing the text-encoder file changes the
cache key, so captions are re-encoded.

## Previews

Previews use Euler sampling over the shift-3 "simple" schedule (the same one the official
ComfyUI node uses). Defaults: **`num_inference_steps = 30`**, **`guidance_scale = 4.0`**,
**`negative_prompt = "oldest, low quality"`**, **`guidance_mode = "alternate"`**, 1024×1024.
`guidance_scale = 1` disables guidance (one forward pass per step). Previews run the DiT
resident; `preview_blocks_to_swap` is ignored.

```toml
[preview]
width = 1024
height = 1024
num_inference_steps = 30
guidance_scale = 4.0
guidance_mode = "alternate"       # "alternate" | "cfg" | "path_drop"
negative_prompt = "oldest, low quality"
prompts = [
  "newest, masterpiece, 1girl, solo, (fennec ears:1.3), long blonde wavy hair, blue eyes, smile, forest, sunlight",
]
```

`preview.guidance_mode` decides how the unconditional pass is built:

- **`alternate`** (default, the reference recommendation): classic CFG on even steps, path-drop
  guidance on odd steps.
- **`cfg`**: classifier-free guidance on every step, using the negative prompt on the full model.
- **`path_drop`**: on every step the unconditional pass is the negative prompt *with the sparse
  middle blocks skipped* (SPRINT path-drop guidance).

Any other value is rejected at validation. Size: the model was trained on roughly 1-megapixel
buckets (from 512×2048 to 2048×512), so keep previews near that area. See
[Training previews](previews.md) for the shared `[preview]` schema.

## Export formats

- **LoRA** writes `adapter_model.safetensors` with ComfyUI keys
  `diffusion_model.<module path>.lora_A.weight` / `.lora_B.weight` plus `.alpha`. In ComfyUI
  load it with **`LoraLoaderModelOnly`**; Nanosaur2 needs the official **`nanosaur2_support`**
  custom node from the
  [Hugging Face repo](https://huggingface.co/well9472/Nanosaur2-670M).
- **Full finetune** writes `nanosaur2_diffusion_model.safetensors` in the official release layout,
  usable wherever the original file is.

## Validate config

```bash
./rengu validate --config my.toml
```

## Smoke tests

With the three files set in `.env` (`RENGU_NANOSAUR2_TRANSFORMER_PATH`,
`RENGU_NANOSAUR2_VAE_PATH`, `RENGU_NANOSAUR2_TEXT_ENCODER_PATH`, see `.env.example`):

```bash
scripts/run_model_smoke.sh nanosaur2        # LoRA, 30 steps + a 1024 px preview
scripts/run_model_smoke.sh nanosaur2_full   # full finetune, 10 steps, 8-bit AdamW
```

Without weights, `python scripts/smoke_mini.py --model nanosaur2` trains a few steps on a tiny
random model on the CPU. See [Smoke tests](../developer/smoke-tests.md).
