# Training Qwen-Image 2.1

Qwen-Image 2.1 (`Qwen/Qwen-Image-2.1`, Qwen Research License) is a **7B single-stream DiT**
text-to-image model. It is **not** the classic Qwen-Image (dual-stream, 60 layers): 2.1 runs one
joint `[prompt | image]` sequence through 32 blocks with a block-causal attention mask, one
modulation shared by every block, a **Qwen3-VL-8B** text encoder and a new **RGBA VAE** with
16× spatial compression and 64 latent channels. In TOML always use:

- `type = "qwen_image21"`

Rengu trains it **text-to-image**: LoRA, LoKr, every LyCORIS type, and full finetune of the
DiT. The VAE and the text encoder are always frozen.

> **Not supported:** image-conditioned (edit) training — the reference pipeline's condition
> images, which the text encoder reads as vision context and the DiT receives as extra latent
> blocks. Every training sample is a caption plus its target image.

## Getting the checkpoint

Download the official diffusers release and point `model.diffusers_path` at it:

```bash
huggingface-cli download Qwen/Qwen-Image-2.1 --local-dir /path/to/Qwen-Image-2.1
```

The folder holds `transformer/` (~14.2 GB bf16, 2 shards), `text_encoder/` (Qwen3-VL-8B,
~17.5 GB, 4 shards), `vae/` (~1.35 GB) and `processor/`. A Hugging Face cache snapshot
(`~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/<sha>/`) works as-is.
Nothing is ever downloaded automatically; rengu never resolves repo ids.

Each component resolves to `<diffusers_path>/<subfolder>`; a per-component path overrides it:

| Component | Override key | Accepted |
|-----------|--------------|----------|
| DiT | `model.transformer_path` | the diffusers `transformer/` folder, or one `.safetensors` in diffusers keys or ComfyUI's layout (`Comfy-Org/Qwen-Image-2.1` → `diffusion_models/qwen_image_2.1_bf16.safetensors`; its fused `img_mlp.gate_up` is split on load) |
| Text encoder | `model.text_encoder_path` | the transformers `text_encoder/` folder, or ComfyUI's `text_encoders/qwen3vl_8b_bf16.safetensors` |
| VAE | `model.vae_path` | the diffusers `vae/` folder, or a diffusers-layout `.safetensors` |
| Tokenizer | `model.processor_path` | the `processor/` folder |

- Pre-quantized files (`*_int8_convrot`, `*_w4a8`, fp8 "scaled") cannot be trained and are
  refused with an error; train from bf16 and use `model.transformer_fp8_matmul` /
  `model.transformer_4bit` for VRAM instead.
- ComfyUI's VAE file (`vae/qwen_image_2.1_vae_bf16.safetensors`) uses the original key layout
  and is **not** converted — use the diffusers `vae/` folder.
- Without `processor/`, the Qwen3-VL tokenizer bundled with rengu is used (it tokenizes the
  Qwen-Image 2.1 prompt template identically).

## `[model]` fields

| Config key | What it is | Required | Default |
|------------|------------|----------|---------|
| **`type`** | `"qwen_image21"`. | Yes | — |
| **`dtype`** | Load/compute dtype for the VAE, text encoder, adapters and (unless overridden) the DiT. | Yes | — |
| **`diffusers_path`** | The Qwen-Image-2.1 download (see above). | Unless all three of `transformer_path` / `vae_path` / `text_encoder_path` are set | — |
| **`transformer_path`**, **`vae_path`**, **`text_encoder_path`**, **`processor_path`** | Per-component overrides (see the table above). | No | `<diffusers_path>/<component>` |

| Optional key | Purpose | Values | Default |
|--------------|---------|--------|---------|
| **`text_encoder_offload`** | Where the 17.5 GB text encoder runs while captions are cached (see [Text encoder and the embedding cache](#text-encoder-and-the-embedding-cache)). | `auto`, `stream`, `none` | `auto` |
| **`transformer_fp8_matmul`** | Store the frozen DiT's block linears as tensorwise-scaled e4m3 fp8 (1 byte/param): ~14.2 → ~7.3 GB. Adapter training only; needs an sm89+ GPU (RTX 40xx / Ada). | `true` / `false` | `false` |
| **`fp8_grad_mode`** | Precision of the input-gradient GEMM through the fp8 base. `fp8` is faster, `bf16` keeps the clean gradient. | `bf16`, `fp8` | `bf16` |
| **`transformer_4bit`** | Store the frozen DiT's block linears as 4-bit NF4 (bitsandbytes), ~4 GB. Adapter training only; pair with `lokr` (LyCORIS types refuse a quantized base). Mutually exclusive with `transformer_fp8_matmul`. | `true` / `false` | `false` |
| **`transformer_dtype`** | DiT load dtype. | a dtype | `dtype` |
| **`shift`** | Fixed timestep shift overriding the dynamic one (below). | number | unset (dynamic) |
| **`timestep_sample_method`** | Training timestep distribution. | `logit_normal`, `uniform` | `logit_normal` |
| **`sigmoid_scale`** | Scale on the logit-normal sample before the sigmoid. | number | `1.0` |

`cache_text_embeddings` is always on for this model (setting it to `false` is an error): the
8B encoder cannot sit inside the training graph.

### Minimal `[model]` example

```toml
[model]
type = "qwen_image21"
dtype = "bfloat16"
diffusers_path = "/path/to/Qwen-Image-2.1"
```

### Training objective

Rectified flow, matching the reference sampler: `x_t = (1 - t)·x0 + t·noise`, the DiT receives
`t ∈ [0, 1]` and predicts the velocity `noise - x0`; the loss covers the target-image tokens
only. Training timesteps get the reference scheduler's **resolution-aware exponential shift**:
`mu` interpolates linearly from 0.5 at 256 latent tokens to 0.9 at 8192 (a 1024×1024 image is a
64×64 = 4096-token latent grid, `mu ≈ 0.70`). A fixed `model.shift` replaces it.

Dataset images are RGB; they are fed to the RGBA VAE with a fully opaque alpha channel. Bucket
sides are rounded to multiples of **32** px (16× VAE, and the latent grid must be even).

## Modes

### LoRA

```toml
[adapter]
type = "lora"
rank = 16
```

### LoKr

```toml
[adapter]
type = "lokr"
rank = 6
factor = -1
```

`lokr` is the quantization-aware adapter: it composes with both `transformer_fp8_matmul` and
`transformer_4bit`.

### LyCORIS networks

Any `lycoris_*` type (`lycoris_locon`, `lycoris_loha`, `lycoris_lokr`, `lycoris_dylora`,
`lycoris_glora`, `lycoris_diag_oft`, `lycoris_boft`; requires the `lycoris` extra). They reject
a 4-bit base; see [Training Cosmos Predict2 / Anima](training-cosmos-predict2-lora-lokr-finetune.md#lycoris-networks)
for the per-type fields.

### Full finetune

Omit `[adapter]`. Every DiT parameter trains (the shared modulation, projections, timestep MLP
and all 32 blocks); `transformer_fp8_matmul` / `transformer_4bit` do not apply. The bf16 weights
alone are ~14 GB before gradients and optimizer state: plan for a **48 GB+** card, or
`blocks_to_swap` + `optimizer.gradient_release = true` (deepspeed engine) on 24 GB. Full
finetune does not fit on 8–16 GB cards. Exports are a diffusers `transformer/` folder
(`config.json` + `diffusion_pytorch_model.safetensors`) loadable by diffusers'
`QwenImage21Pipeline`.

### Adapter targets

Adapters attach to **every Linear of the DiT** by default. Narrow them with named layer groups
(`adapter.layer_groups`) and/or globs (`adapter.target_include` / `target_exclude`) over the
module paths:

| Group | Modules |
|-------|---------|
| `attention` | `transformer_blocks.*.attn.*` (`to_q`, `to_k`, `to_v`, `to_out.0`) |
| `feedforward` | `transformer_blocks.*.img_mlp.*` (SwiGLU `gate_layer`, `proj`, `out`) |
| `modulation` | `modulation.*` (the one projection every block reads its scales/gates from), `time_text_embed.*`, `norm_out.*` |
| `text_projection` | `txt_in.*` |
| `image_in_out` | `img_in`, `proj_out` |

```toml
[adapter]
type = "lora"
rank = 16
layer_groups = ["attention", "feedforward"]
```

## VRAM guidance

| Card | Recommended setup |
|------|-------------------|
| **8 GB** | LoRA / LoKr with `model.transformer_fp8_matmul = true` (sm89+; else `transformer_4bit = true` + `lokr`), `blocks_to_swap = 24`, `activation_checkpointing = true`, `micro_batch_size_per_gpu = 1`. Previews with `preview_blocks_to_swap = 32`. ~24 GB of free host RAM is needed on top (the swapped fp8 blocks plus, while caching, the streamed text encoder). |
| **16 GB** | Adapters with the fp8 base and a smaller `blocks_to_swap` (e.g. 8), or the bf16 base with a larger one. |
| **24 GB** | Adapters on the fp8 base without block swap; bf16 base with moderate block swap. Full finetune only with block swap + `optimizer.gradient_release = true`. |
| **48 GB+** | Adapters on the bf16 base; full finetune. |

```toml
# 8 GB: fp8 frozen base + block swap
blocks_to_swap = 24
activation_checkpointing = true

[model]
type = "qwen_image21"
dtype = "bfloat16"
diffusers_path = "/path/to/Qwen-Image-2.1"
transformer_fp8_matmul = true

[adapter]
type = "lora"
rank = 16
```

For the general OOM playbook see the
[VRAM ladder](training-loop-and-eval.md#if-it-doesnt-fit-the-vram-ladder) and
[VRAM optimization](../developer/vram-optimization.md). `[tread]` token routing is not
supported for this model.

## Text encoder and the embedding cache

Captions are encoded once with Qwen3-VL-8B (only its text decoder is loaded; t2i never runs the
vision tower) and cached; training steps never touch the encoder. The encoder follows the
reference pipeline exactly: the raw t2i chat template with the system prompt *"Comprehend and
analyze the provided prompt."*, left padding within an encoding batch, the hidden state that
enters the decoder's final RMSNorm, and the 14 system-prompt tokens dropped. Each cached caption
is its real token count × 4096 values (~8 KB per token in bf16).

The encoder is **loaded lazily** — only when captions still need encoding — and unloaded after
caching, so a run with a warm cache never reads its 17.5 GB. `model.text_encoder_offload`
decides how it runs on the GPU:

- **`auto`** (default): if the encoder does not fit in free VRAM (with ~2 GB headroom), its 36
  decoder layers stay in pinned host RAM and stream to the GPU one at a time during each forward
  (~1.5 GB VRAM for the embeddings plus one layer). Otherwise it is loaded whole.
- **`stream`**: always stream.
- **`none`**: always load it whole (~17.5 GB VRAM).

Streaming is bound by host-to-GPU bandwidth (the whole encoder crosses PCIe once per encoding
batch), so raise `caching_batch_size` to encode more captions per pass.

## Previews

Previews follow the reference sampler: sigmas `linspace(1, 1/n, n)` with the same
resolution-aware exponential shift, stretched so the last step lands on 0.02 (the checkpoint's
`shift_terminal`), Euler steps, and a prefix KV cache (the prompt's keys/values are computed on
the first step and reused, since text tokens are modulated at `t = 0`). Defaults:

| Preview key | Default | Notes |
|-------------|---------|-------|
| `num_inference_steps` | `28` | The reference uses 40. |
| `guidance_scale` | `1.0` | The model is meant to be sampled without CFG. Above `1.0`, CFG runs with `negative_prompt` (an empty negative is encoded as `" "`, like the reference) as `neg + g·(pos − neg)`. |
| `width` / `height` | `1024` | Rounded to multiples of 32. |
| `preview_blocks_to_swap` | `0` | Set `32` on 8 GB cards to stream the DiT blocks during the preview. |
| `preview_offload_text_encoder` | `true` | Moves the encoder off the GPU once the prompts are encoded. |

Preview prompts are encoded once and memoized, so the text encoder is loaded for the first
preview only (and again only for a prompt never seen before); it is unloaded after each preview
instead of being parked in host RAM. Above 512 px the VAE decode runs in 512 px tiles with a
blended 64 px overlap: an untiled 1024×1024 decode needs ~6.6 GB of activations, tiled ~1.7 GB.
The decoded RGBA image is shown composited over white. Previews require `pipeline_stages = 1`.

Measured on an RTX 3000 Ada laptop GPU (8 GB), fp8 base, `preview_blocks_to_swap = 32`,
1024×1024, 28 steps: first preview ~145 s including ~30 s to load and stream the text encoder
(1.8 GB VRAM peak while encoding).

## Export formats

**Adapters** are saved as `adapter_model.safetensors` with the `transformer.` key prefix over the
diffusers module names (e.g. `transformer.transformer_blocks.0.attn.to_q.lora_A.weight`,
`transformer.transformer_blocks.0.img_mlp.gate_layer.lora_B.weight`). This is the form ComfyUI's
Qwen-Image LoRA loader maps for 2.1 — including the `img_mlp.gate_layer` / `img_mlp.proj`
halves of its fused `img_mlp.gate_up` weight — and the one diffusers' Qwen-Image LoRA loader
reads. (A `diffusion_model.` prefix would silently drop every MLP adapter in ComfyUI, whose model
only has the fused key.) `adapter.init_from_existing` accepts the same format.

**Full finetune** exports a diffusers `transformer/` folder (see [Full finetune](#full-finetune)).

## Validate config

```bash
./rengu validate --config examples/minimal_config_qwen_image21_lora.toml
```

Example configs: [`minimal_config_qwen_image21_lora.toml`](../../examples/minimal_config_qwen_image21_lora.toml),
[`minimal_config_qwen_image21_lokr.toml`](../../examples/minimal_config_qwen_image21_lokr.toml),
[`minimal_config_qwen_image21_finetune.toml`](../../examples/minimal_config_qwen_image21_finetune.toml),
dataset [`minimal_qwen_image21_dataset.toml`](../../examples/minimal_qwen_image21_dataset.toml).
