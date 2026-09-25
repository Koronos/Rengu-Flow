# Training Qwen-Image 2.1

Qwen-Image 2.1 (`Qwen/Qwen-Image-2.1`, Qwen Research License) is a **7B single-stream DiT**
text-to-image model. It is **not** the classic Qwen-Image (dual-stream, 60 layers): 2.1 runs one
joint `[prompt | image]` sequence through 32 blocks with a block-causal attention mask, one
modulation shared by every block, a **Qwen3-VL-8B** text encoder and a new **RGBA VAE** with
16× spatial compression and 64 latent channels. In TOML always use:

- `type = "qwen_image21"`

Rengu trains it **text-to-image** and **image editing** (image-conditioned, see
[Image editing](#image-editing-edit-training)), also mixed in one run: LoRA, LoKr, every LyCORIS
type, and full finetune of the DiT. The VAE and the text encoder are always frozen.

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
| **`shift`** | Fixed timestep shift (> 0) overriding the dynamic one (below). | number | unset (dynamic) |
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

Captions are encoded once with Qwen3-VL-8B (only its text decoder is loaded for text-to-image
captions; the vision tower is added only for edit captions, see
[Image editing](#image-editing-edit-training)) and cached; training steps never touch the
encoder. The encoder follows the
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

## Image editing (edit training)

Qwen-Image 2.1 is also an instruction-based editor: given one or more **condition images** and an
instruction ("make it snowy", "put the hat from picture 2 on the dog in picture 1"), it generates
the edited image. Rengu trains this the way the reference pipeline samples it:

- the instruction and the condition images are encoded **together** by Qwen3-VL-8B (its vision
  tower reads the images; each one becomes a block of vision slots inside the prompt, in the
  reference's `<image1>…<image2>…` template, RGBA images composited over white);
- each condition image is VAE-encoded (clean, the distribution's mode) and its latents fill that
  block of the DiT sequence: `[prompt with condition images 1 … N | target]`;
- **noise, the loss and the timestep shift involve the target only** (the shift uses the
  target's own token count, like the reference); the prompt and the condition blocks are
  modulated at `t = 0`.

### Dataset format

An edit folder is a normal `[[directory]]` plus `control_path`: targets (the edited results) with
a `.txt` holding the **instruction**, and a sibling folder of condition images paired by file
stem:

```text
edit_set/
  targets/    house.png  house.txt ("make it snowy")   dog.png  dog.txt
  controls/   house.png                                dog_0.png  dog_1.png
```

`stem.<ext>` gives one condition image; `stem_0.<ext>, stem_1.<ext>, …` give several, in order
(contiguous from 0). Pairing is strict: a target without a valid control set is an error.

No instructions yet? [`rengu prep edit_caption`](dataset-prep.md#edit-instructions--rengu-prep-edit_caption)
has a VLM write one per pair onto line 1 of each target's `.txt` (same pairing rules; unpaired
targets are reported, not fatal). Review them before training (Studio → Tag editor shows each
target's line 1): VLMs hallucinate differences between two images.

```toml
[[directory]]
path = "edit_set/targets"
control_path = "edit_set/controls"
# control_resolution = 1024   # optional: side of each condition image's target area

[[directory]]                  # optional: text-to-image data in the same run
path = "t2i_images"
```

Each condition image **keeps its own aspect ratio** (it is not cropped to the target's bucket):
it is resized to the area `control_resolution²`, floored to 32 px. The default is the target's
bucket resolution. Image augmentations cannot be combined with `control_path`. Full details
(naming rules, batching, caching): [Dataset config — Control images](dataset-config.md#control-images-control_path-edit-training).

Two limits apply to every edit target. Both are checked right after the metadata is built, before
any image or caption is encoded, and the error lists the first offending targets:

| Limit | Why | How to fix |
|-------|-----|------------|
| Each condition image at least **256×256 in area** after resizing | The Qwen3-VL vision encoder upsamples anything smaller, and the image then no longer lines up with its VAE latents. | Raise `control_resolution` (the web UI does not offer less than 256). Each side is floored to 32 px, so a non-square condition image needs somewhat more: a 2:1 image at 256 ends up 352×160, below the minimum; at 288 it is 384×192. |
| At most **10 condition images** per target | A limit of this implementation, not of the model: the layout of the condition blocks is carried in the shape of a tensor passed between the DiT layers (two dimensions per condition image), and PyTorch allows at most 25 dimensions per tensor. | Use fewer condition images per target. |

In practice VRAM runs out before the 10-image limit: each 1024² condition image adds 4096 tokens to
the DiT sequence (see below), so a target with ten of them is an 11× longer sequence than
text-to-image.

### Mixed text-to-image + edit runs

Folders with and without `control_path` can share one dataset: batches never mix the two kinds
(nor edit rows with different condition counts or sizes), so each step is either a text-to-image
step or an edit step, weighted by the folders' `num_repeats`. Keeping some text-to-image data in
an edit run (and some edit pairs in a style run) is the simplest guard against losing the other
skill: an adapter trained only on edits drifts the shared weights that text-to-image also uses.

### Cost of the text encoder with vision

Edit captions add the Qwen3-VL vision tower (~0.6B params, ~1.2 GB bf16) and the condition
images' vision tokens to the encoder pass: a 1024×1024 condition image is 1024 extra prompt
tokens (`(W/32)·(H/32)`). The vision tower is loaded the first time a caption with images is
encoded and stays resident next to the (streamed) text decoder until caching ends; text-to-image
captions never load it. Measured on an RTX 3000 Ada laptop GPU (8 GB), streamed decoder, one
1024×1024 condition image: **~2.7 s per edit caption vs ~1.2 s per text-to-image caption**
once loaded (the first one also pays ~21-26 s to load the encoder and the vision tower), and
**3.25 GB VRAM peak vs 2.6 GB**. Host RAM while caching: ~20 GB (the pinned
decoder layers plus the vision tower). A lower `control_resolution` (e.g. 768) cuts the extra
tokens roughly in half.

In training the condition blocks lengthen the DiT sequence the same way (a 1024² condition image
adds 4096 latent tokens to a 1024² target's 4096), so an edit step costs roughly twice a
text-to-image step at the same resolution in attention/activations; keep `blocks_to_swap` and
`activation_checkpointing` at least as aggressive as for text-to-image.

### Rank and steps

Starting points, not tuned results: **LoRA/LoKr rank 16–32**, `lr = 1e-4` (AdamW), a few
thousand steps for a focused edit (one kind of transformation), and preview both an edit prompt
and a text-to-image prompt (see below) to catch the adapter overwriting the other skill early.
Raise the rank only if a varied edit set underfits; a higher rank with long training on edits
alone is the fastest way to degrade text-to-image. For a mixed run, 20–30 % text-to-image steps
(via `num_repeats`) is a reasonable first split.

### Edit previews

A preview prompt can carry `control_images` (paths, in order); the prompt is then the
instruction, and the preview encodes the images exactly like training (same resize helper,
vision tower, clean latents) and keeps them in the KV-cache prefix:

```toml
[[preview.prompts]]
name = "snow"
prompt = "make it snowy"
control_images = ["edit_set/controls/house.png"]
```

The output takes the last condition image's aspect ratio at the `width × height` area;
`preview.control_resolution` (default: the side of `width × height`) sizes the condition images.
See [Previews — Edit previews](previews.md#edit-previews-qwen-image-21). Measured (8 GB RTX 3000
Ada, fp8 base, `preview_blocks_to_swap = 32`, one 1024×1024 condition image, 1024×1024 output,
28 steps): ~114 s with the prompt already encoded (the edit sequence is twice a text-to-image one), 4.4 GB VRAM peak allocated.

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
[`minimal_config_qwen_image21_edit_lora.toml`](../../examples/minimal_config_qwen_image21_edit_lora.toml) (edit + text-to-image, with an edit preview),
[`minimal_config_qwen_image21_lokr.toml`](../../examples/minimal_config_qwen_image21_lokr.toml),
[`minimal_config_qwen_image21_finetune.toml`](../../examples/minimal_config_qwen_image21_finetune.toml),
datasets [`minimal_qwen_image21_dataset.toml`](../../examples/minimal_qwen_image21_dataset.toml) and
[`minimal_qwen_image21_edit_dataset.toml`](../../examples/minimal_qwen_image21_edit_dataset.toml) (edit pairs + text-to-image).
