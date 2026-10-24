# Training MOSS-Image

MOSS-Image (OpenMOSS, **Apache-2.0**) is a **5B single-stream DiT** text-to-image model: one joint
`[image tokens ; text tokens]` sequence through 42 blocks with a modulation shared by every block,
a **Qwen3.5-9B** text encoder (the last hidden state of every token of the chat-templated prompt)
and the **FLUX.2 VAE** (8x spatial compression plus a 2x2 patchify = 16x, 128 latent channels). Two
checkpoints share the same architecture and code:

- [`OpenMOSS-Team/MOSS-Image-Base`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) — the
  pretrained model, **the one to train** (the official user guide says Base is meant for LoRA and
  finetuning);
- [`OpenMOSS-Team/MOSS-Image-SFT`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) — the
  post-trained model for generation; it loads with the same settings.

In TOML always use:

- `type = "moss_image"`

Rengu trains it **text-to-image**: LoRA, LoKr, every LyCORIS type, and full finetune of the DiT. The
VAE and the text encoder are always frozen (the authors advise against training the encoder).
Edit / reference-image training, text-encoder training and the few-step distilled variants are not
supported.

## Getting the checkpoint

MOSS-Image trains from **three local files**, one per component — the same pattern as Krea 2 and
Qwen-Image 2.1. Download the bf16 files from
[Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) (public, no login) and point one
config key at each:

| Component | Config key | `.env` variable | Direct download | Size | Notes |
|-----------|------------|-----------------|-----------------|------|-------|
| DiT (the model you train) | `model.transformer_path` | `RENGU_MOSS_IMAGE_TRANSFORMER_PATH` | [moss_image_base_bf16.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_base_bf16.safetensors) | 10.16 GB | ComfyUI layout: each block's `q` / `k` / `v` is fused into `proj.qkv` and split back into the official names on load. |
| Image VAE | `model.vae_path` | `RENGU_MOSS_IMAGE_VAE_PATH` | [flux2-vae_small.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae_small.safetensors) | 0.25 GB | FLUX.2 VAE with the small decoder (what the OpenMOSS `vae/` folder holds). [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae.safetensors) (0.34 GB, full decoder, original key layout converted on load) also works, but it is not bit-identical: the release `vae/` and `flux2-vae_small` are the bf16-rounded version of it, batch-norm statistics included, so latents differ by that rounding. Use `flux2-vae_small` to match the release. |
| Text encoder (Qwen3.5-9B) | `model.text_encoder_path` | `RENGU_MOSS_IMAGE_TEXT_ENCODER_PATH` | [qwen3.5_9b_bf16.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/text_encoders/qwen3.5_9b_bf16.safetensors) | 19.3 GB | Only the text decoder is read; the vision tower, LM head and MTP head in the file are skipped. |

```toml
[model]
type = "moss_image"
dtype = "bfloat16"
transformer_path = "/path/to/moss_image_base_bf16.safetensors"
vae_path = "/path/to/flux2-vae_small.safetensors"
text_encoder_path = "/path/to/qwen3.5_9b_bf16.safetensors"
```

The Qwen3.5 tokenizer and chat template are bundled with rengu (copied from the Base repository) —
no download or path needed unless you override them with `model.tokenizer_path`.

**Refused with an error.** The pre-quantized files cannot be trained or used to encode captions; rengu
stops with a message:

- `moss_image_*_int8_convrot`, `qwen3.5_9b_int8_convrot` and `qwen3.5_9b_w4a8_gptq`: train from bf16 and
  use `model.transformer_fp8_matmul` / `model.transformer_4bit` for VRAM instead.

**Not the training base.** These load, but they are not what you want to start from:

- `moss_image_sft_bf16` trains, but it is the post-trained model; start from Base unless you want to
  continue from the SFT behavior.
- `loras/moss_image_color_depth_control` is a ready-made LoRA, not a base model.

**Full diffusers folder (alternative):** point `model.diffusers_path` at the official
[OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) download
instead of setting the three files:

```bash
huggingface-cli download OpenMOSS-Team/MOSS-Image-Base --local-dir /path/to/MOSS-Image-Base
```

It holds `transformer/` (10.16 GB bf16, 3 shards), `text_encoder/` (Qwen3.5-9B, 4 shards, ~16.8 GB),
`vae/` (0.25 GB, fp32) and `tokenizer/` (27.2 GB in total with the repo's other files). A Hugging
Face cache snapshot (`~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/<sha>/`)
works as-is. Each component left empty resolves to `<diffusers_path>/<transformer|vae|text_encoder|tokenizer>`;
a `transformer_path` / `vae_path` / `text_encoder_path` set alongside it always overrides that one
component. Rengu never downloads anything and never resolves a repo id.

Every path can also come from the environment (`RENGU_MOSS_IMAGE_<KEY>`, see `.env.example`).

## `[model]` fields

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| `transformer_path` | The DiT you train. | File or the diffusers `transformer/` folder. | required unless `diffusers_path` |
| `vae_path` | The FLUX.2 VAE (encodes your images into the cache; decodes previews). | File or the `vae/` folder. | required unless `diffusers_path` |
| `text_encoder_path` | The Qwen3.5-9B encoder (captions are encoded once into the cache). | File or the `text_encoder/` folder. | required unless `diffusers_path` |
| `diffusers_path` | A whole diffusers download that fills any component left empty. | Folder. | unset |
| `tokenizer_path` | Tokenizer override. | Folder with the Qwen3.5 tokenizer files. | `<diffusers_path>/tokenizer`, else the bundled one |
| `dtype` | Precision of the DiT and the text encoder. | `bfloat16`. **`float16` is refused:** a few joint-stream activations exceed 65504 and overflow. | `bfloat16` |
| `text_encoder_offload` | How the 17 GB encoder runs while captions are cached. | `auto` streams its 32 layers from pinned host RAM when it does not fit in free VRAM; `stream` always; `none` loads it whole on the GPU. | `auto` |
| `transformer_fp8_matmul` | Frozen base in tensorwise-scaled e4m3 (~10.2 -> ~5.2 GB). Adapter training only, sm89+ GPU. | `true` / `false` | `false` |
| `transformer_4bit` | Frozen base in NF4 (~3 GB). Adapter training only; use LoKr. | `true` / `false` | `false` |
| `fp8_grad_mode` | Input-gradient GEMM precision through the fp8 base. | `bf16` / `fp8` | `bf16` |
| `transformer_dtype` | DiT load precision. | A dtype. | `dtype` |
| `timestep_sample_method` | How training sigmas are sampled. | `moss`, `logit_normal`, `uniform` | `moss` |
| `timestep_mu`, `timestep_std`, `timestep_cutoff`, `timestep_uniform_mix` | Parameters of the `moss` sampler. | numbers (`std > 0`, `0 < cutoff < 1`, `0 <= mix <= 1`) | `-1.09`, `0.95`, `0.95`, `0.01` |
| `sigmoid_scale` | Scale of the `logit_normal` sample before the sigmoid. | number | `1.0` |
| `shift` | Fixed timestep shift on top of the sampled sigmas. | number `> 0` | unset (no shift) |

## Training objective

This is the technical report's final-stage recipe (MOSS-Image technical report, Table 6 and
Eqs. 3-4), not a generic flow-matching default:

- sigma = 1 is pure noise: `x = (1 - sigma) x0 + sigma eps`, the target is the velocity `eps - x0`,
  and the model's time input is `1000 * sigma`.
- **Sigma sampler** (`timestep_sample_method = "moss"`): sigma on the grid `j / 1000`, `j = 1..1000`,
  with `p_j = (1 - rho) (F(a j / T) - F(a (j - 1) / T)) / F(a) + rho / T`, where `F` is the CDF of
  `S = 1 - sigmoid(Z)`, `Z ~ N(mu, s^2)`. Defaults `mu = -1.09`, `s = 0.95`, cutoff `a = 0.95`,
  `rho = 0.01` (the median sigma is about 0.78). There is **no shift** while training: the report
  trains its final stage at shift 1. The fixed shift 5.0 of the reference sampler is for sampling
  (previews) only.
- **Loss**: plain velocity MSE, no sigma / SNR weighting; text tokens carry no loss. The report
  averages over all valid image tokens of a token-packed batch (Eq. 3); rengu averages over the
  elements of each micro-batch, which is the same for equal-size latents (one bucket per batch) and
  differs slightly across gradient-accumulation steps and mixed-size packing.
- **Caption dropout** is the framework default (the dataset's `uncond_fraction`); the report gives
  no number. The dropped caption is the encoded empty prompt, which is also the reference's
  unconditional input. KohakuMoss uses 10% text dropout as a community setting.
- The report's structured-prompt augmentation (random dropping of JSON fields, shuffled items) is
  **not** applied; feed the captions you want trained.

## Modes

The DiT's linears are named exactly as in the official checkpoint (`layers.N.proj.q`, ...), so the
adapters export with the official module paths.

Settings below are the **KohakuMoss community recipe** (KohakuBlueleaf/KohakuMossTrainer defaults),
not an official recipe: the technical report only gives the pretraining optimizer.

### LoRA

```toml
[adapter]
type = "lora"
rank = 32     # rengu sets alpha = rank (the community recipe's alpha 32)

[optimizer]
type = "adamw"
lr = 1e-4
weight_decay = 0.01
```

Community recipe: scope = attention + feed-forward of the 42 blocks (the default here), LR `1e-4`,
AdamW weight decay 0.01, cosine schedule with 100 warmup steps (`warmup_steps = 100`), gradient clip
1.0 (rengu's default), about 4000 steps at an area of 1024x1024. See
`examples/minimal_config_moss_image_lora.toml`.

### LoKr

```toml
[adapter]
type = "lokr"
rank = 10000      # >= the layer width: both Kronecker factors are stored whole
factor = 8
full_matrix = true

[optimizer]
type = "adamw"
lr = 2e-4
```

Community recipe: factor 8, full matrix, LR `2e-4` (with both factors stored whole the scale is 1; rengu otherwise always sets alpha = rank). LoKr is also the adapter that composes
with the quantized base (`transformer_4bit`). See `examples/minimal_config_moss_image_lokr.toml`.

### LyCORIS networks

Every LyCORIS type (`lycoris_locon`, `lycoris_loha`, `lycoris_lokr`, ...) is available on the same
scope; they refuse a quantized base (use LoKr there).

### Full finetune

No `[adapter]` section. The 5B DiT takes ~10 GB for the bf16 weights alone, plus gradients and
optimizer state; `blocks_to_swap` and `optimizer.gradient_release = true` (deepspeed engine) trade
speed for memory (see [VRAM guidance](#vram-guidance)). Reference values (the report's *pretraining* optimizer, Table 7): LR
`5e-5`, AdamW betas `[0.9, 0.95]`, weight decay 0.01, clip 1.0; a small dataset wants a lower LR to
start (for example `1e-5`). See `examples/minimal_config_moss_image_finetune.toml`. The export is
the official diffusers transformer folder (see [Export formats](#export-formats)).

### Precision

Train in **bfloat16**. The reference runs the DiT and the encoder in bf16; only the time sinusoid
(fp64, then fp32), the RoPE phases, CFG and the Euler update are fp32. The time MLP and the modulation
linears run at the weight dtype (their output is then used in fp32 by the blocks), and rengu does the
same. fp16 overflows (a few joint-stream activations exceed 65504) and is refused by config validation.

### Adapter targets

The default adapter scope is the **attention (`q`, `k`, `v`, `o`) and feed-forward (`ffn_up_gate`,
`ffn_down`) linears of the 42 blocks** (`layers.N.proj.*`), what KohakuMoss defaults to and what
ComfyUI maps. `adapter.layer_groups` (named groups), `target_include` and `target_exclude` (globs
over dotted module paths) widen or narrow it; setting either of the first two replaces the default.

| Group | Modules |
|-------|---------|
| `attention` | `layers.*.proj.{q,k,v,o}` |
| `feedforward` | `layers.*.proj.{ffn_up_gate,ffn_down}` |
| `text_refiner` | `text_embedder.blocks.*` (the 2 text-refiner blocks) |
| `text_projection` | `text_embedder.proj` |
| `modulation` | `shared_block_modulation`, `video_head_modulation`, `time_embedding.*` |
| `image_in_out` | `video_embedder`, `final_linear_video` |

## VRAM guidance

The DiT is 10.2 GB in bf16 (5.2 GB with `transformer_fp8_matmul`, ~3 GB with NF4). Measured on an RTX 3000
Ada Laptop GPU (8 GB, sm89), native Windows, accelerate engine, with the Comfy-Org bf16 files
(`moss_image_base_bf16`, `qwen3.5_9b_bf16`, `flux2-vae_small`), 512x512 training images, micro-batch 1, fp8
base, `blocks_to_swap = 24` and activation checkpointing; peak = torch `cuda_peak`:

| Setup | Peak VRAM | Speed | Result |
|-------|-----------|-------|--------|
| LoRA rank 32, lr 1e-4 | 4.40 GB | ~4-7 s/it (3.7-4.1 s/it on a quiet host, 6.9 s/it p50 on a loaded one) | loss 0.82 -> 0.51 in 10 steps; export check OK (756 tensors, 252 modules under `diffusion_model.`) |
| LoKr rank 6, factor -1, lr 3e-5 | 3.46 GB | ~7 s/it | loss 0.84 -> 0.61 in 10 steps; export check OK (1008 tensors) |
| Preview, 512x512, 20 steps, CFG 3, `preview_blocks_to_swap = 24` | not recorded | ~121 s per image | a coherent image matching the prompt |
| LoRA, bf16 base, no swap | not measured | | |
| Full finetune | not measured | | Needs a much bigger card; on Windows block swap with full training is unsupported. |

Caching 13 captions with the Qwen3.5-9B encoder streamed from the 19.3 GB file (`text_encoder_offload =
auto`) took 57 s with the torch Gated DeltaNet fallback (no `flash-linear-attention` / `causal-conv1d`).

**8 GB recipe.** fp8 base (`transformer_fp8_matmul = true`), `blocks_to_swap = 24`, activation
checkpointing, and previews at 512x512 with `preview_blocks_to_swap = 24`. A **1024x1024 preview with no
preview block swap does not fit on 8 GB**: VRAM filled to 7.9 / 8 GB, Windows spilled to shared system memory
and the preview stalled for over 30 minutes. For 1024x1024 previews raise `preview_blocks_to_swap` or use a
bigger card. 1024x1024 *training* on 8 GB was not measured.

`blocks_to_swap` is a top-level key (it is ignored under `[model]`); lower it on bigger cards.
Measure VRAM by the torch `cuda_peak` of the bench output, not `nvidia-smi`.

## Text encoder and the embedding cache

Captions are encoded **once** into the cache (`cache_text_embeddings` is required: the 9B encoder
never runs in the training graph). Each caption goes through the Qwen3.5 chat template (thinking
on, no system prompt), is truncated at 4096 tokens and is forwarded **on its own, unpadded** (the
Gated DeltaNet layers are padding-sensitive); the cache stores the final-norm hidden state of every
token of the templated sequence as a variable-length row (about 33 MB for a full 4096 tokens in
bf16). The encoder loads only when captions still need encoding, streams its layers from host RAM
when it does not fit in VRAM (`text_encoder_offload`) and is unloaded afterwards. The transformers
Gated DeltaNet layers fall back to a plain torch implementation unless
`flash-linear-attention` and `causal-conv1d` are installed; it is slower but identical in effect.

A training batch right-pads the text and masks the padded keys in the text-refiner and joint
attention, with each sample's RoPE positions at its own text length, so every sample trains exactly
as the reference's one-sample-at-a-time forward would.

### JSON layout captions

MOSS-Image can place text and objects from a JSON layout prompt (experimental in the release):
`{"summary": {"meta": ..., "canvas": ..., "caption": ...}, "items": [{"label": ..., "caption": ...,
"text": ..., "bbox": [y_min, x_min, y_max, x_max]}]}` with `bbox_format "yxyx"` and
`bbox_coord_space "abs_centralized"` (absolute pixels, origin at the canvas center, y down).
Rengu has no special handling: a layout prompt is **plain caption text** (serialize it with
`json.dumps(layout, ensure_ascii=False)` into the caption file) and goes through the same encoder.
Tag-style captions use the order quality, score, meta, year, character, series, artist, general;
line breaks in a caption are preserved.

## Previews

Previews run the reference sampler: Euler, 50 steps, **true CFG 3.0** (`uncond + g (cond - uncond)`,
fp32, two forwards per step), a fixed shift of 5.0 snapped to the 1000-step training grid, an empty
negative prompt (encoded through the same template; `preview.negative_prompt` overrides it) and the
negative branch **skipping block 22**, as the reference does by default. `guidance_scale = 1.0` runs
one forward per step. Pixels round to multiples of 16. Fifty steps at 1024x1024 are slow on a block-swapped
8 GB card: lower `num_inference_steps` for routine previews.

## Export formats

- **Adapters** are written as `adapter_model.safetensors` with `diffusion_model.<module path>` keys
  (dotted official module paths, plus a `.alpha` scalar per module; LoKr writes `lokr_w1` /
  `lokr_w2_a` / `lokr_w2_b` beside it). Load them in ComfyUI with the stock **Load LoRA (Model and
  CLIP)** (`LoraLoader`) node. You need a ComfyUI with native MOSS-Image support (mid-October 2026 or
  later).
  The same loader also accepts `lora_unet_<path_with_underscores>` and bare paths. Separate `q` / `k`
  / `v` keys load on both the unfused (diffusers) checkpoint and the fused Comfy-Org one, where each
  key lands on its third of `proj.qkv`.
  - **Tested live:** ComfyUI master `fb3befd` (October 2026) on the Comfy-Org bf16 checkpoint. A LoRA
    export and a LoKr export with the default scope each loaded with every key consumed: no
    `lora key not loaded` lines, and 168 weights patched (42 blocks × `qkv` / `o` / `ffn_up_gate` /
    `ffn_down`). Each changed the image at a fixed seed.
  - **Unfused layout:** checked only against ComfyUI's key-naming rules, not in a live load.
- **Full finetune** writes the official diffusers transformer layout (`config.json` +
  `diffusion_pytorch_model.safetensors` with the official key names and `rope.bands`): it reloads in
  the official pipeline (`MOSSImageTransformer.from_pretrained`) and as `model.transformer_path`.

## Validate config

```bash
./rengu validate --config examples/minimal_config_moss_image_lora.toml
```
