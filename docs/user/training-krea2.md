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

## Recommended defaults

Most runs only need the three component paths plus these; leave everything else unset.

- **Adapter:** LoRA `rank = 32` (the model authors' reference); on a quantized base use the
  quantization-aware `lokr` instead (recipe below). `alpha` is always `rank` (setting it is
  rejected).
- **Timesteps:** keep the defaults — `logit_normal` sampling plus the resolution-aware
  dynamic shift is the reference Krea 2 scheduler. `shift`, `sigmoid_scale` and
  `timestep_sample_method` are advanced A/B knobs.
- **`activation_checkpointing = true`.**
- **16 GB card:** a quantized base (`model.transformer_4bit = true`, or
  `model.transformer_fp8_matmul = true` on RTX 40xx) plus `blocks_to_swap = 20`, with an
  `[adapter]`:

```toml
activation_checkpointing = true
blocks_to_swap = 20          # top-level: every key after [model] belongs to [model]

[model]
type = "krea2"
dtype = "bfloat16"
transformer_path = "path/to/krea2_raw_bf16.safetensors"
vae_path = "path/to/qwen_image_vae.safetensors"
text_encoder_path = "path/to/qwen3vl_4b_bf16.safetensors"
transformer_4bit = true

[adapter]
type = "lokr"                # quantization-aware; lycoris_* refuse a quantized base
rank = 6
factor = -1
```

Top-level training keys (`blocks_to_swap`, `compile*`, `activation_checkpointing`,
`reentrant_activation_checkpointing`, `gradient_accumulation_steps`, ...) must sit **before**
the first `[section]` header. Written under `[model]` or `[adapter]` they would be silently
ignored, so validation rejects them with "belongs at top level".

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
| **`dtype`** | Load/compute dtype for the VAE, text encoder and (unless `transformer_dtype` is set) the DiT. Adapter weights are separate: `adapter.dtype`, default `float32`. | Yes | — |
| **`transformer_path`** | DiT: the official `raw.safetensors` / ComfyUI's `krea2_raw_bf16.safetensors`, or a diffusers `transformer/` folder. Rejects pre-quantized fp8/nvfp4 "scaled" files. | One of `transformer_path` / `checkpoint_path` | — |
| **`vae_path`** | Qwen-Image VAE: `qwen_image_vae.safetensors` (same file Cosmos uses) or a diffusers `vae/` folder. | One of `vae_path` / `checkpoint_path` | — |
| **`text_encoder_path`** | Qwen3-VL: `qwen3vl_4b_bf16.safetensors` or a transformers `text_encoder/` folder. | One of `text_encoder_path` / `checkpoint_path` | — |
| **`checkpoint_path`** | Full diffusers-layout folder (`transformer/`, `vae/`, `text_encoder/`); fills any of the three component paths left empty. Set either this or the three component paths. | No | Unset |
| **`tokenizer_path`** | Folder with tokenizer files. | No | Bundled Qwen3-VL tokenizer (`rengu_flow/model/krea2/assets/qwen3vl_4b`) |
| **`max_sequence_length`** | Prompt token budget before truncation (integer >= 1). Lower it to shrink the text-embedding cache; captions longer than this lose their tail. | No | `512` |
| **`transformer_dtype`** | DiT checkpoint load dtype only (VAE/text unaffected). | No | `dtype` |
| **`transformer_4bit`** | Quantize the frozen DiT's linears to 4-bit NF4 (bitsandbytes). Adapter training only; mutually exclusive with `transformer_fp8_matmul`. | No | `false` |
| **`transformer_fp8_matmul`** | Quantize the frozen DiT's linears to fp8 (tensorwise-scaled e4m3, 1 byte/param; always e4m3). Adapter training only; mutually exclusive with `transformer_4bit`. | No | `false` |
| **`fp8_grad_mode`** | `"bf16"` or `"fp8"`: backward input-gradient GEMM precision, only used when `transformer_fp8_matmul = true`. | No | `"bf16"` |
| **`timestep_sample_method`** | Advanced. `"logit_normal"` or `"uniform"` timestep sampling for training; other values are rejected. | No | `"logit_normal"` |
| **`sigmoid_scale`** | Advanced. Scales the logit-normal sample before the sigmoid; only used when `timestep_sample_method = "logit_normal"`. | No | `1.0` |
| **`shift`** | Advanced. Fixed rectified-flow time shift (must be > 0). When set, it **overrides** the default resolution-aware dynamic shift below. | No | Unset (dynamic) |
| **`cache_text_embeddings`** | Always required `true` — the tapped 12-layer Qwen3-VL stack cannot run inside the training graph. Setting it `false` is rejected at startup. | No | `true` |

`diffusion_model_dtype` is accepted in TOML but not shown in the web UI for Krea 2: it only
overrides the training autocast dtype and never the DiT load dtype (that is
`transformer_dtype`), so `dtype` / `transformer_dtype` already cover it.

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
reference Krea 2 scheduler config. The exponential shift at `mu` is exactly the usual fixed
shift `t' = s*t / (1 + (s - 1)*t)` with `s = exp(mu)`, so the default spans `s ~ 1.65` (256
tokens) to `s ~ 3.16` (6400 tokens); 1024x1024 is 4096 tokens. Use that mapping when comparing
with trainers that take a single fixed shift value (such as musubi-tuner). Set `model.shift`
to a fixed number only if you want to override the per-resolution behavior with a constant.

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
  by exact class name `Linear`, so it would silently skip the quantized linears
  (`Fp8TensorwiseLinear` for Krea 2's fp8 base, `Linear4bit` for 4-bit) — config validation
  rejects `transformer_4bit` / `transformer_fp8_matmul` together with any `lycoris_*` adapter.
  Use `adapter.type = "lokr"` (quantization-aware) on a quantized base instead.
- **`train_conv`, `use_tucker`, `train_norm` do nothing here** (hidden in the web UI): the DiT
  has no Conv layers, and its norms are Krea's own RMSNorm, which LyCORIS' norm training
  (affine LayerNorm/GroupNorm only) does not match — `train_norm = true` fails at startup.

### Full finetune

Omit the `[adapter]` section. All DiT parameters with `requires_grad` are trained (a quantized
base — `transformer_4bit` / `transformer_fp8_matmul` — is rejected at validation: it is
frozen by design); export
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
layer_groups = ["text_fusion"]
# or combine several:
# layer_groups = ["text_fusion", "attention"]
```

Krea 2 groups: `text_fusion` (the Krea2TextFusion stack + txt_in projection), `attention`
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

## Speed: fp8 base + token routing

Three knobs combine into an "F8T" recipe that measured a large per-step speedup on a 16 GB
card, at the cost of some setup complexity and an open quality A/B:

```toml
# Top-level keys first — anything after [model] belongs to [model].
compile = true
compile_dynamic = true
compile_scope = "block"
activation_checkpointing = true
blocks_to_swap = 16
# reentrant_activation_checkpointing defaults to true for this combo (see below)

[model]
# ...type / dtype / component paths...
transformer_fp8_matmul = true

[tread]
drop_ratio = 0.5
disable_after_frac = 0.85
```

- **`model.transformer_fp8_matmul`** stores the frozen DiT's big linears as tensorwise-scaled
  e4m3 (1 byte/param): 25.6 -> 12.9 GB resident, and ~2x bf16 GEMM throughput on the cuBLASLt
  kernel — but only once `compile_scope = "block"` is also on (eager fp8 nets ~nothing).
- **`compile_scope = "block"`** compiles each transformer block alone instead of the whole
  pipeline, so activation checkpointing and block swap stay eager (whole-module compile
  graph-breaks on those hooks). This is the prerequisite for the fp8 GEMM speedup above, and
  measured 1.14x alone (bf16, no fp8).
- **`[tread]` `drop_ratio`** (TREAD-style token routing, arXiv 2501.04765) trains the blocks
  between `start_block` (default `2`) and `end_block` (default `-3`, negative = from the end)
  on a random batch-shared subset of image tokens instead of the full sequence; text tokens
  are always kept, and dropped tokens still get gradient through the bypass. Leaving
  `drop_ratio` unset keeps the `[tread]` table out of the TOML entirely (routing off — there
  is no default). Training-only: never active for eval, previews, or val probes. Checked at
  validation: `drop_ratio` in (0, 1), `disable_after_frac` in (0, 1], and the route must
  satisfy `0 < start < end < 27` (28 blocks; the first and last block stay unrouted).

**Constraint:** `transformer_fp8_matmul` + `compile_scope = "block"` with activation
checkpointing requires `reentrant_activation_checkpointing = true`, so it **defaults to
`true`** for that combination (an explicit `false` is kept but logs a warning). Non-reentrant
AC's recompute compares checkpoint
metadata against the forward graph, and that comparison fails once the block is compiled
(static-vs-dynamic-shape divergence, pytorch#166926); reentrant AC runs the recompute under
`no_grad` and skips the comparison entirely.

**`tread.disable_after_frac`** is the routing off-ramp: past that fraction of the run
(0.85 = the final 15%) the routing turns off and training finishes on full sequences, which
recovers nearly all of the routed-training quality gap (below) at a fraction of the speed
cost. `1.0` (default) never disables.

Measured on the wlop 96 bench (RTX 4080 16 GB, `blocks_to_swap = 16`):

| Config | 512 @ bs2 | 1024 @ bs1 | VRAM peak | Pinned host RAM |
|--------|-----------|------------|-----------|-----------------|
| bf16 + swap25 baseline | 2.87 s/it | 6.23 s/it | 12.1 GB | ~24 GB |
| F8T (fp8 + block-compile + TREAD 0.5) | 1.66 s/it | 2.62 s/it | 12.9 GB | ~7 GB |
| + `fp8_grad_mode = "fp8"` | ~1.7 s/it | ~2.3 s/it | 13.3 GB | ~7 GB |

**Quality (measured, 300-step wlop A/B, all adapters probed on the clean bf16 base):** the
baseline adapter scored val 0.0925; TREAD 0.5 alone cost +7.4% (val 0.0993) and fp8 added
only +0.6% on top — the e4m3 tensorwise quantization (2.65% RMS weight error vs NF4's 9.55%)
is effectively free. With the `disable_after_frac = 0.85` off-ramp the full recipe landed at
**val 0.0936 (+1.2% vs baseline)** while keeping routed speed for 85% of the run. TREAD shows
an initial loss spike that converges within a few hundred steps.

**Known issue:** with this recipe, mid-run val-gap probes (`val_gap_enable` +
`eval_datasets`) can stall for minutes per probe inside the block-swap prefetch throttle.
Until fixed, keep in-run probes off for fp8+block-compile runs and evaluate the exported
adapter instead. `fp8_grad_mode = "fp8"` shares this interaction.

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

Changing `max_sequence_length` changes the cache key, so the next run re-encodes the
captions instead of reusing embeddings built with the old setting. Swapping the text
encoder or tokenizer does not: delete the text-embedding cache yourself in that case.

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

To check the whole training path on your GPU: `python scripts/smoke_krea2_mini.py` trains a
few steps on a small random-weight Krea 2 (no downloads, any 4 GB+ GPU); with the real files
in `.env` (`RENGU_KREA2_*`, see `.env.example`), `scripts/run_model_smoke.sh krea2` runs a
16 GB-sized LoRA smoke (Linux/WSL). See [Smoke tests](../developer/smoke-tests.md).
