# Training previews (user guide)

During training you can generate **sample images** from fixed prompts and view them in **TensorBoard** (similar to OneTrainer). This helps you judge quality without waiting for a full `save_every_n_*` export. Kohya-style “sample every N steps” is covered by the schedule options below; on-demand runs use a **signal file**.

Previews are supported for **SDXL** (`model.type = "sdxl"`), **Cosmos Predict2** (`model.type = "cosmos_predict2"` or `anima`), and **Krea 2** (`model.type = "krea2"`, defaults 28 steps / CFG 4.5 — see [Training Krea 2](training-krea2.md)). Cosmos and Krea 2 require **`pipeline_stages = 1`** (single-GPU DiT path).

> **Previews are off by default** (`preview.enabled = false`). Generating samples during training costs extra VRAM and time — a 1024×1024 SDXL preview can OOM on small GPUs (e.g. 8 GB). Enable it only when you want in-training samples, and lower `width`/`height` if you are tight on VRAM.

## Config editor (web UI)

On the training config **Previews** tab:

1. **Preview configurations** — table of entries (Add, Edit, Duplicate, Remove). Each row is one item in `preview.prompts` in TOML (a string or a `[[preview.prompts]]` table).
2. **Global preview settings** — schedule (`preview_every_n_*`), enable flag, default size/seeds/CFG, and Cosmos-only options. These apply to every row unless a row overrides them in the edit dialog.

## Enable previews in TOML

Add a `[preview]` section with at least one prompt:

```toml
[preview]
# Sampling settings apply to all prompts — they are read at the [preview] level, not per prompt.
negative_prompt = "blurry, low quality"
width = 1024
height = 1024
num_inference_steps = 20
guidance_scale = 7.0
seed = 42
seed_stride = 0  # Keep the same noise for each prompt across training steps.
preview_every_n_steps = 500
# preview_every_n_epochs = 1
# preview_before_first_step = false

# Each prompt entry is a string, or a [[preview.prompts]] table that reads only `name` and `prompt`.
prompts = [
  "photo of a red sports car, studio lighting",
]

[[preview.prompts]]
name = "portrait"
prompt = "1woman, soft light, detailed face"
```

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`preview.enabled`** | Turn previews on or off without removing prompts. | `true` or `false`. | `true` when `prompts` is set |
| **`preview.prompts`** | Prompts to render. Each entry is a string, or a table with `prompt` (or `text`) and optional `name` for the TensorBoard tag. | List of strings or tables. | Required to enable previews |
| **`preview.negative_prompt`** | Negative prompt passed to the pipeline. | String. | `""` |
| **`preview.width`** | Output width in pixels. | Positive integer. | `1024` |
| **`preview.height`** | Output height in pixels. | Positive integer. | `1024` |
| **`preview.num_inference_steps`** | Denoising steps per image. | Positive integer. | `20` |
| **`preview.guidance_scale`** | Classifier-free guidance scale. | Positive number. | `7.0` |
| **`preview.seed`** | Base RNG seed. | Integer. | `0` |
| **`preview.seed_stride`** | Optional per-step offset (`seed + step * seed_stride + index`). Keep `0` to compare model evolution using the same noise; set a positive value to vary noise between previews. | Integer. | `0` |
| **`preview.preview_every_n_steps`** | Generate previews every N training steps. | Positive integer. | Omitted (no step schedule) |
| **`preview.preview_every_n_epochs`** | Generate at the end of every N epochs. | Positive integer. | Omitted |
| **`preview.preview_before_first_step`** | Run once before step 1 (like eval). | `true` or `false`. | `false` |
| **`disable_block_swap_for_preview`** | When training uses `blocks_to_swap`, set `true` to run preview with the full DiT on GPU. | `true` or `false`. | Same as `disable_block_swap_for_eval` |

### Cosmos Predict2 (`cosmos_predict2` / `anima`)

Cosmos / **Anima** use **Euler flow-matching** sampling aligned with training (not the SDXL diffusers scheduler). **Classifier-free guidance** is applied at preview time only: `v = v_uncond + guidance_scale × (v_cond − v_uncond)` with a second forward pass per step when `guidance_scale ≠ 1`.

Recommended for **Anima** previews: **`num_inference_steps = 20`**, **`guidance_scale = 4`**, **`negative_prompt = ""`** (empty string still encodes an unconditional embedding). Training itself does not use CFG.

| Key | Description | Default (Cosmos / Anima) |
|-----|-------------|--------------------------|
| **`preview.width`** / **`preview.height`** | Output size in pixels (multiple of 16). | `1024` |
| **`preview.num_inference_steps`** | Euler steps from noise to data. | `20` |
| **`preview.guidance_scale`** | CFG scale for preview sampling. | `4.0` |
| **`preview.negative_prompt`** | Unconditional caption for CFG. | `""` |
| **`preview.preview_offload_text_encoder`** | Move LLM/T5 to CPU during the Euler loop to save VRAM. | `true` |
| **`preview.preview_blocks_to_swap`** | Cosmos only: DiT blocks kept on CPU between Euler preview steps. `0` disables. Uses the same offloader as training `blocks_to_swap` ([Training loop — block swap](training-loop-and-eval.md#block-swap-vram-adapter-training)). | `0` |
| **`preview.preview_offload_dit_for_decode`** | Cosmos only: move the DiT to CPU during the VAE decode. **Unsafe on DeepSpeed/compiled runs** (the CPU↔GPU round-trip invalidates parameter storage the engine still points at, crashing the next NCCL op). Rarely needed — the decode is tiled (see note below). | `false` |
| **`preview.preview_save_png`** | Also write `preview/{name}_step{N}.png` under the run directory (same folder as TensorBoard logs). **The web UI run page's preview gallery reads these PNGs, so enable this to see previews in the UI** — TensorBoard's IMAGES tab shows them either way. | `false` |

The preview **VAE decode is tiled**: latents larger than 512×512 px decode in overlapping 512 px tiles that are blended together, so the decode's activation peak stays small enough to fit next to the resident DiT and training state even on a 16 GB GPU at 1024×1024.

Video previews (`frame_buckets` > 1) are not supported in v1 — only a single frame (`T=1`).

With `preview_save_png = true`, PNGs are also on disk at `output/<run_dir>/preview/prompt_0_step42.png`.

## View in TensorBoard

Previews appear under the **`preview/`** namespace, e.g. `preview/prompt_0`, `preview/portrait`.

**Important:** point TensorBoard at the **parent `output/` directory**, not the run folder. If you use `--logdir output/20250217_14-30-00_my_experiment`, the sidebar shows only a dot (`.`) as the run name.

```bash
tensorboard --logdir output
```

Then select your run (e.g. `20260527_00-50-09_smoke_signals`) in the left sidebar → **IMAGES** tab.

Select the **IMAGES** tab. The step axis matches training scalars unless you set `x_axis_examples = true` (then previews use the examples axis too).

If WandB is enabled (add `"wandb"` to `tracking.backends`), preview images are also logged there.

## Force a preview with a signal

Create a file named **`preview_now`** in the run directory (same place as `save` / `export_model`). The name is `preview_now`, not `preview`, so it cannot collide with the run folder's `preview/` image directory:

```bash
touch /path/to/output/20250217_14-30-00/preview_now
```

On the **next training step**, the run generates all configured prompts and logs them to TensorBoard, then removes the file. See [Signal files](signal-files.md).

## Edit previews live (no restart)

You can change previews **while a run is training** — prompts, cadence
(`preview_every_n_steps` / `preview_every_n_epochs`), sampling params, or turn them
off entirely with `enabled = false`. Only the `[preview]` section is hot-reloaded;
model/optimizer/dataset changes still require a new run.

**Web UI:** open the run, use the **Live preview settings** panel (shown while the
run is active), edit, and click **Apply** (or **Apply & preview now**). Changes are
written to the run's config and persist if you later resume/continue it.

**CLI:** edit the `[preview]` section in the run's config `.toml` (under the run
folder / the file you passed to `--config`), then drop the reload signal:

```bash
# 1) edit [preview] in your config (e.g. add prompts, change preview_every_n_steps,
#    or set enabled = false to stop previews)
# 2) apply it live:
touch /path/to/output/20250217_14-30-00/reload_config
```

On the next step the trainer re-reads `[preview]` and applies it. See
[Signal files](signal-files.md).

## Notes

- Previews use the **in-memory** model (current LoRA / UNet weights). They do not read from disk exports.
- Preview runs pause the training step briefly and use extra VRAM; use a modest `num_inference_steps` or longer `preview_every_n_steps` on tight GPUs.
- Resume checkpoints and preview images are independent; use `save` for resume state and `export_model` for inference-ready files.
