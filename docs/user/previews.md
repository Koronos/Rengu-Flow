# Training previews (user guide)

During training you can generate **sample images** from fixed prompts and view them in **TensorBoard** (similar to OneTrainer). This helps you judge quality without waiting for a full `save_every_n_*` export. Kohya-style “sample every N steps” is covered by the schedule options below; on-demand runs use a **signal file**.

Previews are supported for **SDXL** (`model.type = "sdxl"`) and **Cosmos Predict2** (`model.type = "cosmos_predict2"` or `anima`). Cosmos requires **`pipeline_stages = 1`** (single-GPU DiT path).

## Config editor (web UI)

On the training config **Previews** tab:

1. **Preview configurations** — table of entries (Add, Edit, Duplicate, Remove). Each row is one item in `preview.prompts` in TOML (a string or a `[[preview.prompts]]` table).
2. **Global preview settings** — schedule (`preview_every_n_*`), enable flag, default size/seeds/CFG, and Cosmos-only options. These apply to every row unless a row overrides them in the edit dialog.

## Enable previews in TOML

Add a `[preview]` section with at least one prompt:

```toml
[preview]
prompts = [
  "photo of a red sports car, studio lighting",
]

[[preview.prompts]]
name = "portrait"
prompt = "1woman, soft light, detailed face"

negative_prompt = "blurry, low quality"
width = 1024
height = 1024
num_inference_steps = 20
guidance_scale = 7.0
seed = 42
preview_every_n_steps = 500
# preview_every_n_epochs = 1
# preview_before_first_step = false
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
| **`preview.seed_stride`** | Added per prompt index and per step (`seed + step * seed_stride + index`). | Integer. | `1` |
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
| **`preview.preview_save_png`** | Also write `preview/{name}_step{N}.png` under the run directory (same folder as TensorBoard logs). | `false` |

Video previews (`frame_buckets` > 1) are not supported in v1 — only a single frame (`T=1`).

With `preview_save_png = true`, PNGs are also on disk at `output/<run_dir>/preview/prompt_0_step42.png`.

## View in TensorBoard

Previews appear under the **`preview/`** namespace, e.g. `preview/prompt_0`, `preview/portrait`.

**Important:** point TensorBoard at the **parent `output/` directory**, not the run folder. If you use `--logdir output/my_experiment_20250217_14-30-00`, the sidebar shows only a dot (`.`) as the run name.

```bash
tensorboard --logdir output
```

Then select your run (e.g. `20260527_00-50-09_smoke_signals`) in the left sidebar → **IMAGES** tab.

Select the **IMAGES** tab. The step axis matches training scalars unless you set `x_axis_examples = true` (then previews use the examples axis too).

If WandB is enabled (`monitoring.enable_wandb = true`), preview images are also logged there.

## Force a preview with a signal

Create a file named **`preview`** in the run directory (same place as `save` / `export_model`):

```bash
touch /path/to/output/20250217_14-30-00/preview
```

On the **next training step**, the run generates all configured prompts and logs them to TensorBoard, then removes the file. See [Signal files](signal-files.md).

## Notes

- Previews use the **in-memory** model (current LoRA / UNet weights). They do not read from disk exports.
- Preview runs pause the training step briefly and use extra VRAM; use a modest `num_inference_steps` or longer `preview_every_n_steps` on tight GPUs.
- Resume checkpoints and preview images are independent; use `save` for resume state and `export_model` for inference-ready files.
