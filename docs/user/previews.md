# Training previews (user guide)

During training you can generate **sample images** from fixed prompts and view them in **TensorBoard** (similar to OneTrainer). This helps you judge quality without waiting for a full `save_every_n_*` export. Kohya-style “sample every N steps” is covered by the schedule options below; on-demand runs use a **signal file**.

Previews are supported for **SDXL** (`model.type = "sdxl"`) in the current release.

## Enable previews in TOML

Add a `[preview]` section with at least one prompt:

```toml
[preview]
prompts = [
  "photo of a red sports car, studio lighting",
  { name = "portrait", prompt = "1woman, soft light, detailed face" },
]
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
| **`disable_block_swap_for_preview`** | Top-level: use full GPU for preview when using block swap (same idea as eval). | `true` or `false`. | Same as `disable_block_swap_for_eval` |

## View in TensorBoard

Previews appear under the **`preview/`** namespace, e.g. `preview/prompt_0`, `preview/portrait`. Open TensorBoard on the run directory:

```bash
tensorboard --logdir output
```

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
