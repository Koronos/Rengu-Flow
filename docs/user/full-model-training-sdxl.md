# Full-model finetuning (SDXL)

This guide explains how to finetune the **full SDXL model** (no LoRA/LoKr) using Renga Flow. For adapter training, see [Training SDXL LoRA and LoKr](training-sdxl-lora-lokr.md).

## When to use full-model vs adapters

- **Adapters (LoRA/LoKr)**: Add a small set of trainable parameters on top of a frozen base. Lower VRAM, faster, good for style/character. Config must include an `[adapter]` section.
- **Full-model**: All (or a subset of) base parameters are trained. Higher VRAM, more capacity. Omit the `[adapter]` section from your config.

## Config: no adapter

Do **not** add an `[adapter]` section. Your config must still have `[model]`, `[optimizer]`, and `dataset`, plus a checkpoint path:

```toml
dataset = "examples/minimal_dataset.toml"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "path/to/sdxl.safetensors"
# optional: freeze_text_encoders = true  # train UNet only (see below)

[optimizer]
type = "adamw"
lr = 1.0e-4
# ... rest as usual
```

Example: see `examples/full_model_sdxl.toml`.

## Optional: train UNet only (freeze text encoders)

To reduce VRAM and often improve stability, you can freeze the two text encoders and train only the UNet:

```toml
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "path/to/sdxl.safetensors"
freeze_text_encoders = true
```

- **`freeze_text_encoders = false`** (default): UNet and both text encoders are trained.
- **`freeze_text_encoders = true`**: Only the UNet is trained; text encoders are frozen.

Example: see `examples/full_model_sdxl_unet_only.toml`.

## Saving full models

- **By epoch**: Set `save_every_n_epochs` (e.g. `1`). Full checkpoints are written to the run directory as subfolders `epoch1`, `epoch2`, etc.
- **By step**: Set `save_every_n_steps`. Saves are named `step500`, `step1000`, etc.
- Each save folder contains **`model.safetensors`** (single-file Comfy-style checkpoint: UNet + VAE + text encoders).
- **Signal files**: `save` / `save_quit` for resume checkpoints; `export_model` to write `model.safetensors` on demand. See [Signal files](signal-files.md) and [Checkpoints, model export, and retention](checkpoint-and-save.md).

## Where outputs go

- **Run directory**: Under `output_dir` (e.g. `output/20250217_14-30-00`).
- **Full-model saves**: Subfolders `epoch<N>` or `step<N>` each containing `model.safetensors`. Use this file in ComfyUI, Forge, or other UIs that load full SDXL checkpoints.

## How to run

1. Use a config **without** an `[adapter]` section.
2. Run with DeepSpeed, e.g.:  
   `deepspeed --num_gpus=1 -m renga_flow.main --config examples/full_model_sdxl.toml`
3. Check the log for `Run dir: ...` and for `Full-model SDXL: text encoders frozen` if you set `freeze_text_encoders = true`.

## Block swapping

Block swapping (offloading layers to CPU) is **not** used for full-model training. It is only supported when training adapters. If your config has `blocks_to_swap` set and no `[adapter]` section, the script will raise an error.
