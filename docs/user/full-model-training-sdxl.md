# Full-model finetuning (SDXL)

This guide explains how to finetune the **full SDXL model** (no LoRA/LoKr) using Rengu. For adapter training, see [Training SDXL LoRA and LoKr](training-sdxl-lora-lokr.md).

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

To **train the text encoders** alongside the UNet, also set `cache_text_embeddings = false` (they
must run live each step). With block swap this is validated to fit 8 GB, but it is **tight** (the
encoders add ~1.6 GB resident) and slower — on a small card prefer UNet-only (`freeze_text_encoders
= true` + cached embeddings); train the encoders on a larger GPU. See [VRAM optimization](../developer/vram-optimization.md).

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
   `deepspeed --num_gpus=1 -m rengu_flow.main --config examples/full_model_sdxl.toml`
3. Check the log for `Run dir: ...` and for `Full-model SDXL: text encoders frozen` if you set `freeze_text_encoders = true`.

## Block swapping (recommended on small GPUs)

**Training block swap** (`blocks_to_swap`) now works for **full-model** SDXL, not just adapters. It keeps only a few UNet blocks resident on the GPU and streams the rest from CPU RAM on demand, which on an 8 GB card cuts steady-state VRAM (~6.9 GB → ~4–6 GB) and avoids the WSL2 sysmem paging that otherwise makes steps ~3× slower.

```toml
blocks_to_swap = 6   # SDXL UNet has 7 swappable blocks (3 down + mid + 3 up); 6 keeps 1 resident

[optimizer]
gradient_release = true   # REQUIRED with full-model block swap
```

- **`gradient_release = true` is required** for full-model block swap: each block's optimizer step runs *inside* the backward pass while that block is on the GPU. A normal end-of-step `optimizer.step()` would need every trainable block resident at once, defeating the swap. (Adapter training does not need this — the base is frozen.)
- Higher `blocks_to_swap` = less VRAM. On an 8 GB card, `blocks_to_swap=6` (1 resident block, ~4.3 GB) is fastest because it stays well clear of the WSL2 sysmem-paging threshold — counter-intuitively, *more* swapping is faster there. `pipeline_stages` must be `1`.
- `block_swap_prefetch = true` (opt-in, off by default) overlaps transfer with compute, but is counterproductive on tight 8 GB WSL2 (needs ≥2 resident blocks); expected to help on bigger GPUs / native Linux.

See [VRAM optimization](../developer/vram-optimization.md) for the measured curve and how the levers interact, plus [Training loop — block swap](training-loop-and-eval.md) and [Shared training techniques](../developer/training-techniques.md).
