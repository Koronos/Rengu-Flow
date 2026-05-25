# Training SDXL LoRA and LoKr (user guide)

This guide explains how to train SDXL with **LoRA** or **LoKr** (LyCORIS) adapters using Renga Flow. No implementation details; task-oriented.

## What you need

- A TOML config file with `[model]` (type `sdxl`), `[optimizer]`, and `dataset`.
- An SDXL checkpoint path in `model.checkpoint_path`.
- For LoKr, you can optionally install the LyCORIS backend:  
  `pip install renga-flow[lycoris]`  
  If you do not install it, LoKr still works using the built-in (vendored) implementation.

## Config: LoRA

Add an `[adapter]` section with `type = "lora"` and a rank (or `dim`):

```toml
[adapter]
type = "lora"
rank = 16
# optional: dim = 16 (alias for rank, Kohya-style), alpha = 16, dropout = 0.0, dtype = "bfloat16"
```

- **rank**: LoRA rank (e.g. 8, 16, 32). Required unless you use `dim`.
- **dim**: Alias for `rank` (Kohya-style). You must set either `rank` or `dim`.
- **alpha**: Optional. Scaling factor; effective scale is `alpha / rank`. Default: same as rank (scale 1.0). Increase for stronger adapter effect, decrease for weaker.
- **dropout**: Optional; default 0.0.
- **dtype**: Optional; defaults to the model dtype.

Example minimal LoRA config: see `examples/minimal_config_lora_sdxl.toml`.

## Config: LoKr

Add an `[adapter]` section with `type = "lokr"` and a rank (or `dim`):

```toml
[adapter]
type = "lokr"
rank = 16
# optional: dim = 16 (alias for rank), alpha = 16, factor = -1, decompose_both = false, full_matrix = false, dtype = "bfloat16"
```

- **rank**: LoKr rank. Required unless you use `dim`.
- **dim**: Alias for `rank` (Kohya-style). You must set either `rank` or `dim`.
- **alpha**: Optional. Scaling factor; effective scale is `alpha / rank`. Default: same as rank (scale 1.0). Increase for stronger adapter effect, decrease for weaker.
- **factor**: Factorization hint; use -1 for automatic. Optional.
- **decompose_both**: Decompose both Kronecker factors. Optional; default false.
- **full_matrix**: Use full matrices instead of low-rank for the second factor. Optional; default false.
- **dtype**: Optional; defaults to the model dtype.

Example minimal LoKr config: see `examples/minimal_config_lokr_sdxl.toml`.

## Loss function (top-level config)

Optional keys in the **main** training TOML (not in `[adapter]`):

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **huber_delta** | Huber loss (PyTorch standard) | positive float | MSE if unset |
| **smooth_l1_beta** | Smooth L1 loss | positive float | MSE if unset |
| **pseudo_huber_c** | Legacy pseudo-Huber from diffusion-pipe configs | positive float | prefer `huber_delta` for new configs |

Only one of these should be set; if none are set, training uses MSE.

## Saving adapters

- **By epoch**: Set `save_every_n_epochs = 1` (or another number) in your config. Adapters are written to the run directory as `epoch1`, `epoch2`, etc.
- **By step**: Set `save_every_n_steps` to save every N steps (e.g. `save_every_n_steps = 500`). Adapters are written as `step500`, `step1000`, etc.
- At least one of `save_every_n_epochs` or `save_every_n_steps` is recommended (defaults set `save_every_n_epochs = 1`).
- **Signal files**: `save` / `save_quit` for resume checkpoints; `export_model` / `export_model_quit` to export weights on demand; `preview` for TensorBoard sample images. See [Signal files](signal-files.md), [Checkpoints, model export, and retention](checkpoint-and-save.md), and [Training previews](previews.md).

## Where outputs go

- **Run directory**: Under `output_dir` (default `output`), e.g. `output/20250217_14-30-00` or `output/20250217_14-30-00_my_run` if you set `run_name`.
- **Adapter saves**: Subfolders named `epoch<N>` or `step<N>` inside the run directory. Each contains:
  - **LoRA**: `lora.safetensors` (Kohya format).
  - **LoKr**: `adapter_model.safetensors` (LyCORIS/Comfy format).
- These files can be used in ComfyUI, Forge, or other inference UIs that support LoRA/LyCORIS.

## How to run training

1. Install: `pip install -e .` (or `pip install renga-flow[lycoris]` for optional LyCORIS backend).
2. Run with DeepSpeed, e.g.:  
   `deepspeed --num_gpus=1 -m renga_flow.main --config examples/minimal_config_lora_sdxl.toml`
3. Check the log for `Run dir: ...` to see where checkpoints and adapters are saved.

## Resuming and loading an existing adapter

- **Resume training**: Use `--resume_from_checkpoint` (or set `resume_from_checkpoint = true` in config) to resume from the latest run directory.
- **Start from an existing LoRA/LoKr file**: Set `init_from_existing` in the adapter section to the path of a directory containing a `.safetensors` file. The script will load those weights before training.
