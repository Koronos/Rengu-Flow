# Checkpoints, model export, and retention (user guide)

Training produces two kinds of artifacts in the **run directory** (under `output_dir`):

1. **Resume checkpoints** — DeepSpeed state to continue training after a crash or stop (`global_step*` folders plus a `latest` pointer). Not meant for inference in ComfyUI/A1111.
2. **Model export** — Adapter (LoRA/LoKr) or full `model.safetensors` for inference, written to subfolders such as `epoch1`, `step500`, or `signal_step1200`.

## Scheduled saves (TOML)

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`checkpoint_every_n_epochs`** | Write a resume checkpoint at the end of every N epochs. | Positive integer. | Omitted (no epoch checkpoints). |
| **`checkpoint_every_n_minutes`** | Write a resume checkpoint when this many minutes have passed since the last checkpoint. | Positive number (minutes). | Omitted. |
| **`max_checkpoints_to_keep`** | Maximum number of `global_step*` checkpoint folders to keep in the run directory. Oldest are deleted after each new checkpoint. | Positive integer. `latest` always points at the newest checkpoint. | Omitted (keep all checkpoints). |
| **`save_every_n_epochs`** | Export model weights every N epochs. | Positive integer. | `1` (set by framework defaults). |
| **`save_every_n_steps`** | Export model weights every N training steps. | Positive integer. | Omitted. |
| **`save_every_n_examples`** | Same as `save_every_n_steps`, but counted in total examples seen; converted using global batch size. | Positive integer. | Omitted. |
| **`save_dtype`** | Cast exported weights to this dtype. | `bfloat16`, `float16`, `float32`, etc. | Model dtype. |
| **`save_full_model`** | Export full trainable backbone (`model.safetensors`) instead of adapter-only files. | `true` or `false`. | `false` (adapter export when `[adapter]` is set). |

Example:

```toml
checkpoint_every_n_epochs = 1
max_checkpoints_to_keep = 3
save_every_n_epochs = 2
save_every_n_steps = 1000
```

## On-demand control (signal files)

See [Signal files](signal-files.md) for the run directory path and timing (once per training step).

| Signal | File | Effect |
|--------|------|--------|
| Resume checkpoint | `save` | DeepSpeed checkpoint on next step. |
| Checkpoint and exit | `save_quit` | Checkpoint, then exit. |
| Export model | `export_model` | Export adapter or full model to `signal_step<N>/` on next step. |
| Export and exit | `export_model_quit` | Export model, then exit. |

```bash
touch /path/to/output/20250217_14-30-00/export_model
```

Use **`save`** when you need to resume training later; use **`export_model`** when you need a usable LoRA or full checkpoint without waiting for `save_every_n_*`.

## Resume training

```bash
deepspeed ... --resume_from_checkpoint
```

Loads the checkpoint indicated by `latest` in the run directory. See [Training loop, evaluation, and logging](training-loop-and-eval.md#resume-from-checkpoint).
