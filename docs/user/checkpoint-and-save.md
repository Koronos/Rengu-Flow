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
| **`max_model_exports_to_keep`** | Maximum number of scheduled export folders (`step*`, `epoch*`) to keep. Oldest eligible folders are removed after each successful export. Manual `signal_step*` exports are never auto-deleted. | Positive integer. | Omitted (keep all exports). |
| **`keep_exports_from_step`** | Drop scheduled exports whose training step is below this value, then apply `max_model_exports_to_keep` if set (both rules apply — the stricter combination wins). `epochN` is compared using `N × steps_per_epoch`. | Integer ≥ 0. | Omitted (no minimum step). |
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

## Disk full behavior

### Resume checkpoints (`save` / scheduled checkpoints)

If writing a DeepSpeed checkpoint fails because the disk is full, the trainer **removes the incomplete `global_step*` folder**, logs a warning, and **continues training**. No manual intervention is required. The next successful checkpoint will update `latest` again.

### Model export (`save_every_n_*` / `export_model`)

If exporting inference weights fails because the disk is full, training **pauses** (weights stay on the GPU). Partial files under the export folder are cleaned up.

1. Free disk space on the run directory.
2. Create the signal file **`continue`** in the run directory (or use **Continue export** in the web UI).
3. The trainer retries the same export (`stepN`, `epochN`, or `signal_stepN`) and then resumes the training loop.

While paused, you can also use:

| Signal | Effect during export wait |
|--------|---------------------------|
| `continue` | Retry the pending export, then resume training. |
| `save` | Write a resume checkpoint, stay in the wait loop. |
| `save_quit` | Resume checkpoint, then exit. |
| `export_model_quit` | Retry export, then exit. |
| `quit` | Exit **without** saving (destructive). |

Enable **`monitoring.enable_status_file = true`** so the web UI can show phase `waiting_disk_export` on the Train page and run detail.
