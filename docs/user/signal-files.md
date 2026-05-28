# Control by signal files (user guide)

You can control a running training job from outside the process by creating small **signal files** in the run directory. No API or socket is required: create or touch a file with a specific name and the training loop will react on the next step.

## What are signal files?

Signal files are empty files (or any file) placed in the **run directory**. The training process checks for them once per training step. When it finds a signal file, it performs the corresponding action and then removes the file.

## Where is the run directory?

The run directory is a subfolder of `output_dir` (set in your config; default is `output`).

- **New run**: The framework creates a timestamped folder, e.g. `output/20250217_14-30-00`, or `output/my_experiment_20250217_14-30-00` if you set `run_name` in config (optional label for folders and TensorBoard).
- **Resume**: The run directory is the checkpoint folder you are resuming from (e.g. the same timestamped folder or one you pass via `--resume_from_checkpoint <folder_name>`).

To find the current run directory:

- Look at the log line at start: `Run dir: /path/to/output/20250217_14-30-00`.
- Or take the **most recent** folder under `output_dir` (by name order) for an active run.
- If you use a manager (e.g. diffusion-pipe manager), it discovers the run directory from the job’s output path.

## Available signals

| Signal | File name | Effect |
|--------|-----------|--------|
| Save (resume checkpoint) | `save` | On the next step: write a DeepSpeed resume checkpoint, then remove the file. |
| Save & quit | `save_quit` | Same as `save`, then exit the training process. |
| Export model | `export_model` | On the next step: export adapter or full model to `signal_step<N>/` (usable weights), then remove the file. |
| Export & quit | `export_model_quit` | Same as `export_model`, then exit. |
| Preview | `preview` | On the next step: run configured [previews](previews.md) and log images to TensorBoard, then remove the file. |
| Continue export | `continue` | While paused after disk-full export: retry that export, then resume training. |
| Quit without save | `quit` | While paused after disk-full export: exit without checkpoint or export. |

Signals are processed **once per step** during normal training. During an export wait loop, `continue` / `quit` / `save*` are polled every few seconds.

**Checkpoint vs model export:** `save` / `save_quit` only create **resume checkpoints** (optimizer, scheduler, dataloader state). They do **not** write `model.safetensors` or adapter files for inference. For that, use `export_model` or configure `save_every_n_epochs` / `save_every_n_steps` in TOML. See [Checkpoints, model export, and retention](checkpoint-and-save.md).

## How to send a signal

Create or touch the file in the run directory.

**From the shell:**

```bash
# Resume checkpoint (replace with your actual run dir)
touch /path/to/output/20250217_14-30-00/save

# Save checkpoint and then quit
touch /path/to/output/20250217_14-30-00/save_quit

# Export model weights for inference (LoRA folder or full model.safetensors)
touch /path/to/output/20250217_14-30-00/export_model

# Export model and then quit
touch /path/to/output/20250217_14-30-00/export_model_quit

# Generate preview images (TensorBoard)
touch /path/to/output/20250217_14-30-00/preview

# After disk-full export pause: retry export and resume training
touch /path/to/output/20250217_14-30-00/continue

# Exit without saving while paused
touch /path/to/output/20250217_14-30-00/quit
```

**From the web UI:** Open a run under **Runs** (job or folder on disk). The **Signals** section lists the same actions as above; hover a button for a short hint, or open **Signal files guide** for this page. The **Train** live panel shows **Continue export** when `status.json` reports `waiting_disk_export`.

**From a script or manager:**

- Write or touch `run_dir/save`, `run_dir/save_quit`, `run_dir/export_model`, or `run_dir/export_model_quit`.
- `save` and `save_quit` use the same names as diffusion-pipe for manager compatibility.

## Summary

- **Run dir**: under `output_dir`, timestamped for new runs (and optional `run_name`).
- **Resume signals**: `save`, `save_quit` — DeepSpeed checkpoints only.
- **Export signals**: `export_model`, `export_model_quit` — inference-ready model export.
- **Preview signal**: `preview` — sample images to TensorBoard (requires `[preview]` in config).
- **When**: Checked once per training step; file is removed after the action.
- **From outside**: Any tool that can create/touch files in that directory (shell, manager, cron, etc.) can trigger the signals.
