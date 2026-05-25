# Training loop, evaluation, and logging (user guide)

How the training loop works, how to enable **evaluation**, **TensorBoard** and **WandB** logging, **resume from checkpoint**, and **activation checkpointing** options.

## Evaluation

You can run validation during training over one or more **eval datasets**. Eval runs at fixed timestep quantiles and logs mean loss per dataset (and per quantile) to TensorBoard and optionally WandB.

### Config keys

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`eval_datasets`** | List of eval dataset configs. Each entry is either a path string (dataset TOML) or a table with `name` and `config` (path to TOML). | List of strings or `{ name = "...", config = "path.toml" }`. | `[]` (no eval). |
| **`eval_gradient_accumulation_steps`** | Gradient accumulation steps used for eval (micro-batches per eval step). | Positive integer. | `1` |
| **`eval_every_n_steps`** | Run evaluation every N training steps. | Positive integer or omit to disable. | `null` |
| **`eval_every_n_epochs`** | Run evaluation at the end of every N epochs. | Positive integer or omit. | `null` |
| **`eval_every_n_examples`** | Run evaluation every N examples (converted to steps using global batch size). | Positive integer or omit. | `null` |
| **`eval_before_first_step`** | Run one evaluation before the first training step (useful for baseline metrics). | `true` or `false`. | `true` |
| **`disable_block_swap_for_eval`** | If using block swap (adapters), disable it during eval so the full model runs on GPU for consistent metrics. | `true` or `false`. | `false` |

### Examples

Eval using a single dataset TOML (name will be derived from the filename):

```toml
eval_datasets = ["path/to/eval_dataset.toml"]
eval_every_n_steps = 500
eval_before_first_step = true
```

Eval using named entries (useful when you have several eval sets):

```toml
[[eval_datasets]]
name = "validation"
config = "configs/eval_val.toml"

[[eval_datasets]]
name = "holdout"
config = "configs/eval_holdout.toml"

eval_every_n_epochs = 1
```

## Logging (TensorBoard and WandB)

### TensorBoard

TensorBoard logs are written automatically to the **run directory** (e.g. `output/20250218_12-00-00_myrun/`). No config needed.

Training scalars (every `logging_steps`):

| Tag | When |
|-----|------|
| `train/loss` | Always |
| `train/grad_norm` | When the optimizer exposes `_grad_norm` (e.g. DeepSpeed) |
| `train/epoch_loss` | End of each epoch |
| `train/prodigy_d` | Optimizer type **Prodigy** |
| `train/automagic_avg_lr` | Optimizer **Automagic** or **GenericOptim** |
| `train/automagic_lrs` | Histogram of per-parameter LRs (same optimizers) |

Eval metrics: `{dataset}/loss`, `{dataset}/loss_quantile_{q}`, `eval/eval_time_sec` (unchanged).

- **train/loss** — Loss per step (and optionally **train/grad_norm**, **train/epoch_loss**).
- **eval** — Per-dataset loss and **eval/eval_time_sec** when evaluation runs.
- **preview/** — Sample images when `[preview]` is configured or the `preview` signal is used. See [Training previews](previews.md).

View with:

```bash
tensorboard --logdir output
```

### WandB (optional)

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`[monitoring]`** | Section for logging and tracking. | Table. | — |
| **`monitoring.enable_wandb`** | Enable Weights & Biases logging. | `true` or `false`. | `false` |
| **`monitoring.wandb_api_key`** | API key for WandB (or set `WANDB_API_KEY` env). | String or omit. | `null` |
| **`monitoring.wandb_tracker_name`** | WandB project name. | String. | `"renga-flow"` |
| **`monitoring.wandb_run_name`** | Run name in WandB. If omitted, run directory path is used. | String or omit. | `null` |

Example:

```toml
[monitoring]
enable_wandb = true
wandb_tracker_name = "my-project"
wandb_run_name = "sdxl-lora-v1"
# wandb_api_key = "..."   # optional if WANDB_API_KEY is set
```

### X-axis (steps vs examples)

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`x_axis_examples`** | Use total examples (instead of step count) as the x-axis for all scalar logs. | `true` or `false`. | `false` |

## Resume from checkpoint

To continue training from the latest run under `output_dir`:

```bash
deepspeed ... --resume_from_checkpoint
```

To resume from a specific run folder:

```bash
deepspeed ... --resume_from_checkpoint 20250218_12-00-00_myrun
```

Checkpoint restores model, optimizer, LR scheduler, and dataloader state (epoch and position). Optional flags:

- **`--reset_dataloader`** — Do not restore dataloader state; only restore epoch number. Useful if you changed the dataset.
- **`--reset_optimizer`** — Do not restore optimizer state (e.g. to change optimizer).
- **`--reset_optimizer_params`** — Restore optimizer state but reset param groups (e.g. learning rate) from config.

## Activation checkpointing

Saves VRAM by recomputing activations in the backward pass. Configure in the main TOML:

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`activation_checkpointing`** | Enable and choose implementation. | `false`, `true`, or `"unsloth"`. | `false` |
| **`reentrant_activation_checkpointing`** | When `activation_checkpointing = true`, use reentrant PyTorch checkpoint. | `true` or `false`. | `false` |

- **`true`** — PyTorch `torch.utils.checkpoint.checkpoint`. Use `reentrant_activation_checkpointing = true` if you hit errors with block swap or certain layers.
- **`"unsloth"`** — Unsloth-style checkpoint that offloads activations to CPU (saves more VRAM; requires `deepspeed` and the unsloth-style helper in the codebase).

Example:

```toml
activation_checkpointing = true
reentrant_activation_checkpointing = false
```
