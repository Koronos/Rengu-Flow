# Optimizer and learning rate scheduler (user guide)

This guide explains how to choose and configure the **optimizer** and **learning rate scheduler** in your TOML config. No implementation details; task-oriented.

**PyTorch reference (which optimizers and schedulers you can use):**

- **Optimizers:** [https://pytorch.org/docs/stable/optim.html](https://pytorch.org/docs/stable/optim.html) — e.g. `Adam`, `AdamW`, `SGD`, `RMSprop`, etc.
- **LR schedulers:** [https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — e.g. `CosineAnnealingLR`, `OneCycleLR`, `StepLR`. Class names live under `torch.optim.lr_scheduler.*`.

Use the built-in short names below or the **fully-qualified path** (e.g. `torch.optim.AdamW`, `torch.optim.lr_scheduler.CosineAnnealingLR`) in your config.

## Optimizer

In the `[optimizer]` section you must set `type` and any parameters the optimizer accepts (e.g. `lr`).

### Built-in names

You can use these names (case-insensitive) for `optimizer.type`:

| Name | Description |
|------|-------------|
| **adamw** | AdamW (`torch.optim`) |
| **adam** | Adam (`torch.optim`) |
| **sgd** | SGD (`torch.optim`) |
| **genericoptim** | GenericOptim (vendored from diffusion-pipe; Muon, Kahan bf16, etc.) |
| **automagic** | Automagic adaptive LR (vendored from diffusion-pipe / AI Toolkit) |
| **adamw8bitkahan** | AdamW 8-bit with Kahan summation (requires `bitsandbytes`) |
| **adamw8bit** | AdamW 8-bit (`bitsandbytes`) |
| **Prodigy** | Resolved via `pytorch-optimizer` if installed (name as in library, case-sensitive) |

Install optional optimizer dependencies:

```bash
pip install -e ".[optim]"
```

### Special optimizer keys (not passed to the constructor)

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **gradient_release** | One optimizer step per parameter after grad accum (saves VRAM) | `true` / `false` | `false` |
| **beta2_half_life** | Recompute `betas[1]` from global batch size before training | positive number | omitted (use `betas` as written) |

`gradient_release` requires **`pipeline_stages = 1`** and data-parallel world size 1 (typical single-process pipeline over all GPUs).

For **genericoptim**, optional **`kahan_buffer_offload`** (boolean) offloads the Kahan buffer to CPU to save VRAM (same as diffusion-pipe).

Example:

```toml
[optimizer]
type = "adamw"
lr = 1.0e-4
```

Other common parameters (depending on optimizer): `weight_decay`, `betas` (for Adam/AdamW), `momentum` (for SGD).

### Fully-qualified path

To use an optimizer that is not built-in (e.g. from `pytorch_optimizer` or a custom package), set `type` to the **fully-qualified class path**:

```toml
[optimizer]
type = "torch.optim.AdamW"
lr = 1.0e-4
```

or, for a third-party package:

```toml
[optimizer]
type = "pytorch_optimizer.Prodigy"
lr = 1.0
# ... other args the class accepts
```

All keys under `[optimizer]` except `type` (and any special keys like `gradient_release`) are passed as keyword arguments to the optimizer constructor.

---

## Learning rate scheduler

Set **`lr_scheduler`** at the top level of your config (or omit it; default is `"constant"`).

### Built-in names

- **constant** — Constant learning rate (no decay).
- **linear** — Linear decay from initial LR to 0 over `total_steps`.
- **cosine** — Cosine annealing; optional minimum LR via `[lr_scheduler_args]`.
- **none** — No scheduler (optimizer LR is used as-is).

Example:

```toml
lr_scheduler = "cosine"

[lr_scheduler_args]
lr_min = 0.0
```

### Runtime tokens in `[lr_scheduler_args]`

For built-in schedulers and for custom schedulers loaded by path, you can use these **string tokens** in `[lr_scheduler_args]`; they are replaced at runtime:

- **`total_steps`** — Total training steps (epochs × steps per epoch).
- **`steps_per_epoch`** — Number of steps in one epoch.
- **`epochs`** — Number of epochs from config.

Example (for a custom scheduler class that accepts `T_max`):

```toml
lr_scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"

[lr_scheduler_args]
T_max = "total_steps"
eta_min = 0.0
```

### Fully-qualified path

To use a scheduler class from PyTorch or another package:

```toml
lr_scheduler = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"

[lr_scheduler_args]
T_0 = 100
T_mult = 2
```

Arguments in `[lr_scheduler_args]` are passed to the scheduler constructor after the optimizer; runtime tokens are substituted before calling the constructor.

---

## Warmup

To add a short linear warmup before the main scheduler, set **`warmup_steps`** in your config (default is 0). The framework wraps the chosen scheduler with a warmup phase: LR goes from `initial_lr / warmup_steps` to `initial_lr` over `warmup_steps` steps, then the main scheduler runs.

```toml
warmup_steps = 100
lr_scheduler = "cosine"
```

This has no effect if `lr_scheduler` is `"none"` or if `warmup_steps` is 0.
