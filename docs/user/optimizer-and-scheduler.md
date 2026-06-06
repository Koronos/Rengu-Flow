# Optimizer and learning rate scheduler (user guide)

This guide explains how to choose and configure the **optimizer** and **learning rate scheduler** in your TOML config. No implementation details; task-oriented.

**PyTorch reference (which optimizers and schedulers you can use):**

- **Optimizers:** [https://pytorch.org/docs/stable/optim.html](https://pytorch.org/docs/stable/optim.html) — e.g. `Adam`, `AdamW`, `SGD`, `RMSprop`, etc.
- **LR schedulers:** [https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — e.g. `CosineAnnealingLR`, `OneCycleLR`, `StepLR`. Class names live under `torch.optim.lr_scheduler.*`.

Use the built-in short names below or the **fully-qualified path** (e.g. `torch.optim.AdamW`, `torch.optim.lr_scheduler.CosineAnnealingLR`) in your config.

## Optimizer

Set **`optimizer.type`** in `[optimizer]`. In the training config form, **Optimizer parameters** is a single key-value list for every optimizer type (built-in and custom). Rows are written as keys under `[optimizer]` in TOML.

Example:

```toml
[optimizer]
type = "adamw"
lr = 1.0e-4
betas = [0.9, 0.999]
weight_decay = 0.01
```

### Built-in names

You can use these names (case-insensitive) for `optimizer.type`:

| Name | Description | Official / upstream docs |
|------|-------------|---------------------------|
| **adamw** | AdamW (`torch.optim`) | [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) |
| **adam** | Adam (`torch.optim`) | [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) |
| **sgd** | SGD (`torch.optim`) | [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) |
| **genericoptim** | GenericOptim (vendored; Muon, Kahan bf16, …) | diffusion-pipe vendor copy (see repo `NOTICE`) |
| **automagic** | Automagic adaptive LR (vendored) | diffusion-pipe / AI Toolkit vendor copy |
| **adamw8bitkahan** | AdamW 8-bit + Kahan (`bitsandbytes`) | [bitsandbytes AdamW8bit](https://github.com/TimDettmers/bitsandbytes) |
| **adamw8bit** | AdamW 8-bit (`bitsandbytes`) | [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| **adamw_optimi**, **stableadamw** | Optimi AdamW variants | [optimi](https://github.com/williamberman/optimi) |
| **offload** | CPU-offload wrapper (`torchao`) | [torchao](https://github.com/pytorch/ao) |
| **prodigy** | Prodigy adaptive LR (`pytorch-optimizer`) | [Prodigy paper/repo](https://github.com/konstmish/prodigy), [pytorch-optimizer Prodigy](https://github.com/kozistr/pytorch_optimizer) |
| **adafusion** | Conv-aware factored optimizer; AdamW-quality at 1–2 B/param with bf16-correct updates (`koptim`) | [K-Optimizers](https://github.com/Koronos/K-Optimizers), [Adafusion docs](https://github.com/Koronos/K-Optimizers/blob/main/docs/adafusion.md) |
| **muon** | Orthogonalized-momentum (Newton-Schulz) with AdamW fallback for 1-D / embedding params (`koptim`) | [K-Optimizers](https://github.com/Koronos/K-Optimizers), [Muon docs](https://github.com/Koronos/K-Optimizers/blob/main/docs/muon.md) |
| **adamuon** | Muon orthogonalized momentum + factored quantized 2nd moment; near-Adafactor memory. **Diffusion lr is much lower than Muon's** — start ~`1e-3` (≈ AdamW lr ÷ 5), not the `2e-2` Muon/LLM default (`koptim`) | [K-Optimizers](https://github.com/Koronos/K-Optimizers), [AdaMuon docs](https://github.com/Koronos/K-Optimizers/blob/main/docs/adamuon.md) |

Install optional optimizer dependencies:

```bash
pip install -e ".[optim]"
```

`adafusion`, `muon`, and `adamuon` come from the git-backed [`koptim`](https://github.com/Koronos/K-Optimizers) package and are installed on demand via the **koptim** install profile when you select one of these types.

### Form pre-fill (optimizer KV)

When you pick a built-in name in the form, common keys are pre-filled (edit as needed):

| `optimizer.type` | Pre-filled keys (values) |
|------------------|--------------------------|
| **adamw** | `lr` → `1e-4`, `betas` → `[0.9, 0.999]`, `weight_decay` → `0.01` |
| **adam** | `lr` → `1e-4`, `betas` → `[0.9, 0.999]`, `weight_decay` → `0.0` |
| **sgd** | `lr` → `1e-3`, `momentum` → `0.9`, `weight_decay` → `0.0` |
| **adamw8bit**, **adamw_optimi**, **stableadamw**, **offload** | `lr`, `betas`, `weight_decay` (same as adamw-style defaults) |
| **adamw8bitkahan** | `lr`, `betas`, `weight_decay`, `kahan_buffer_offload` → `false` |
| **genericoptim** | `lr`, `betas`, `weight_decay`, `muon`, `adamuon`, `correct_bias` |
| **automagic** | `min_lr`, `max_lr`, `lr_bump` |
| **prodigy** | `lr` → `1.0`, `betas` → `[0.9, 0.99]`, `weight_decay` → `0.01`, `d0` → `1e-6`, `d_coef` → `1.0`, `weight_decouple` → `true`, `bias_correction` → `true`, `safeguard_warmup` → `true` |
| **adafusion** | `lr` → `1e-4`, `betas` → `[0.9, 0.999]`, `eps` → `[1e-30, 1e-3]`, `weight_decay` → `0.0`, `clip_threshold` → `1.0`, `momentum_dtype` → `"bfloat16"`, `cautious` → `true`, `bf16_method` → `"stochastic_rounding"` |
| **muon** | `lr` → `2e-2`, `momentum` → `0.95`, `adamw_lr` → `3e-4`, `bf16_method` → `"stochastic_rounding"` |
| **adamuon** | `lr` → `1e-3` (diffusion-scale, **not** Muon's `2e-2`), `betas` → `[0.95, 0.999]`, `eps` → `[1e-30, 1e-3]`, `weight_decay` → `0.0`, `ns_steps` → `2`, `clip_threshold` → `1.0`, `momentum_dtype` → `"bfloat16"`, `cautious` → `true`, `bf16_method` → `"stochastic_rounding"` |
| Custom class path | (empty list until you add rows) |

### Common parameters (by family)

| Key | Used by | Notes |
|-----|---------|--------|
| **lr** | Most optimizers | Base learning rate |
| **betas** | Adam, AdamW, 8-bit variants | Exactly two floats `[beta1, beta2]` when set |
| **weight_decay** | AdamW, 8-bit, Optimi, GenericOptim | L2-style decay |
| **momentum** | SGD | Typical default `0.9` |
| **gradient_release** | Trainer special key | `true` / `false`; requires `pipeline_stages = 1` |
| **beta2_half_life** | Trainer special key | Recompute `betas[1]` from global batch size |
| **kahan_buffer_offload** | **genericoptim**, **adamw8bitkahan** | Offload Kahan buffer to CPU (saves VRAM) |
| **d0**, **d_coef** | **prodigy** | D-adaptation initial estimate and scale; tune `d_coef` (not `lr`) to force larger/smaller adaptive LR |
| **weight_decouple**, **bias_correction**, **safeguard_warmup** | **prodigy** | `pytorch-optimizer` kwargs; diffusion-friendly defaults are pre-filled in the form |
| **eps** | **adafusion**, **adamuon** | Two floats `[eps_factored, eps_clip]` (default `[1e-30, 1e-3]`) |
| **clip_threshold** | **adafusion**, **adamuon** | RMS ceiling on the normalized update (default `1.0`). Internal / load-bearing — an Adafactor-style RMS clip, **not** the DeepSpeed `gradient_clipping` grad-norm clip; leave at `1.0` |
| **momentum_dtype** | **adafusion**, **muon**, **adamuon** | Momentum storage: `"float32"`, `"bfloat16"`, `"int8"`, or `"4bit"`. bf16 keeps state at ~2 B/param (int8 ~1 B, 4bit ~0.5 B). For Adafusion, set `betas` to `[0.0, …]` for a true no-momentum (lowest-memory) run |
| **cautious** | **adafusion**, **adamuon** | Cautious update masking; helps with momentum (on by default for AdaMuon — flips it from a loss to a win vs Adafusion). For Adafusion, set `false` when `betas[0] = 0.0` (no momentum), where it's a no-op |
| **ns_steps** | **adamuon** | Newton-Schulz orthogonalization steps (default `2`). `2` is the validated sweet spot for diffusion; `5` (LLM Muon) over-orthogonalizes |
| **bf16_method** | **adafusion**, **muon**, **adamuon** | bf16-correct weight update: `"stochastic_rounding"` (no extra buffer), `"kahan"`, or `"none"` |
| **compile** | **adamuon** | Optional whole-step `torch.compile` (off by default; workload-dependent) |
| **adamw_lr** | **muon** | LR for Muon's AdamW fallback on 1-D / embedding params (default `3e-4`) |

For the full Adafusion / Muon / AdaMuon parameter set (e.g. `foreach` batching, `momentum_4bit_block`, Muon's `nesterov`), see the [koptim docs](https://github.com/Koronos/K-Optimizers/tree/main/docs). Any key under `[optimizer]` is forwarded to the constructor, so unlisted kwargs work too.

`gradient_release` requires **`pipeline_stages = 1`** and data-parallel world size 1.

### Fully-qualified path

To use an optimizer that is not built-in (e.g. from `pytorch_optimizer` or a custom package), set `type` to the **fully-qualified class path** and add constructor kwargs in the KV list / TOML:

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
decouple = true
```

All keys under `[optimizer]` except `type` are passed as keyword arguments to the optimizer constructor (plus trainer handling for `gradient_release` and `beta2_half_life`).

---

## Learning rate scheduler

Set **`lr_scheduler`** at the top level of your config (or omit it; default is `"constant"`).

### Built-in names

- **constant** — Constant learning rate (no decay).
- **linear** — Linear decay from initial LR to 0 over training.
- **cosine** — Cosine annealing; minimum LR via `lr_min` in `[lr_scheduler_args]`.
- **rex** — REX reflected-exponential decay (Chen et al. 2021); decays slowly early and faster near the end. With remaining fraction `z = 1 - step/total_steps`, the multiplier is `z / ((1−d) + d·z)` (1.0 at the start, 0.0 at the end) and `lr = lr_min + (base_lr − lr_min)·multiplier`. Tune the curve via `rex_d` in `[lr_scheduler_args]` (`0.0` = linear, `0.5` = canonical REX (default), `→1.0` = holds LR higher then drops sharply) and the floor via `lr_min`. Often a strong default for short-budget LoRA runs.
- **none** — No scheduler (optimizer LR is used as-is).

In the training config form, **Scheduler parameters** is a single key-value list for every scheduler type (built-in and custom). Rows map to TOML as follows:

| KV key | Written to TOML |
|--------|-----------------|
| `warmup_steps` | Top-level `warmup_steps` |
| Any other key | `[lr_scheduler_args].<key>` |

Example:

```toml
lr_scheduler = "cosine"
warmup_steps = 100

[lr_scheduler_args]
lr_min = 0.0
```

### Built-in form pre-fill (scheduler KV)

When you pick a built-in name in the form, common keys are pre-filled (edit as needed):

| `lr_scheduler` | Pre-filled keys (values) |
|----------------|--------------------------|
| **none** | (empty) |
| **constant** | `factor` → `1.0`, `warmup_steps` → `0` |
| **linear** | `start_factor`, `end_factor`, `total_iters` → `total_steps`, `warmup_steps` → `0` |
| **cosine** | `lr_min` → `0.0`, `warmup_steps` → `0` |
| **rex** | `lr_min` → `0.0`, `rex_d` → `0.5`, `warmup_steps` → `0` |

Built-in training code uses fixed defaults for some keys today (e.g. linear decay uses full `total_steps`); extra `[lr_scheduler_args]` keys are stored for clarity and forward compatibility.

### Runtime tokens in `[lr_scheduler_args]`

You can use these **string tokens** as values in `[lr_scheduler_args]` or in the form’s **Scheduler parameters** list. They are replaced with integers when training starts. In the training config form, open the **(i)** help on **Scheduler parameters** for a short glossary, or use the table below.

| Token | Resolved value |
|-------|----------------|
| **`total_steps`** | `epochs × steps_per_epoch` (optimizer steps per full training run). |
| **`effective_total_steps`** | `min(total_steps, max_steps)` when `max_steps` is set in config; otherwise same as `total_steps`. Use for `T_max` / `total_iters` when training may stop early. |
| **`steps_per_epoch`** | Training steps in one epoch (after gradient accumulation). |
| **`epochs`** | `epochs` from your config. |
| **`max_steps`** | Only substituted when `max_steps` is set in config. |
| **`gradient_accumulation_steps`** | Only substituted when set in config. |

Tokens are most useful for **fully-qualified** PyTorch schedulers (`T_max`, `total_iters`, etc.). Built-in short names accept the same tokens in TOML where applicable.

Example (custom PyTorch scheduler):

```toml
lr_scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"

[lr_scheduler_args]
T_max = "effective_total_steps"
eta_min = 0.0
```

With an early stop cap:

```toml
epochs = 10
max_steps = 5000

[lr_scheduler_args]
T_max = "effective_total_steps"
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

### Suggested PyTorch schedulers (form pre-fill)

When you pick one of these fully-qualified types in the training config form, **Scheduler parameters** is pre-filled with common constructor keys (edit as needed):

| Scheduler | Pre-filled keys (values) |
|-----------|---------------------------|
| `torch.optim.lr_scheduler.CosineAnnealingLR` | `T_max` → `effective_total_steps`, `eta_min` → `0.0` |
| `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` | `T_0` → `steps_per_epoch`, `T_mult` → `2`, `eta_min` → `0.0` |
| `torch.optim.lr_scheduler.StepLR` | `step_size` → `steps_per_epoch`, `gamma` → `0.1` |
| `torch.optim.lr_scheduler.MultiStepLR` | `milestones` → `[50]`, `gamma` → `0.1` |
| `torch.optim.lr_scheduler.OneCycleLR` | `max_lr`, `total_steps` → `effective_total_steps`, `pct_start` |
| `torch.optim.lr_scheduler.ExponentialLR` | `gamma` → `0.95` |
| `torch.optim.lr_scheduler.PolynomialLR` | `total_iters` → `effective_total_steps`, `power` → `1.0` |
| `torch.optim.lr_scheduler.LinearLR` | `start_factor`, `end_factor`, `total_iters` → `effective_total_steps` |
| `torch.optim.lr_scheduler.ConstantLR` | `factor`, `total_iters` → `effective_total_steps` |

Fully-qualified schedulers also get `warmup_steps` → `0` in the pre-fill when applicable.

---

## Warmup

To add a short linear warmup before the main scheduler, set **`warmup_steps`** at the top level of your config (default is 0), or add a `warmup_steps` row in **Scheduler parameters** in the form. The framework wraps the chosen scheduler with a warmup phase: LR goes from `initial_lr / warmup_steps` to `initial_lr` over `warmup_steps` steps, then the main scheduler runs.

This applies to **built-in** schedulers and **fully-qualified** PyTorch classes: warmup is a trainer-level wrap (`SequentialLR`), not a constructor argument under `[lr_scheduler_args]`. Custom schedulers do **not** receive `warmup_steps` in `**kwargs`; if your class has its own warmup parameter, use a different name in `[lr_scheduler_args]` or rely on this top-level wrap.

This has no effect if `lr_scheduler` is `"none"` or if `warmup_steps` is 0.
