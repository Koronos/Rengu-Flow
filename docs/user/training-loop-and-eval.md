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
| **`val_gap_enable`** | Enable the deterministic **generalization probe**: a held-out validation loss plus the **train–val gap** (see below). | `true` or `false`. | `true` |
| **`val_gap_probe_batches`** | Forward batches per probe (per timestep quantile). Keeps the probe cheap regardless of dataset size. | Positive integer. | `8` |

### Generalization probe (train–val gap)

Train loss alone is misleading for diffusion fine-tuning: a model can drive train loss down while overfitting/memorizing and producing worse samples. The recognized cheap signal (EveryDream2 / kohya / OneTrainer) is a **held-out validation loss**, and especially the **train–val gap** (rising gap = overfitting).

When `val_gap_enable` is on and at least one `eval_datasets` entry is configured, the trainer runs a deterministic, forward-only probe on the existing eval cadence (`eval_every_n_*`):

- **`val/loss`** — held-out validation loss on the first eval dataset.
- **`train/probe`** — the same probe on a small fixed train subset.
- **`val/gap`** — `val/loss − train/probe`, the headline overfitting signal.

The probe is deterministic — the timestep is fixed per pass (averaged over a fixed spread of quantiles) and the per-item noise is frozen by reseeding before every probe — so the curves are smooth and comparable across steps. It reuses the model's own training loss (eps-pred / v-pred / flow-matching), runs under `torch.no_grad()` with the model in inference state, and restores training state after. There is **no sampling/generation** (forward passes only). All three scalars plot in TensorBoard alongside `train/loss`, and `val/loss` + `val/gap` are surfaced live in the UI next to the train loss. If no `eval_datasets` are configured, the probe no-ops gracefully (no crash).

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
- **preview/** — Sample images when `[preview]` is configured or the `preview_now` signal is used. See [Training previews](previews.md).

View with:

```bash
tensorboard --logdir output
```

### Experiment tracking and WandB (optional)

Tracking is configured under a **`[tracking]`** section. One sink fans out to the
backends you list; the local store (TensorBoard event files, `run.json`,
`run_events.jsonl`) lives in the run directory. WandB is **opt-in** by adding
`"wandb"` to `backends`.

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`[tracking]`** | Section for experiment tracking. | Table. | — |
| **`tracking.enabled`** | Master switch. `false` is the full disconnect (no-op sink). | `true` or `false`. | `true` |
| **`tracking.backends`** | Backends the sink writes to. Add `"wandb"` to enable Weights & Biases. | List of `"manifest"`, `"tensorboard"`, `"wandb"`. | `["manifest", "tensorboard"]` |
| **`tracking.wandb.project`** | WandB project name. | String. | `"rengu-flow"` |
| **`tracking.wandb.run_name`** | Run name in WandB. If omitted, the run directory name is used. | String or omit. | `null` |
| **`tracking.wandb.api_key`** | API key for WandB (or set `WANDB_API_KEY` env). | String or omit. | `null` |

Example:

```toml
[tracking]
backends = ["manifest", "tensorboard", "wandb"]

[tracking.wandb]
project = "my-project"
run_name = "sdxl-lora-v1"
# api_key = "..."   # optional if WANDB_API_KEY is set
```

### X-axis (steps vs examples)

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`x_axis_examples`** | Use total examples (instead of step count) as the x-axis for all scalar logs. | `true` or `false`. | `false` |

## Resume from checkpoint

To continue training from the latest run under `output_dir`:

```bash
rengu train --config my.toml --resume-from-checkpoint
```

To resume from a specific run folder:

```bash
rengu train --config my.toml --resume-from-checkpoint 20250218_12-00-00_myrun
```

Checkpoint restores model, optimizer, LR scheduler, and dataloader state (epoch and position). Optional flags:

- **`--reset_dataloader`** — Do not restore dataloader state; only restore epoch number. Useful if you changed the dataset.
- **`--reset_optimizer`** — Do not restore optimizer state (e.g. to change optimizer).
- **`--reset_optimizer_params`** — Restore optimizer state but reset param groups (e.g. learning rate) from config.

## Pipeline, cache, and debug options

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`pipeline_stages`** | Number of pipeline-parallel stages (usually matches GPU count). `"deepspeed"` engine only. | Positive integer. | `1` |
| **`partition_method`** | How transformer layers are assigned to stages. | `parameters` (balance by param count), `uniform`, or `manual`. | `parameters` |
| **`partition_split`** | Layer indices for manual partitioning. | JSON list of integers. | Omitted (required when `partition_method = manual`). |
| **`steps_per_print`** | How often DeepSpeed prints step timing to the console. | Positive integer. | `1` |
| **`synthetic_num_batches`** | Train on in-memory fake SDXL batches (no real dataset). | Positive integer or omit. | Omitted (use real data). |
| **`caching_batch_size`** | Batch size during the dataset cache phase (latents + text embeddings). | Positive integer. | `1` |
| **`cache_root`** | Root folder for all v2 dataset caches (metadata, latents, text embeddings). | Path string. | `cache/` under the install directory (gitignored). |
| **`cache_num_proc`** | Parallel CPU preprocessing threads for metadata map and latent/embedding cache (image load/decode/resize runs on these threads; GPU encode stays on the main process). | Positive integer. | `min(8, CPU count)` |
| **`cache_keep_in_memory`** | Keep the HuggingFace dataset slice in RAM while resuming cache. | `true` / `false`. | `false` (lower RAM; OS page cache still helps train reads) |
| **`cache_dedup_text_embeddings`** | During `--cache_only`, reuse text-encoder outputs when captions are identical (hash dedup). | `true` or `false`. | `false` |
| **`dataloader_num_workers`** | PyTorch DataLoader workers for training (load cached latents from disk). | Non-negative integer. | `0` |
| **`dataloader_prefetch`** | Background thread loads the next raw batch while the GPU trains (only when `dataloader_num_workers = 0`). Off, the load runs synchronously and stalls the GPU every step. | `true` / `false`. | `true` |
| **`dataloader_pin_memory`** | Page-locked CPU memory for faster host→GPU copies when using CUDA. | `true` / `false`. | `false` |
| **`dataloader_prefetch_factor`** | Batches prefetched per worker when `dataloader_num_workers > 0`. | Positive integer. | `2` |
| **`dataloader_persistent_workers`** | Keep DataLoader worker processes alive between epochs. | `true` / `false`. | `true` |
| **`image_micro_batch_size_per_gpu`** | Micro-batch for image-only steps when mixing modalities. | Integer or dict, or omit to use `micro_batch_size_per_gpu`. | Same as `micro_batch_size_per_gpu` |

**Disk hygiene:** Dataset cache lives under **`cache_root`** / `<dataset_id>` / `<directory_id>` / `<model_name>/` (see **`cache_root`** above). Each bucket stores `manifest.json`, `tensors/*.bin`, and `meta.db` under `latents/` and `text_embeddings_*`. A legacy v1 cache is rejected — use `--regenerate_cache`. GPU smokes via `scripts/run_model_smoke.sh` delete `output/` and fixture caches under the default **`cache_root`** by default. Set `KEEP_SMOKE_ARTIFACTS=1` to keep them for inspection.

Developer notes (POC benchmarks, v2 layout): [performance-cpu-ram](../developer/performance-cpu-ram.md), [dataset and cache](../developer/dataset-and-cache.md).

**Already tuned for speed/VRAM (see also [Cosmos performance](training-cosmos-predict2-lora-lokr-finetune.md#performance-and-vram-anima--cosmos)):** run `--cache_only` once before long jobs; `compile=true` for long runs; `cache_text_embeddings=true`; `RENGU_TUNING_TF32_APPLY=1` when supported; keep dataset cache on SSD; use `--trust_cache` to resume without re-encoding.

**Compare dataloader flags (GPU):** `scripts/smoke_perf_ab.sh sdxl [prefetch|workers2]` — developer details in [performance-cpu-ram](../developer/performance-cpu-ram.md).

Example (4-GPU pipeline):

```toml
pipeline_stages = 4
partition_method = "parameters"
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 4
```

**When does `partition_method` apply?** Only with **`pipeline_stages > 1`** — i.e. *pipeline
parallelism*, where the model's transformer layers are split into stages that run on **different
GPUs**. On a single GPU (`pipeline_stages = 1`, the default) there is only one stage, so
`partition_method` is ignored. When there is more than one stage, the method decides which layers
land on which stage:

- **`parameters`** (default) — give each stage a similar **parameter count** so VRAM is balanced.
  Because layers differ in size, this rarely means an equal number of layers per stage — but it is
  usually the best balance for memory.
- **`uniform`** — give each stage the **same number of layers**. Simpler, but stages can end up
  with uneven VRAM if some layers are much heavier than others.
- **`manual`** — you choose the boundaries yourself via **`partition_split`**, a JSON list of layer
  indices where each stage ends. For example `partition_split = [10, 20]` with `pipeline_stages = 3`
  puts layers 0–9 on stage 0, 10–19 on stage 1, and 20+ on stage 2.

To reduce VRAM on a **single GPU**, pipeline parallelism is not the tool — use
[block swap](#block-swap-vram-adapter-training) (`blocks_to_swap`) plus activation checkpointing
instead.

## Activation checkpointing

Saves VRAM by recomputing activations in the backward pass. Configure in the main TOML:

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`activation_checkpointing`** | Enable and choose implementation. | `false`, `true`, or `"auto"`. | `false` |
| **`reentrant_activation_checkpointing`** | When `activation_checkpointing = true`, use reentrant PyTorch checkpoint. | `true` or `false`. | `false` (`true` auto-default for `cosmos_predict2` when AC is on — see [Cosmos/Anima guide](training-cosmos-predict2-lora-lokr-finetune.md#performance-and-vram-anima--cosmos)) |

- **`true`** — PyTorch `torch.utils.checkpoint.checkpoint`. Use `reentrant_activation_checkpointing = true` if you hit errors with block swap or certain layers. For **Cosmos/Anima**, keeping it `true` is recommended (~3% faster in LoKR tuning vs `false`).
- **`"auto"`** — compiler-driven (requires `compile = true`): Inductor's memory-budget partitioner picks the optimal save/recompute split per compiled graph; dial it with **`activation_memory_budget`** (0.0 ≈ full-checkpoint VRAM, 1.0 ≈ no-checkpoint speed, default 0.3). Exact recompute — no precision cost. Measured @1024 LoKr it beats `"selective"` on speed AND VRAM at budget 0.1, and reaches −21% step time at 0.5. See [Cosmos guide](training-cosmos-predict2-lora-lokr-finetune.md#performance-and-vram-anima--cosmos).
- Retired values: **`"selective"`** (SAC) and **`"unsloth"`** fall back to `true` with a warning — `"auto"` measured faster AND lighter than SAC (see `docs/EXPERIMENTS_GRAVEYARD.md`).

If a run OOMs, follow the [VRAM ladder](#if-it-doesnt-fit-the-vram-ladder) below.

### If it doesn't fit: the VRAM ladder

Model-agnostic: every step applies to any supported model (Cosmos, SDXL, ...). When a run OOMs, enable these **in order** — each trades a little speed (or setup) for memory, cheapest first. Stop as soon as it fits; the steps compose freely.

1. **`cache_text_embeddings = true` + run `--cache_only` once** (default on for Cosmos/SDXL) — keeps the text encoder out of the training graph entirely.
2. **`activation_checkpointing = true`** — the big one for activations (Cosmos @1024 LoKr: OOM >15.5 GB → 5.76 GB). If you are on `"auto"`, first **lower `activation_memory_budget`** (0.5 → 0.3 → 0.1 → 0.0) before falling back to `true` — budget 0.1 is still faster than full checkpointing at almost the same VRAM.
3. **`micro_batch_size_per_gpu = 1` + `gradient_accumulation_steps = N`** — same effective batch, activations of one sample at a time.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (native Linux only, not WSL2) — no math change; fights allocator fragmentation, which multi-res shape changes aggravate.
5. **Memory-efficient optimizer states** — the biggest lever for **full finetune**, where AdamW's fp32 states cost 8 bytes/param (~16 GB on a 2B model). Prefer the in-house [kaon optimizers](optimizer-and-scheduler.md) (installed on demand via the `kaon` profile): factored/quantized state at **~1–2 bytes/param** with AdamW-quality, bf16-correct updates — `adakaon` (the workhorse), `adapnm` (loss↔gap dial, optional fused Triton step), `nekaon` (flat-minima flagship, 4-bit momentum by default) — down to **~0.5 bytes/param** with `lion` (no second moment). Alternative outside kaon: `optimizer.type = "adamw8bit"` (bitsandbytes, ~4 B/param). For LoRA/LoKr the states are tiny either way and this step does little.
6. **`blocks_to_swap = N`** — stream the model's blocks from CPU (Cosmos: `transformer.blocks`; SDXL: UNet down/mid/up blocks). Start near half the block count and raise until it fits. For **full finetune** also set `optimizer.gradient_release = true` (required). This is the step that fits a 2B full finetune on a 16 GB card.
7. **Extreme case**: kaon `lion` with `momentum_dtype = "4bit"` (the absolute-minimum optimizer state) or `genericoptim` with `cpu_offload` — last resort.

Not on the ladder: lowering resolution, steps, or what you train — those change the result, not the footprint of producing it. Model-specific measured numbers live in the [Cosmos guide](training-cosmos-predict2-lora-lokr-finetune.md#performance-and-vram-anima--cosmos) and [VRAM optimization](../developer/vram-optimization.md).

Example:

```toml
activation_checkpointing = true
reentrant_activation_checkpointing = false
```

## Block swap (VRAM, adapter training)

Offloads UNet or DiT blocks to CPU between forward steps. Shared implementation for **SDXL** and **Cosmos Predict2** ([developer reference](../developer/training-techniques.md)).

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **`blocks_to_swap`** | Number of backbone blocks kept on CPU between steps (higher = less VRAM, slower steps). | Non-negative integer; `0` disables. | `0` |
| **`disable_block_swap_for_eval`** | Load full backbone on GPU during eval. | `true` or `false`. | `false` |
| **`disable_block_swap_for_preview`** | Load full backbone on GPU during preview sampling. | `true` or `false`. | Same as eval default |

**Requirements:** `[adapter]` must be set (LoRA/LoKr). **`pipeline_stages = 1`**. Do not use for full-model finetune (omit `[adapter]` and leave `blocks_to_swap` at `0`).

Example (Cosmos LoKr on ~16 GB):

```toml
[adapter]
type = "lokr"
rank = 6

blocks_to_swap = 16
activation_checkpointing = true
pipeline_stages = 1
```

## EMA shadow weights (optional)

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **`ema_decay`** | Exponential moving average of trainable weights (stored on CPU). | Float in `(0, 1)`, e.g. `0.999`. | Omitted (disabled) |

EMA updates run after each successful training step. Export of EMA weights is not automatic today — use for monitoring or future export hooks.

## Skipping batches on CUDA OOM (optional)

When a single training step runs out of GPU memory (e.g. after a resolution bucket change), you can skip that step and continue instead of aborting the whole run. This matches the behaviour described in many flow-model trainers (see developer doc [training-step-skip-on-oom](../developer/training-step-skip-on-oom.md)).

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`[train.oom_skip]`** | Optional section. | Table. | Omitted (`enabled` = false). |
| **`train.oom_skip.enabled`** | Catch CUDA OOM around each training step. | `true` or `false`. | `false` |
| **`train.oom_skip.max_consecutive`** | Abort after this many OOM skips **in a row**. | Integer ≥ 1. | `3` |
| **`train.oom_skip.clear_cache_on_skip`** | Call CUDA cache flush helpers after a skip. | `true` or `false`. | `true` |

Example: `examples/config_oom_skip.toml`.

**Note:** Intended for single-GPU runs (`pipeline_stages = 1`). Multi-GPU training may desynchronize if only one rank OOMs; disable OOM skip on multi-GPU until your setup documents otherwise.
