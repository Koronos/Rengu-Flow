# Training loop and evaluation (developer guide)

Where the training loop lives, how **evaluation** is invoked, where metrics are written (TensorBoard / WandB), and how this integrates with the **Saver** and **signal files**.

**Block swap:** [`rengu_flow/training/block_swap.py`](../../rengu_flow/training/block_swap.py) via `BasePipeline.enable_block_swap` — SDXL UNet blocks and Cosmos `transformer.blocks`. See [training-techniques.md](training-techniques.md).

**OOM step skip** is implemented — see [training-step-skip-on-oom.md](training-step-skip-on-oom.md) and `rengu_flow/utils/oom_skip.py`.

## Execution flow

1. **Config load** — Main TOML and dataset TOML(s); defaults and validation.
2. **Distributed init** — `_distributed_init(args)`, `deepspeed.init_distributed()`.
3. **Model and data** — Model from registry; train (and optional eval) datasets; cache if real data.
4. **Pipeline** — `model.to_layers()` → `ManualPipelineModule`; activation checkpointing (PyTorch checkpoint or compile-budget "auto"); `deepspeed.initialize`; `_configure_optimizer`.
5. **Dataloaders** — `PipelineDataLoader` for train and for each eval dataset (`eval_data_map`).
6. **Resume** — If `resume_from_checkpoint`, `model_engine.load_checkpoint(run_dir, ...)`; restore `step`, `examples`, `train_dataloader.state_dict()` (or epoch only if `--reset_dataloader`).
7. **Loop** — Per step: `get_data_iterator_for_step` → `model_engine.train_batch(iterator)` → `train_dataloader.sync_epoch()`; then, on an epoch boundary from the single `EpochSchedule` authority, `saver.process_epoch_boundary(completed_epoch, …)`, plus `saver.process_step`; optional `evaluate()` and `run_previews()`; TensorBoard/WandB logging. Epoch numbers (naming, save/eval cadence, progress, termination) all come from `EpochSchedule` (step-based budget), not the dataloader's own counter.
8. **Exit** — Always write a final resume checkpoint (unless one was already written at this exact step) and the final model; print completion.

Code entry: **`rengu_flow/main.py`** — `_run_training(args, config)`.

## Evaluation

- **Module:** `rengu_flow.utils.eval`.
- **Constants:** `TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, ..., 0.9]`.
- **Functions:**
  - **`evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None, max_batches=None)`** — Sets `eval_dataloader.set_eval_quantile(quantile)`, runs a loop: `get_data_iterator_for_step` → `model_engine.eval_batch(iterator, num_micro_batches=...)` → `eval_dataloader.sync_epoch()`, until `eval_dataloader.epoch == 2` (or `max_batches` reached); then `eval_dataloader.reset()`; returns mean loss.
  - **`_evaluate(model_engine, eval_dataloaders, step, eval_gradient_accumulation_steps, sink)`** — For each dataset and each quantile, calls `evaluate_single`; logs `{name}/loss_quantile_{q}`, `{name}/loss`, `eval/eval_time_sec` via the tracking `sink`.
  - **`evaluate(model, model_engine, eval_dataloaders, sink, step, eval_gradient_accumulation_steps, disable_block_swap, optimizer=None)`** — If no eval dataloaders, returns. Otherwise: optional `optimizer.eval()`; `empty_cuda_cache()`; `model.prepare_block_swap_inference(disable_block_swap)`; `torch.no_grad()` + `isolate_rng()` (fixed seed per rank); `_evaluate(...)`; `empty_cuda_cache()`; `model.prepare_block_swap_training()`; optional `optimizer.train()`.

**When `evaluate()` is called:**

- Once before the first step if `eval_before_first_step` and not resuming and `eval_dataloaders` non-empty.
- In the loop when `eval_every_n_steps` and `step % eval_every_n_steps == 0`, or when a full epoch finished and `eval_every_n_epochs` and `epoch % eval_every_n_epochs == 0`.

**Reproducibility:** Evaluation runs inside `isolate_rng()` and sets `random`, `torch`, and `numpy` seeds to `get_rank()` so each rank has deterministic eval without affecting the training RNG after the context exits. See **`rengu_flow.utils.isolate_rng`**.

**Loader reset:** Between quantiles, `PipelineDataLoader.reset()` is used so the same eval dataloader can be reused: it sets `epoch=1`, `num_batches_pulled=0`, `next_micro_batch=None`, and reinitializes the internal batch iterator. Implemented in **`rengu_flow/data/loader.py`**.

**Block swap during eval:** When `blocks_to_swap > 0`, `evaluate()` calls `model.prepare_block_swap_inference(disable_block_swap)` where `disable_block_swap` comes from top-level config `disable_block_swap_for_eval` (default `false`). Same pattern for previews via `disable_block_swap_for_preview` in **`rengu_flow.utils.preview.run_previews`**.

User-facing option tables: **`docs/user/training-loop-and-eval.md`** (Evaluation and Logging sections).

## DeepSpeed pipeline and training data modes

**Pipeline construction** (`rengu_flow/main.py` after `model.to_layers()`):

| Config key | Code location | Notes |
|------------|---------------|--------|
| `pipeline_stages` | `num_stages` → `ManualPipelineModule(..., num_stages=...)` | Default `1`. Should match GPU count for pipeline parallel. |
| `partition_method` | `ManualPipelineModule(..., partition_method=...)` | Passed to DeepSpeed `PipelineModule._partition_layers`. Values: `parameters`, `uniform`, `manual`. Default from `set_config_defaults`: `parameters`. |
| `partition_split` | `manual_partition_split=` when `partition_method == "manual"` | List of layer indices (length `pipeline_stages - 1`). If omitted and `num_stages > 1`, defaults to even split: `[len(layers) // num_stages] * (num_stages - 1)`. |
| `activation_checkpointing` | `activation_checkpoint_interval`, `checkpointable_layers`, `activation_checkpoint_func` | `true` → PyTorch checkpoint (`reentrant_activation_checkpointing` → `use_reentrant`); `"auto"` → no wrapper, Inductor memory-budget partitioner (`training/activation_budget.py`). Retired: `"selective"`/`"unsloth"` degrade to `true`. |
| `steps_per_print` | `ds_config["steps_per_print"]` | DeepSpeed console interval. Default `1` in defaults. |
| `micro_batch_size_per_gpu` | `train_micro_batch_size_per_gpu` in DeepSpeed config | If value is a dict, first value is used for DS init (image-specific dict handled later for dataloaders). |
| `gradient_accumulation_steps` | `ds_config` + dataloader `post_init` | |
| `gradient_clipping` | Set to `0.0` when `optimizer.gradient_release` is true | |

**`ManualPipelineModule`** (`rengu_flow/utils/pipeline.py`): subclasses `deepspeed.pipe.PipelineModule`. When `partition_method.lower() == "manual"`, uses `manual_partition_split` to set stage boundaries and prints per-stage layer names on rank 0; otherwise delegates to DeepSpeed’s built-in methods.

**Real vs synthetic training data:**

| Condition | Behavior |
|-----------|----------|
| `dataset` set, `synthetic_num_batches` omitted | Load dataset TOML, `Dataset` + `DatasetManager.cache()`, train on cached latents/embeddings. |
| `synthetic_num_batches` set | Skip real data path; use **`SyntheticSDXLDataset`** (`rengu_flow/data/synthetic.py`) with `num_batches` from config (default `50` in code if key present without value handling — see `main.py`). Dataset TOML still copied into run dir but not used for training iterators. |
| No `dataset` | Depends on validation; typical examples always set `dataset`. |

**`caching_batch_size`:** Passed to `DatasetManager(..., caching_batch_size=...)` and into worker `_cache_fn` / `_map_and_cache` (`rengu_flow/data/cache_utils.py`: `pool.imap(..., batch_size=caching_batch_size)`). Default `1` in `set_config_defaults`.

**`image_micro_batch_size_per_gpu`:** After DeepSpeed init, `train_data.post_init(..., per_device_batch_size_image=...)` receives either the top-level int or a dict keyed by modality (`main.py` normalizes non-dict to `{None: value}`). Used when mixing image and video buckets.

**Examples axis:** `x_axis_examples` in config selects whether `log_training_step` and eval/preview logging use `examples` or `step` as the TensorBoard/WandB x-coordinate (`training_metrics.py` / loop in `main.py`).

## Tracking sink (`rengu_track`)

All metrics, images, and lifecycle events are routed through a single **tracking sink** built from
the external `rengu_track` package, not direct TensorBoard/WandB calls. On rank 0, `main.py` builds
`sink = build_sink(config, run_dir)`; other ranks get a no-op `NullSink()`. Configuration is under
`config["tracking"]` (see `set_config_defaults`):

| Key | Default | Role |
|-----|---------|------|
| `tracking.enabled` | `true` | `false` is the full disconnect (no-op sink). |
| `tracking.backends` | `["manifest", "tensorboard"]` | Fan-out targets. Add `"wandb"` to enable WandB. |
| `tracking.capture_lineage` | `true` | `sink.set_lineage(...)` / `sink.set_hardware(...)`. |
| `tracking.system_sampler.enabled` / `.interval_sec` | `true` / `10` | Background system-metrics sampler. |
| `tracking.wandb.project` / `.run_name` / `.api_key` | `"rengu-flow"` / `None` / `None` | WandB backend options. |

The local store in `run_dir` is the manifest (`run.json`), TB event files, and the event log
(`run_events.jsonl`). Lifecycle events use `sink.event(...)` (e.g. `EVENT_FINISHED`, `EVENT_FAILED`,
`EVENT_STOP_REQUESTED`) and `sink.close(status=...)`. The web UI reads the manifest
(`rengu_track.read_manifest`) rather than a separate status file; see **`docs/developer/web-ui.md`**.

## Where metrics are written

Training step logging via **`rengu_flow.utils.training_metrics.log_training_step`** emits through the
sink: `train/loss`, `train/grad_norm`, `train/prodigy_d` (Prodigy), `train/automagic_avg_lr` and
histogram `train/automagic_lrs` (Automagic / GenericOptim). Epoch: `train/epoch_loss`. Eval:
`{name}/loss_quantile_{q}`, `{name}/loss`, `eval/eval_time_sec` (`sink.scalar(...)` in `_evaluate`).
X-axis is `examples` if `x_axis_examples` else `step`. The configured `backends` decide whether each
value lands in the manifest, TensorBoard, and/or WandB — the loop and `_evaluate` call `sink.scalar`
/ `sink.histogram` / `sink.image` once, regardless of backend.

## Saver and signal files

- **Saver** (`rengu_flow.utils.saver.Saver`): `process_epoch_boundary` / `process_step` handle scheduled checkpoint/export, `max_checkpoints_to_keep` pruning after `save_checkpoint`, and **signal files** via `process_signals()`:
  - **`save` / `save_quit`** — DeepSpeed resume checkpoint; quit on `save_quit`.
  - **`export_model` / `export_model_quit`** — Export to `signal_step<N>/`; quit on `export_model_quit`.
- See **`docs/user/checkpoint-and-save.md`**, **`docs/user/signal-files.md`**, and **`docs/developer/signal-files.md`**.
- **Previews**: **`rengu_flow.utils.preview.run_previews`** when `[preview]` is configured or `signals.should_preview`; see **`docs/user/previews.md`** and **`docs/developer/previews.md`**.

## Config and eval dataset loading

- Eval datasets are built in `_run_training` from `config.get("eval_datasets", [])`. Each entry is passed to **`load_eval_dataset_config`** (`rengu_flow.config.loader`): string → `(name, dataset_config)` using path stem as name; dict with `name` and `config` → load TOML at `config`. Then `Dataset(eval_dataset_config, model, ...)` is created, registered with `DatasetManager` (so cache includes eval data), and after `model_engine` exists, `post_init` is called for each eval dataset. **`eval_dataloaders`** is `{ name: PipelineDataLoader(eval_data, model_engine, eval_gradient_accumulation_steps, model, 0) for name, eval_data in eval_data_map.items() }`.
