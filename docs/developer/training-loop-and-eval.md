# Training loop and evaluation (developer guide)

Where the training loop lives, how **evaluation** is invoked, where metrics are written (TensorBoard / WandB), and how this integrates with the **Saver** and **signal files**.

**`[TODO]` (not in this module):** `enable_block_swap` on SDXL (`main.py` calls it when `blocks_to_swap` is set, but `SDXLPipeline` inherits `BasePipeline.enable_block_swap` → `NotImplementedError`). OOM step skip — see [training-step-skip-on-oom.md](training-step-skip-on-oom.md).

## Execution flow

1. **Config load** — Main TOML and dataset TOML(s); defaults and validation.
2. **Distributed init** — `_distributed_init(args)`, `deepspeed.init_distributed()`.
3. **Model and data** — Model from registry; train (and optional eval) datasets; cache if real data.
4. **Pipeline** — `model.to_layers()` → `ManualPipelineModule`; activation checkpointing (PyTorch or unsloth); `deepspeed.initialize`; `_configure_optimizer`.
5. **Dataloaders** — `PipelineDataLoader` for train and for each eval dataset (`eval_data_map`).
6. **Resume** — If `resume_from_checkpoint`, `model_engine.load_checkpoint(run_dir, ...)`; restore `step`, `examples`, `train_dataloader.state_dict()` (or epoch only if `--reset_dataloader`).
7. **Loop** — Per step: `get_data_iterator_for_step` → `model_engine.train_batch(iterator)` → `train_dataloader.sync_epoch()`; then `saver.process_epoch` / `saver.process_step`; optional `evaluate()` and `run_previews()`; TensorBoard/WandB logging.
8. **Exit** — Save final checkpoint and model if not already saved; print completion.

Code entry: **`renga_flow/main.py`** — `_run_training(args, config)`.

## Evaluation

- **Module:** `renga_flow.utils.eval`.
- **Constants:** `TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, ..., 0.9]`.
- **Functions:**
  - **`evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None)`** — Sets `eval_dataloader.set_eval_quantile(quantile)`, runs a loop: `get_data_iterator_for_step` → `model_engine.eval_batch(iterator, num_micro_batches=...)` → `eval_dataloader.sync_epoch()`, until `eval_dataloader.epoch == 2`; then `eval_dataloader.reset()`; returns mean loss.
  - **`_evaluate(model_engine, eval_dataloaders, step, eval_gradient_accumulation_steps, tb_writer, wandb_enable)`** — For each dataset and each quantile, calls `evaluate_single`; logs `{name}/loss_quantile_{q}`, `{name}/loss`, `eval/eval_time_sec` to TensorBoard and optionally WandB.
  - **`evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap, optimizer=None, wandb_enable=False)`** — If no eval dataloaders, returns. Otherwise: optional `optimizer.eval()`; `empty_cuda_cache()`; `model.prepare_block_swap_inference(disable_block_swap)`; `torch.no_grad()` + `isolate_rng()` (fixed seed per rank); `_evaluate(...)`; `empty_cuda_cache()`; `model.prepare_block_swap_training()`; optional `optimizer.train()`.

**When `evaluate()` is called:**

- Once before the first step if `eval_before_first_step` and not resuming and `eval_dataloaders` non-empty.
- In the loop when `eval_every_n_steps` and `step % eval_every_n_steps == 0`, or when a full epoch finished and `eval_every_n_epochs` and `epoch % eval_every_n_epochs == 0`.

**Reproducibility:** Evaluation runs inside `isolate_rng()` and sets `random`, `torch`, and `numpy` seeds to `get_rank()` so each rank has deterministic eval without affecting the training RNG after the context exits. See **`renga_flow.utils.isolate_rng`**.

**Loader reset:** Between quantiles, `PipelineDataLoader.reset()` is used so the same eval dataloader can be reused: it sets `epoch=1`, `num_batches_pulled=0`, `next_micro_batch=None`, and reinitializes the internal batch iterator. Implemented in **`renga_flow/data/loader.py`**.

## Where metrics are written

- **TensorBoard:** `SummaryWriter(log_dir=run_dir)` (main process only). Training step logging via **`renga_flow.utils.training_metrics.log_training_step`**: `train/loss`, `train/grad_norm`, `train/prodigy_d` (Prodigy), `train/automagic_avg_lr` and histogram `train/automagic_lrs` (Automagic / GenericOptim). Epoch: `train/epoch_loss`. Eval: `{name}/loss_quantile_{q}`, `{name}/loss`, `eval/eval_time_sec`. X-axis is `examples` if `x_axis_examples` else `step`.
- **WandB:** Optional; only if `config["monitoring"]["enable_wandb"]`. Same keys and step/examples as TensorBoard via `wandb.log(...)` in the loop and inside `_evaluate`. Lazy import so WandB is not required at install time.

## Saver and signal files

- **Saver** (`renga_flow.utils.saver.Saver`): `process_epoch` / `process_step` handle scheduled checkpoint/export, `max_checkpoints_to_keep` pruning after `save_checkpoint`, and **signal files** via `process_signals()`:
  - **`save` / `save_quit`** — DeepSpeed resume checkpoint; quit on `save_quit`.
  - **`export_model` / `export_model_quit`** — Export to `signal_step<N>/`; quit on `export_model_quit`.
- See **`docs/user/checkpoint-and-save.md`**, **`docs/user/signal-files.md`**, and **`docs/developer/signal-files.md`**.
- **Previews**: **`renga_flow.utils.preview.run_previews`** when `[preview]` is configured or `signals.should_preview`; see **`docs/user/previews.md`** and **`docs/developer/previews.md`**.

## Config and eval dataset loading

- Eval datasets are built in `_run_training` from `config.get("eval_datasets", [])`. Each entry is passed to **`load_eval_dataset_config`** (`renga_flow.config.loader`): string → `(name, dataset_config)` using path stem as name; dict with `name` and `config` → load TOML at `config`. Then `Dataset(eval_dataset_config, model, ...)` is created, registered with `DatasetManager` (so cache includes eval data), and after `model_engine` exists, `post_init` is called for each eval dataset. **`eval_dataloaders`** is `{ name: PipelineDataLoader(eval_data, model_engine, eval_gradient_accumulation_steps, model, 0) for name, eval_data in eval_data_map.items() }`.
