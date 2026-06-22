# Checkpoints, model export, and retention (developer guide)

User-facing option tables: `docs/user/checkpoint-and-save.md`.

## Saver responsibilities

**Module**: `rengu_flow.utils.saver.Saver`

| Method | Role |
|--------|------|
| `save_checkpoint` | DeepSpeed checkpoint; returns `False` on ENOSPC after rollback; `_prune_old_checkpoints` on success. |
| `save_model` | Export with ENOSPC wait loop (`wait_for_export_recovery`); `_prune_old_exports` on success. |
| `_save_model_once` → `_run_pipeline_export` (sync/async) → `_persist_export` | Gather shards via `prepare_export_tmp`; `_persist_export` delegates to `model.save_adapter` / `model.save_model` (atomic safetensors) based on `is_adapter`. |
| `process_epoch_boundary` / `process_step` | Per-epoch saves (named by the **completed** epoch from `EpochSchedule`) + `process_signals`. |

**Helpers**: `rengu_flow.utils.save_io` — `is_disk_full_error`, `atomic_save_safetensors`, `rollback_failed_checkpoint`, `cleanup_export_dir`, export retention parsing.

## Saving at the optimizer's true iterate (lookahead optimizers)

**Any path that persists live weights must read them with the optimizer in eval mode.**
`save_checkpoint` and `_run_pipeline_export` both wrap their reads in
`Saver._persist_at_true_iterate()` (calls `model_engine.optimizer.eval()` before, `.train()`
after; no-op for optimizers without those methods).

Why: lookahead-style optimizers from K-Optimizers (**Nekaon, MSAM, ScheduleFree, Lookahead**)
deliberately keep the *between-step* live weights displaced from the iterate they converge to —
Nekaon is `MSAM(rho=−k)`, so the live weights sit at `w + k·m` (a k-step momentum lookahead).
Only `optimizer.eval()` restores the true `w`; `train()` re-applies the displacement. A save
taken in train mode therefore stores `w + k·m`:

- **Checkpoint**: on resume `MSAM.load_state_dict` resets its "displacement present" flag, so the
  optimizer never removes that offset — training continues from the wrong point and the model
  degrades (it "forgets" what it learned). Sharpest on near-identity adapters (**BOFT**:
  `oft_blocks` start at the identity rotation, so the learned signal is small next to `k·m`).
- **Export**: the displacement is baked straight into the exported adapter.

The previews/eval paths (`utils/preview.py`, `utils/eval.py`) already bracket their reads this
way, which is why a run can *preview* fine yet *resume* broken. The adapter tensors themselves
round-trip correctly through DeepSpeed `exclude_frozen_parameters` — the bug was the weight
*values*, not the keys. New save paths added to the `Saver` must reuse `_persist_at_true_iterate()`.

## Export retention (`_prune_old_exports`)

1. Eligible dirs: `step*`, `epoch*` only (`signal_step*` exempt).
2. If `keep_exports_from_step` set: remove dirs below threshold (epoch → `epoch * steps_per_epoch`).
3. If `max_model_exports_to_keep` set: remove oldest among survivors until count ≤ N.

Both keys optional; intersection policy (see user doc).

## Disk full

- **Checkpoint**: `save_checkpoint` catches ENOSPC, rank 0 calls `rollback_failed_checkpoint`, training continues.
- **Export**: `save_model` catches ENOSPC, `cleanup_export_dir`, `write_status_file(phase="waiting_disk_export")`, `wait_for_export_recovery` until `continue` (or related signals), then retries `_save_model_once`.

`main.py` calls `saver.set_status_context(step, examples, epoch, loss)` each step for status during wait.

## Export paths and dtypes

Adapter vs full export: driven by **`bool(config.get("adapter"))`** (`main.py`), not the TOML key `save_full_model` (BACKLOG P3-1). `save_dtype` via `DTYPE_MAP` in defaults.

## Async model export (POC)

Config key: `async_model_export = true` (default off). Requires `pipeline_stages = 1`.

| Phase | Behavior |
|-------|----------|
| Gather | Rank 0 estimates snapshot size; if it fits in available RAM (see below), clones weights to CPU (`clone_state_dict_to_cpu`). Otherwise falls back to synchronous export for that save. |
| Train | Ranks resume; rank 0 queues safetensors write on a background thread (`AsyncModelExportWriter`). |
| Sync points | `save_model` / `save_checkpoint` call `_wait_async_export()` first; end of training calls `shutdown_async_exports()` in `main.py`. |

Optional TOML keys (RAM guard for async snapshot):

| Key | Default | Role |
|-----|---------|------|
| `async_model_export_ram_margin` | `0.25` | Fraction of reported available RAM held back as headroom. |
| `async_model_export_min_free_ram_gb` | unset | Extra GiB to keep free after the snapshot. |
| `async_model_export_max_snapshot_gb` | unset | Force sync export when estimated snapshot exceeds this size. |

DeepSpeed checkpoints stay synchronous. Disk-full retry loop applies to synchronous export only; async write errors surface on the next `wait_done`.

GPU smoke: `./scripts/smoke_async_export_poc.sh` — single train run (cache inline), 20 steps, checks `step10/` and `step20/` LoRA exports.

## Tests

- `tests/test_async_model_export.py` — CPU snapshot helper
- `tests/test_save_io.py`, `tests/test_export_retention.py`, `tests/test_checkpoint_rollback.py`, `tests/test_saver_export_wait.py`
- `tests/test_saver_optimizer_eval.py` — checkpoint/export read in `optimizer.eval()` mode (lookahead optimizers)
- `tests/test_signal_files.py`, `tests/test_checkpoint_prune.py`, `tests/test_saver_signals.py`
