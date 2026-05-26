# Checkpoints, model export, and retention (developer guide)

User-facing option tables: `docs/user/checkpoint-and-save.md`.

## Saver responsibilities

**Module**: `renga_flow.utils.saver.Saver`

| Method | Role |
|--------|------|
| `save_checkpoint` | `model_engine.save_checkpoint(..., save_latest=True)` + `_prune_old_checkpoints` when `max_checkpoints_to_keep` is set. |
| `save_model` / `save_adapter` / `save_full_model` | Gather pipeline parameters and delegate to `model.save_adapter` or `model.save_model`. |
| `process_epoch` | Epoch-bound checkpoint (`checkpoint_every_n_epochs`) and export (`save_every_n_epochs`). |
| `process_step` | Step-bound export, signals via `process_signals`, time-based checkpoint (`checkpoint_every_n_minutes`). |

## Export paths and dtypes

**`save_model(name)`** branches on `self.is_adapter` (set from `bool(config.get("adapter"))` in `main.py`):

- **Adapter training:** `save_adapter(name)` — gathers `pipeline_model` parameters with `requires_grad` and `original_name`, merges stage shards on rank 0, optional **`save_dtype`** cast via `_convert_state_dict_dtype`, then `model.save_adapter(save_dir, state_dict)`.
- **Full finetune:** `save_full_model(name)` — gathers all parameters with `original_name`, same dtype cast, then `model.save_model(save_dir, state_dict)`.

**`save_dtype`:** String key in main TOML (e.g. `bfloat16`). Resolved to `torch.dtype` in **`renga_flow.config.defaults.set_config_defaults`** via `DTYPE_MAP`. Applied only at export time in `saver.py`, not during training.

**Scheduled export triggers** (`process_epoch` / `process_step`):

| Config key | Handler |
|------------|---------|
| `save_every_n_epochs` | `save_model(f"epoch{epoch}")` at epoch boundary |
| `save_every_n_steps` | `save_model(f"step{step}")` when `step % N == 0` |
| `save_every_n_examples` | Converted to step interval using global batch size, then same as steps |

There is no separate `save_full_model` config flag in code: full vs adapter export is implicit from presence of `[adapter]`. User doc describes **`save_full_model`** as documentation for “export backbone weights” intent; the trainer uses `is_adapter` only.

**`save_full_model` (TOML):** Not read by `Saver` today; omit `[adapter]` for full-model export. If a future flag is added, wire it in `Saver.save_model` and `config/validation.py`.

## Checkpoint retention

```python
def _prune_old_checkpoints(save_root: Path, max_keep: int | None) -> None:
```

- Lists directories under `save_root` whose names start with `global_step`.
- Sorts by numeric step suffix.
- Deletes oldest directories until at most `max_keep` remain.
- Called on rank 0 only, bracketed by `dist.barrier()` after `save_checkpoint` (all ranks must participate in DeepSpeed save first).

Config key: `max_checkpoints_to_keep` (optional int). No default in `set_config_defaults` — omitted means no pruning.

## Export signal naming

On-demand export uses folder name `signal_step{step}` to avoid colliding with scheduled `step{N}` or `epoch{N}` exports.

## Tests

- `tests/test_signal_files.py` — `process_signals` without distributed init.
- `tests/test_checkpoint_prune.py` — `_prune_old_checkpoints` directory pruning.
