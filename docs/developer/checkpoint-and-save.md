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
