# Signal files (developer guide)

This page describes the **technical contract** and **implementation** of the file-based signal system so you can extend it or integrate it into another loop.

## Contract

- **Location**: Signal files live in the **root** of the run directory (`run_dir`). There is no `signals/` subdirectory; `save` / `save_quit` match diffusion-pipe for manager/script compatibility.
- **Names**: `save`, `save_quit`, `export_model`, `export_model_quit`, `preview` (see constants below).
- **Consumption**: Checked once per training step via `Saver.process_step` → `process_signals()`. Only rank 0 reads and removes files; result is broadcast to all ranks.
- **Barriers**: Barriers keep ranks aligned before/after file removal.

## Where the code lives

- **Module**: `renga_flow.utils.signal_files`
- **Constants**: `SIGNAL_SAVE`, `SIGNAL_SAVE_QUIT`, `SIGNAL_EXPORT_MODEL`, `SIGNAL_EXPORT_MODEL_QUIT`, `SIGNAL_PREVIEW`
- **Return type**: `SignalResult(should_checkpoint, should_quit, should_export_model, should_export_quit, should_preview)`
- **Function**: `process_signals(run_dir: str | Path) -> SignalResult`
- **Call site**: `renga_flow.utils.saver.Saver.process_step` — reacts to export signals with `save_model(f"signal_step{step}")`, checkpoint signals with `save_checkpoint`, and `sys.exit(0)` when `should_quit` or `should_export_quit`.

Checkpoint retention (`max_checkpoints_to_keep`) is implemented in `saver._prune_old_checkpoints` after each `save_checkpoint`. See `docs/user/checkpoint-and-save.md`.

## API: `process_signals`

```python
class SignalResult(NamedTuple):
    should_checkpoint: bool
    should_quit: bool
    should_export_model: bool
    should_export_quit: bool

def process_signals(run_dir: str | Path) -> SignalResult:
    ...
```

- **Input**: `run_dir` — directory that holds DeepSpeed checkpoints and signal files.
- **Output**: Four booleans; multiple signals in one step can all be true if several files were touched before the step (unusual).
- **Distributed**: Uses `deepspeed.comm.dist.barrier()` and `torch.distributed.broadcast_object_list` when DeepSpeed is initialized.

## Adding a new signal

1. **Constant** in `renga_flow.utils.signal_files`, e.g. `SIGNAL_PAUSE = "pause"`.
2. **Detection** in `process_signals` on rank 0; extend `SignalResult` with a new field (preferred over ad-hoc return shapes).
3. **Broadcast** the extended result list to all ranks.
4. **Consume** the file on rank 0 after broadcast; `dist.barrier()` again.
5. **React** in `Saver.process_step` or the training loop.

Keeping `save` / `save_quit` names and the run_dir root preserves diffusion-pipe manager compatibility.

## Compatibility with diffusion-pipe

- Same resume signal names: `save`, `save_quit`.
- Same location: root of the run directory.
- **New in renga-flow**: `export_model`, `export_model_quit` (diffusion-pipe managers do not send these unless extended).
