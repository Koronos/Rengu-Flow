"""Signal files for external control of training (save, save_quit, export_model). Compatible with diffusion-pipe."""

from pathlib import Path
from typing import NamedTuple

import torch

from renga_flow.utils.common import is_main_process

# Signal file names (save/save_quit match diffusion-pipe for manager/script compatibility)
SIGNAL_SAVE = "save"
SIGNAL_SAVE_QUIT = "save_quit"
SIGNAL_EXPORT_MODEL = "export_model"
SIGNAL_EXPORT_MODEL_QUIT = "export_model_quit"
SIGNAL_PREVIEW = "preview"


class SignalResult(NamedTuple):
    """Result of checking signal files in the run directory."""

    should_checkpoint: bool
    should_quit: bool
    should_export_model: bool
    should_export_quit: bool
    should_preview: bool


def process_signals(run_dir: str | Path) -> SignalResult:
    """Check for signal files in run_dir, consume them, and return requested actions.

    Only rank 0 reads and removes files; barriers keep all ranks in sync.
    - should_checkpoint: write a DeepSpeed resume checkpoint (``save`` / ``save_quit``).
    - should_quit: exit after handling other signals (``save_quit`` / ``export_model_quit``).
    - should_export_model: export adapter or full model weights (``export_model`` / ``export_model_quit``).
    - should_preview: run image previews and log to TensorBoard (``preview``).
    """
    try:
        from deepspeed import comm as dist
    except ImportError:
        dist = None

    root = Path(run_dir)
    save_path = root / SIGNAL_SAVE
    save_quit_path = root / SIGNAL_SAVE_QUIT
    export_path = root / SIGNAL_EXPORT_MODEL
    export_quit_path = root / SIGNAL_EXPORT_MODEL_QUIT
    preview_path = root / SIGNAL_PREVIEW

    should_checkpoint = False
    should_quit = False
    should_export_model = False
    should_export_quit = False
    should_preview = False

    if is_main_process():
        if save_quit_path.exists() and save_quit_path.is_file():
            should_checkpoint = True
            should_quit = True
        elif save_path.exists() and save_path.is_file():
            should_checkpoint = True
        if export_quit_path.exists() and export_quit_path.is_file():
            should_export_model = True
            should_export_quit = True
        elif export_path.exists() and export_path.is_file():
            should_export_model = True
        if preview_path.exists() and preview_path.is_file():
            should_preview = True

    use_dist = dist is not None and dist.is_initialized()
    if use_dist:
        result = [should_checkpoint, should_quit, should_export_model, should_export_quit, should_preview]
        dist.barrier()
        torch.distributed.broadcast_object_list(result, src=0)
        should_checkpoint, should_quit, should_export_model, should_export_quit, should_preview = result
        dist.barrier()

    if is_main_process():
        for path in (save_quit_path, save_path, export_quit_path, export_path, preview_path):
            if path.exists() and path.is_file():
                path.unlink()

    if use_dist:
        dist.barrier()

    return SignalResult(
        should_checkpoint=should_checkpoint,
        should_quit=should_quit,
        should_export_model=should_export_model,
        should_export_quit=should_export_quit,
        should_preview=should_preview,
    )
