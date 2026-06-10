"""Signal files for external control of training (save, save_quit, export_model). Compatible with diffusion-pipe."""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import torch

from rengu_flow.utils.common import is_main_process

# Signal file names (save/save_quit match diffusion-pipe for manager/script compatibility)
SIGNAL_SAVE = "save"
SIGNAL_SAVE_QUIT = "save_quit"
SIGNAL_EXPORT_MODEL = "export_model"
SIGNAL_EXPORT_MODEL_QUIT = "export_model_quit"
# NOT "preview": preview PNGs and the UI's image listing use a `preview/` directory in the run
# folder. A signal file named `preview` collides with that directory — `Path.touch()` on a dir
# silently no-ops (updates mtime) and `is_file()` is False, so the signal is "received" by the UI
# but never seen by the trainer (should_preview stays False). Use a distinct, non-colliding name.
SIGNAL_PREVIEW = "preview_now"
SIGNAL_RELOAD_CONFIG = "reload_config"
SIGNAL_CONTINUE = "continue"
SIGNAL_QUIT = "quit"

# Every control-signal filename, used to sweep stale signals at startup.
ALL_SIGNAL_FILES = (
    SIGNAL_SAVE,
    SIGNAL_SAVE_QUIT,
    SIGNAL_EXPORT_MODEL,
    SIGNAL_EXPORT_MODEL_QUIT,
    SIGNAL_PREVIEW,
    SIGNAL_RELOAD_CONFIG,
    SIGNAL_CONTINUE,
    SIGNAL_QUIT,
)

_EXPORT_RECOVERY_POLL_SEC = 2.0


def _log_signal(name: str, action: str) -> None:
    """Rank-0 log line emitted when a control signal is consumed (before it is acted on)."""
    print(f"rengu_flow: signal received: {name} -> {action}", flush=True)


def clear_stale_signals(run_dir: str | Path) -> list[str]:
    """Remove any signal files left over in run_dir before the training loop starts (rank 0).

    A force-stop kills the process tree, so a ``save_quit`` it dropped is never consumed by a
    step and lingers in the folder. Without this sweep the next run reusing that folder would
    read the stale ``save_quit`` on its very first step and immediately checkpoint-and-quit.
    Run once at startup; live signals sent during training are unaffected.
    """
    if not is_main_process():
        return []
    root = Path(run_dir)
    removed: list[str] = []
    for name in ALL_SIGNAL_FILES:
        path = root / name
        if path.exists() and path.is_file():
            try:
                path.unlink()
                removed.append(name)
            except OSError:
                pass
    return removed


class SignalResult(NamedTuple):
    """Result of checking signal files in the run directory."""

    should_checkpoint: bool
    should_quit: bool
    should_export_model: bool
    should_export_quit: bool
    should_preview: bool
    should_reload_config: bool


class ExportRecoveryAction(Enum):
    """Action chosen while training is paused waiting for disk space during export."""

    CONTINUE = "continue"
    QUIT = "quit"
    CHECKPOINT_AND_QUIT = "checkpoint_and_quit"
    EXPORT_AND_QUIT = "export_and_quit"
    CHECKPOINT = "checkpoint"


def _dist_module():
    from rengu_flow import distributed as dist

    return dist


def _broadcast_object_list(values: list, src: int = 0) -> list:
    dist = _dist_module()
    if dist is not None and dist.is_initialized():
        dist.barrier()
        torch.distributed.broadcast_object_list(values, src=src)
        dist.barrier()
    return values


def _sync_ranks_after_rank0() -> None:
    """Barrier so non-main ranks wait after rank 0 touches the filesystem."""
    dist = _dist_module()
    if dist is not None and dist.is_initialized():
        dist.barrier()


def process_signals(run_dir: str | Path) -> SignalResult:
    """Check for signal files in run_dir, consume them, and return requested actions.

    Only rank 0 reads and removes files; barriers keep all ranks in sync.
    """
    root = Path(run_dir)
    save_path = root / SIGNAL_SAVE
    save_quit_path = root / SIGNAL_SAVE_QUIT
    export_path = root / SIGNAL_EXPORT_MODEL
    export_quit_path = root / SIGNAL_EXPORT_MODEL_QUIT
    preview_path = root / SIGNAL_PREVIEW
    reload_config_path = root / SIGNAL_RELOAD_CONFIG

    should_checkpoint = False
    should_quit = False
    should_export_model = False
    should_export_quit = False
    should_preview = False
    should_reload_config = False

    if is_main_process():
        # Log every signal as it's picked up (before acting on it) so a run's log shows exactly
        # what was requested and when — invaluable when debugging "why did it stop / export / etc."
        if save_quit_path.exists() and save_quit_path.is_file():
            should_checkpoint = True
            should_quit = True
            _log_signal("save_quit", "write resume checkpoint, then quit")
        elif save_path.exists() and save_path.is_file():
            should_checkpoint = True
            _log_signal("save", "write resume checkpoint")
        if export_quit_path.exists() and export_quit_path.is_file():
            should_export_model = True
            should_export_quit = True
            _log_signal("export_model_quit", "export inference weights, then quit")
        elif export_path.exists() and export_path.is_file():
            should_export_model = True
            _log_signal("export_model", "export inference weights")
        if preview_path.exists() and preview_path.is_file():
            should_preview = True
            _log_signal("preview", "render previews now")
        if reload_config_path.exists() and reload_config_path.is_file():
            should_reload_config = True
            _log_signal("reload_config", "hot-reload the [preview] section")

    result = _broadcast_object_list(
        [
            should_checkpoint,
            should_quit,
            should_export_model,
            should_export_quit,
            should_preview,
            should_reload_config,
        ]
    )
    (
        should_checkpoint,
        should_quit,
        should_export_model,
        should_export_quit,
        should_preview,
        should_reload_config,
    ) = result

    if is_main_process():
        for path in (
            save_quit_path,
            save_path,
            export_quit_path,
            export_path,
            preview_path,
            reload_config_path,
        ):
            if path.exists() and path.is_file():
                path.unlink()

    _sync_ranks_after_rank0()

    return SignalResult(
        should_checkpoint=should_checkpoint,
        should_quit=should_quit,
        should_export_model=should_export_model,
        should_export_quit=should_export_quit,
        should_preview=should_preview,
        should_reload_config=should_reload_config,
    )


def _read_export_recovery_signals(run_dir: Path) -> ExportRecoveryAction | None:
    """Rank-0 only: peek recovery signals without consuming unrelated training signals."""
    if (run_dir / SIGNAL_QUIT).is_file():
        return ExportRecoveryAction.QUIT
    if (run_dir / SIGNAL_SAVE_QUIT).is_file():
        return ExportRecoveryAction.CHECKPOINT_AND_QUIT
    if (run_dir / SIGNAL_EXPORT_MODEL_QUIT).is_file():
        return ExportRecoveryAction.EXPORT_AND_QUIT
    if (run_dir / SIGNAL_CONTINUE).is_file():
        return ExportRecoveryAction.CONTINUE
    if (run_dir / SIGNAL_SAVE).is_file():
        return ExportRecoveryAction.CHECKPOINT
    if (run_dir / SIGNAL_EXPORT_MODEL).is_file():
        return ExportRecoveryAction.CONTINUE
    return None


def _consume_export_recovery_signal(run_dir: Path, action: ExportRecoveryAction) -> None:
    if not is_main_process():
        return
    mapping = {
        ExportRecoveryAction.CONTINUE: (SIGNAL_CONTINUE, SIGNAL_EXPORT_MODEL),
        ExportRecoveryAction.QUIT: (SIGNAL_QUIT,),
        ExportRecoveryAction.CHECKPOINT_AND_QUIT: (SIGNAL_SAVE_QUIT,),
        ExportRecoveryAction.EXPORT_AND_QUIT: (SIGNAL_EXPORT_MODEL_QUIT,),
        ExportRecoveryAction.CHECKPOINT: (SIGNAL_SAVE,),
    }
    for name in mapping.get(action, ()):
        path = run_dir / name
        if path.is_file():
            path.unlink()


def wait_for_export_recovery(run_dir: str | Path) -> ExportRecoveryAction:
    """Block all ranks until rank 0 sees a recovery signal; broadcast the chosen action."""
    root = Path(run_dir)
    polls_without_signal = 0
    # Log a reminder every ~60s (30 polls * 2s) so operators know training is alive and waiting.
    _REMINDER_EVERY_POLLS = 30
    while True:
        found: ExportRecoveryAction | None = None
        if is_main_process():
            found = _read_export_recovery_signals(root)
            if found is not None:
                _log_signal(found.value, "disk-export recovery action")
                _consume_export_recovery_signal(root, found)
        payload = _broadcast_object_list([found.value if found is not None else None])
        if payload[0] is not None:
            return ExportRecoveryAction(payload[0])
        if is_main_process():
            polls_without_signal += 1
            if polls_without_signal % _REMINDER_EVERY_POLLS == 0:
                minutes = (polls_without_signal * _EXPORT_RECOVERY_POLL_SEC) / 60
                print(
                    f"[export wait] Still waiting for recovery signal ({minutes:.0f}m elapsed). "
                    f"Touch '{root / 'continue'}' to retry, or '{root / 'quit'}' to abort."
                )
            time.sleep(_EXPORT_RECOVERY_POLL_SEC)
        _sync_ranks_after_rank0()
