"""Checkpoint and adapter save logic. Aligned with diffusion-pipe utils/saver."""

import contextlib
import shutil
import sys
import time
from pathlib import Path

import torch

from rengu_flow import distributed as dist
from rengu_flow.utils.async_model_export import (
    AsyncModelExportWriter,
    ModelExportJob,
    async_snapshot_fits_from_config,
    clone_state_dict_to_cpu,
    format_byte_size,
)
from rengu_flow.utils.common import is_main_process
from rengu_flow.utils.logging import logger
from rengu_flow.utils.rng_state import capture_rng_state
from rengu_flow.utils.save_io import (
    cleanup_export_dir,
    global_step_sort_key,
    is_disk_full_error,
    list_prunable_export_dirs,
    parse_export_sort_key,
    prepare_export_tmp,
    rollback_failed_checkpoint,
    snapshot_global_step_dirs,
)
from rengu_flow.utils.signal_files import (
    ExportRecoveryAction,
    process_signals,
    wait_for_export_recovery,
)


def _convert_state_dict_dtype(state_dict, dtype):
    for key in list(state_dict.keys()):
        state_dict[key] = state_dict[key].to(device="cpu", dtype=dtype)


_last_checkpoint_time = None

# WSD pre-decay "fork" checkpoint tag. Fixed (not a global_step* dir) so retention pruning
# never removes or counts it — see Saver.save_fork_checkpoint.
FORK_CHECKPOINT_TAG = "predecay"


def _need_to_checkpoint(config, epoch=None):
    global _last_checkpoint_time
    if epoch is not None:
        if "checkpoint_every_n_epochs" in config and epoch % config["checkpoint_every_n_epochs"] == 0:
            _last_checkpoint_time = time.time()
            return True
        return False
    if "checkpoint_every_n_minutes" not in config:
        return False
    if is_main_process():
        now = time.time()
        if _last_checkpoint_time is None:
            _last_checkpoint_time = now
            result = [False]
        elif (now - _last_checkpoint_time) / 60 > config["checkpoint_every_n_minutes"]:
            _last_checkpoint_time = now
            result = [True]
        else:
            result = [False]
    else:
        result = [False]
    if dist.is_initialized():
        torch.distributed.broadcast_object_list(result, src=0)
    return result[0]


def _prune_old_checkpoints(save_root: Path, max_keep: int | None) -> None:
    """Remove oldest DeepSpeed ``global_step*`` dirs when over the retention limit."""
    if max_keep is None or max_keep <= 0:
        return
    ckpt_dirs = sorted(
        (p for p in save_root.iterdir() if p.is_dir() and p.name.startswith("global_step")),
        key=lambda p: global_step_sort_key(p.name),
    )
    while len(ckpt_dirs) > max_keep:
        oldest = ckpt_dirs.pop(0)
        _remove_tree(oldest, reason=f"Removing old checkpoint directory {oldest.name}")


def _remove_tree(path: Path, *, reason: str) -> None:
    if is_main_process():
        print(reason)
    try:
        shutil.rmtree(path)
    except OSError as exc:
        if is_main_process():
            print(f"WARNING: failed to remove {path.name}: {exc}")


def _prune_old_exports(save_root: Path, config: dict, steps_per_epoch: int) -> None:
    """Apply keep_exports_from_step then max_model_exports_to_keep (intersection policy)."""
    min_step = config.get("keep_exports_from_step")
    max_keep = config.get("max_model_exports_to_keep")
    if min_step is None and max_keep is None:
        return
    dirs = list_prunable_export_dirs(save_root)
    if not dirs:
        return
    spe = max(int(steps_per_epoch), 1)
    threshold = int(min_step) if min_step is not None else None
    survivors: list[tuple[Path, int, float]] = []
    for path in dirs:
        key = parse_export_sort_key(path.name, spe)
        if key is None:
            continue
        if threshold is not None and key < threshold:
            _remove_tree(
                path,
                reason=f"Removing export {path.name} (below keep_exports_from_step={threshold})",
            )
            continue
        survivors.append((path, key, path.stat().st_mtime))
    if max_keep is None or max_keep <= 0:
        return
    survivors.sort(key=lambda row: (row[1], row[2]))
    while len(survivors) > int(max_keep):
        oldest = survivors.pop(0)[0]
        _remove_tree(oldest, reason=f"Removing old export directory {oldest.name}")


class Saver:
    """Handles checkpoint and adapter/full-model saves; integrates with signal files."""

    def __init__(
        self,
        args,
        config,
        is_adapter,
        save_root,
        model,
        train_dataloader,
        model_engine,
        pipeline_model,
        *,
        steps_per_epoch: int = 1,
        training_ema=None,
    ):
        self.args = args
        self.config = config
        self.is_adapter = is_adapter
        self.save_root = Path(save_root)
        self.model = model
        self.train_dataloader = train_dataloader
        self.model_engine = model_engine
        self.pipeline_model = pipeline_model
        self.training_ema = training_ema
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self._last_status_step: int | None = None
        self._last_status_examples: int | None = None
        self._last_status_loss: float | None = None
        self._last_status_epoch: int | None = None
        self._async_writer: AsyncModelExportWriter | None = None
        if self._use_async_export():
            if is_main_process():
                print("[async_export] POC enabled: CPU snapshot + background disk write on rank 0")
            self._async_writer = AsyncModelExportWriter(self._write_export_job)

    def _use_async_export(self) -> bool:
        if not self.config.get("async_model_export", False):
            return False
        try:
            pp_world = self.model_engine.grid.get_pipe_parallel_world_size()
        except AttributeError:
            pp_world = 1
        if pp_world > 1:
            if is_main_process():
                logger.warning(
                    "async_model_export is disabled when pipeline_stages > 1; using synchronous export"
                )
            return False
        return True

    def _write_export_job(self, job: ModelExportJob) -> None:
        self._persist_export(job.save_dir, job.state_dict, adapter_only=job.is_adapter)
        shutil.copy(job.config_path, job.save_dir)
        _prune_old_exports(self.save_root, self.config, self.steps_per_epoch)

    def _persist_export(self, save_dir: Path, state_dict: dict, *, adapter_only: bool) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if adapter_only:
            self.model.save_adapter(save_dir, state_dict)
        else:
            self.model.save_model(save_dir, state_dict)

    def _wait_async_export(self) -> None:
        if self._async_writer is None:
            return
        if is_main_process():
            self._async_writer.wait_done()
        dist.barrier()

    def shutdown_async_exports(self) -> None:
        if self._async_writer is None:
            return
        if is_main_process():
            self._async_writer.shutdown()
        dist.barrier()

    def set_status_context(self, step: int, examples: int, epoch: int, loss: float) -> None:
        """Remember latest training metrics for the phase-change marker on disk waits."""
        self._last_status_step = step
        self._last_status_examples = examples
        self._last_status_epoch = epoch
        self._last_status_loss = loss

    def _write_training_status(self, phase: str) -> None:
        """Emit a phase-change progress marker to stdout (rank 0 only).

        Used for the disk-export-wait pause so the web UI can surface the
        "Continue export" action. This is a boundary event (not per-iteration), so
        it always emits. status.json is no longer written.
        """
        if not is_main_process() or self._last_status_step is None:
            return
        from rengu_flow.control.progress_stream import ProgressEmitter

        ProgressEmitter().emit(
            {
                "phase": phase,
                "step": self._last_status_step,
                "epoch": self._last_status_epoch or 1,
                "loss": round(float(self._last_status_loss or 0.0), 6),
            },
            force=True,
        )

    def _partial_export_state_dict(self, *, adapter_only: bool) -> dict:
        if adapter_only:
            partial: dict = {}
            skipped = 0
            for pname, p in self.pipeline_model.named_parameters():
                if not p.requires_grad:
                    continue
                if not hasattr(p, "original_name"):
                    skipped += 1
                    continue
                key = p.original_name.replace(".default", "").replace(".modules_to_save", "")
                partial[key] = p.detach()
            if skipped and is_main_process():
                logger.warning(
                    "WARNING: %d parameters require_grad but lack original_name; not saved.",
                    skipped,
                )
            return partial
        return {
            p.original_name: p.detach()
            for p in self.pipeline_model.parameters()
            if hasattr(p, "original_name")
        }

    def _run_pipeline_export_async(self, name: str, *, adapter_only: bool) -> None:
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        use_async = True
        partial = None
        save_dtype = self.config.get("save_dtype")
        if dp_id == 0:
            partial = self._partial_export_state_dict(adapter_only=adapter_only)
            use_async, needed, available = async_snapshot_fits_from_config(
                partial, save_dtype, self.config
            )
            if not use_async:
                avail_text = (
                    format_byte_size(available) if available is not None else "unknown"
                )
                print(
                    f"[async_export] {name} snapshot needs {format_byte_size(needed)} "
                    f"but only {avail_text} RAM usable; falling back to synchronous export"
                )

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        flag = torch.tensor([1 if use_async else 0], dtype=torch.int32, device=device)
        dist.broadcast(flag, src=0)
        if flag.item() == 0:
            self._run_pipeline_export_sync(name, adapter_only=adapter_only)
            return

        if dp_id == 0:
            assert partial is not None
            t0 = time.perf_counter()
            state_dict = clone_state_dict_to_cpu(partial, save_dtype)
            elapsed = time.perf_counter() - t0
            print(
                f"[async_export] {name} CPU snapshot in {elapsed:.2f}s ({len(state_dict)} tensors)"
            )
            writer = self._async_writer
            if writer is None:
                raise RuntimeError("async export writer not initialized")
            writer.submit(
                ModelExportJob(
                    name=name,
                    save_dir=self.save_root / name,
                    state_dict=state_dict,
                    is_adapter=adapter_only,
                    config_path=self.args.config,
                )
            )
            print(f"[async_export] queued {name} for background write")
        dist.barrier()

    def _run_pipeline_export_sync(self, name: str, *, adapter_only: bool) -> None:
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        if dp_id == 0 and stage_id == 0:
            prepare_export_tmp(save_dir)
        dist.barrier()
        if dp_id == 0:
            partial_state_dict = self._partial_export_state_dict(adapter_only=adapter_only)
            if "save_dtype" in self.config:
                _convert_state_dict_dtype(partial_state_dict, self.config["save_dtype"])
            torch.save(
                partial_state_dict,
                save_dir / "tmp" / f"state_dict_{stage_id}.bin",
            )
        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            tmp_dir = save_dir / "tmp"
            state_dict = {}
            for path in sorted(tmp_dir.glob("*.bin")):
                state_dict.update(torch.load(path, weights_only=True, map_location="cpu"))
            self._persist_export(save_dir, state_dict, adapter_only=adapter_only)
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)
            if self._async_writer is not None and is_main_process():
                _prune_old_exports(self.save_root, self.config, self.steps_per_epoch)

    def _run_pipeline_export(self, name: str, *, adapter_only: bool) -> None:
        # Export reads the live weights (sync, or the synchronous CPU snapshot the async
        # path takes before its background write), so it must read the optimizer's TRUE
        # iterate — same reason save_checkpoint does. Otherwise a lookahead optimizer's
        # between-step displacement is baked into the exported adapter (see
        # _persist_at_true_iterate).
        # EMA swap nests INSIDE _persist_at_true_iterate: the optimizer's eval() (true iterate)
        # must run first, then EMA weights are written on top and read by the export. Reversing
        # the order would let eval() overwrite the swapped-in EMA weights.
        with self._persist_at_true_iterate(), self._averaged_weights():
            if self._async_writer is not None:
                self._run_pipeline_export_async(name, adapter_only=adapter_only)
                return
            self._run_pipeline_export_sync(name, adapter_only=adapter_only)

    def _save_model_once(self, name: str) -> None:
        self._run_pipeline_export(name, adapter_only=self.is_adapter)

    def _finalize_successful_export(self) -> None:
        dist.barrier()
        if self._async_writer is None and is_main_process():
            _prune_old_exports(self.save_root, self.config, self.steps_per_epoch)
        dist.barrier()

    def save_model(self, name: str) -> bool:
        """Export model weights; on disk full, wait for recovery signals and retry.

        Returns True if an export was written successfully.
        """
        self._wait_async_export()
        if is_main_process():
            print(f"Saving model to directory {name}")
        save_dir = self.save_root / name
        while True:
            try:
                self._save_model_once(name)
            except OSError as exc:
                if not is_disk_full_error(exc):
                    raise
                if is_main_process():
                    print(f"Disk full while exporting to {name}: {exc}")
                    cleanup_export_dir(save_dir)
                dist.barrier()
                self._write_training_status("waiting_disk_export")
                if is_main_process():
                    print(
                        f"Training paused waiting for disk space. "
                        f"Free space, then touch {self.save_root / 'continue'} to retry export."
                    )
                action = wait_for_export_recovery(self.save_root)
                self._write_training_status("training")
                if action == ExportRecoveryAction.QUIT:
                    if is_main_process():
                        print("Quitting without save (quit signal during export wait)")
                    sys.exit(0)
                if action == ExportRecoveryAction.CHECKPOINT_AND_QUIT:
                    if is_main_process():
                        print("Checkpoint then quit (save_quit during export wait)")
                    self.save_checkpoint(
                        self._last_status_step or 1,
                        self._last_status_examples or 0,
                    )
                    sys.exit(0)
                if action == ExportRecoveryAction.EXPORT_AND_QUIT:
                    if is_main_process():
                        print("Retry export then quit (export_model_quit during export wait)")
                    try:
                        self._save_model_once(name)
                    except OSError as retry_exc:
                        if is_disk_full_error(retry_exc):
                            continue
                        raise
                    self._finalize_successful_export()
                    sys.exit(0)
                if action == ExportRecoveryAction.CHECKPOINT:
                    self.save_checkpoint(
                        self._last_status_step or 1,
                        self._last_status_examples or 0,
                    )
                    continue
                continue
            self._finalize_successful_export()
            return True

    def _trainable_params(self):
        """Trainable parameters in a stable order (same identity/order as EMA's shadow)."""
        return [p for p in self.pipeline_model.parameters() if p.requires_grad]

    def _save_ema_shadow(self) -> None:
        """Persist the EMA shadow next to the checkpoint just written. No-op if EMA off."""
        if self.training_ema is None:
            return
        from rengu_flow.training.ema import save_ema_checkpoint

        save_ema_checkpoint(self.save_root, self.training_ema, self._trainable_params())

    def _averaged_weights(self):
        """Swap EMA weights into the model for the duration of an export (no-op if EMA off)."""
        if self.training_ema is None:
            return contextlib.nullcontext()
        return self.training_ema.average_parameters(self._trainable_params())

    @contextlib.contextmanager
    def _persist_at_true_iterate(self):
        """Hold the live weights at the optimizer's TRUE iterate while persisting them.

        Lookahead-style optimizers (MSAM/Nekaon, ScheduleFree, Lookahead) deliberately
        keep the between-step weights displaced from the iterate they converge to; only
        ``optimizer.eval()`` restores the real weights (``train()`` re-applies the
        displacement). A resume checkpoint written in train mode therefore stores the
        *displaced* weights, and the optimizer cannot undo that on load — its
        "displacement present" flag resets — so the run resumes off-point and degrades
        (the symptom is sharpest on near-identity adapters like BOFT). The preview/eval
        paths already bracket their reads this way; the checkpoint save must too. No-op
        for plain optimizers without eval/train.
        """
        opt = getattr(self.model_engine, "optimizer", None)
        toggle = callable(getattr(opt, "eval", None)) and callable(getattr(opt, "train", None))
        if toggle:
            opt.eval()
        try:
            yield
        finally:
            if toggle:
                opt.train()

    def save_checkpoint(self, step, examples) -> bool:
        """Write DeepSpeed resume checkpoint; return False if disk full (training continues)."""
        self._wait_async_export()
        with self._persist_at_true_iterate():
            before = snapshot_global_step_dirs(self.save_root) if is_main_process() else set()
            dist.barrier()
            try:
                self.model_engine.save_checkpoint(
                    str(self.save_root),
                    client_state={
                        "step": step,
                        "examples": examples,
                        # Recorded so a resume can detect a changed batch/schedule (steps_per_epoch
                        # differs) and report the epoch re-mapping instead of drifting silently.
                        "steps_per_epoch": self.steps_per_epoch,
                        "custom_loader": self.train_dataloader.state_dict(),
                        "rng_state": capture_rng_state(),
                    },
                    save_latest=True,
                    exclude_frozen_parameters=True,
                )
            except OSError as exc:
                dist.barrier()
                if not is_disk_full_error(exc):
                    raise
                if is_main_process():
                    print(f"Disk full while writing checkpoint: {exc}")
                    after = snapshot_global_step_dirs(self.save_root)
                    rollback_failed_checkpoint(self.save_root, before, after)
                dist.barrier()
                return False
            dist.barrier()
            self._save_ema_shadow()
            if is_main_process():
                _prune_old_checkpoints(self.save_root, self.config.get("max_checkpoints_to_keep"))
            dist.barrier()
            return True

    def save_fork_checkpoint(self, step, examples) -> bool:
        """WSD: write the protected pre-decay "fork" resume checkpoint (tag ``predecay``).

        Saved once, at the decay onset, while the LR is still at base — the clean branch point
        to extend from. It uses a fixed tag (not a ``global_step*`` dir), so retention pruning
        never sees or counts it: ``max_checkpoints_to_keep`` keeps its rolling budget *plus*
        this fork. ``save_latest=False`` so normal resume still uses the rolling ``latest``.
        Overwrites any earlier fork so it always marks the current decay onset. Returns False on
        a full disk (training continues; the fork is best-effort).
        """
        self._wait_async_export()
        with self._persist_at_true_iterate():
            dist.barrier()
            try:
                self.model_engine.save_checkpoint(
                    str(self.save_root),
                    tag=FORK_CHECKPOINT_TAG,
                    client_state={
                        "step": step,
                        "examples": examples,
                        # Recorded so a resume can detect a changed batch/schedule (steps_per_epoch
                        # differs) and report the epoch re-mapping instead of drifting silently.
                        "steps_per_epoch": self.steps_per_epoch,
                        "custom_loader": self.train_dataloader.state_dict(),
                        "rng_state": capture_rng_state(),
                    },
                    save_latest=False,
                    exclude_frozen_parameters=True,
                )
            except OSError as exc:
                dist.barrier()
                if not is_disk_full_error(exc):
                    raise
                if is_main_process():
                    print(f"Disk full while writing the WSD fork checkpoint (skipped): {exc}")
                dist.barrier()
                return False
            dist.barrier()
            return True

    def process_epoch_boundary(self, completed_epoch: int, step: int, examples: int):
        """Per-epoch checkpoint/export for the epoch that just **completed** at ``step``.

        The training loop calls this only on an epoch boundary, with ``completed_epoch`` from
        the single :class:`~rengu_flow.training_progress.EpochSchedule` authority (step-based).
        The save is therefore named by the finished epoch — the first one is ``epoch1``, not
        ``epoch2`` — and run termination is decided by the loop's step budget, not here.

        Returns ``(checkpointed, saved)``.
        """
        checkpointed = saved = False
        if _need_to_checkpoint(self.config, completed_epoch):
            checkpointed = self.save_checkpoint(step, examples)
        if (
            "save_every_n_epochs" in self.config
            and completed_epoch % self.config["save_every_n_epochs"] == 0
        ):
            saved = self.save_model(f"epoch{completed_epoch}")
        if is_main_process():
            print(f"Completed epoch {completed_epoch}")
        return checkpointed, saved

    def process_step(self, step, examples):
        checkpointed, saved = False, False
        signals = process_signals(self.save_root)

        if "save_every_n_steps" in self.config and step % self.config["save_every_n_steps"] == 0:
            saved = self.save_model(f"step{step}")

        if signals.should_export_model:
            saved = self.save_model(f"signal_step{step}") or saved

        if _need_to_checkpoint(self.config) or signals.should_checkpoint:
            if self.save_checkpoint(step, examples):
                checkpointed = True

        if signals.should_quit or signals.should_export_quit:
            if is_main_process():
                reason = "save_quit" if signals.should_checkpoint else "export_model_quit"
                print(f"Manually quitting ({reason})")
            sys.exit(0)

        return checkpointed, saved, signals
