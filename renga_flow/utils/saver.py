"""Checkpoint and adapter save logic. Aligned with diffusion-pipe utils/saver."""

import os
import shutil
import sys
import time
from pathlib import Path

import torch
from deepspeed import comm as dist
from deepspeed.utils.logging import logger

from renga_flow.utils.common import is_main_process
from renga_flow.utils.signal_files import process_signals


def _convert_state_dict_dtype(state_dict, dtype):
    for key in list(state_dict.keys()):
        state_dict[key] = state_dict[key].to(device="cpu", dtype=dtype)


_last_checkpoint_time = None


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
    torch.distributed.broadcast_object_list(result, src=0)
    return result[0]


def _global_step_sort_key(path: Path) -> int:
    suffix = path.name.removeprefix("global_step")
    return int(suffix) if suffix.isdigit() else 0


def _prune_old_checkpoints(save_root: Path, max_keep: int | None) -> None:
    """Remove oldest DeepSpeed ``global_step*`` dirs when over the retention limit."""
    if max_keep is None or max_keep <= 0:
        return
    ckpt_dirs = sorted(
        (p for p in save_root.iterdir() if p.is_dir() and p.name.startswith("global_step")),
        key=_global_step_sort_key,
    )
    while len(ckpt_dirs) > max_keep:
        oldest = ckpt_dirs.pop(0)
        if is_main_process():
            print(f"Removing old checkpoint directory {oldest.name}")
        shutil.rmtree(oldest, ignore_errors=True)


class Saver:
    """Handles checkpoint and adapter/full-model saves; integrates with signal files."""

    def __init__(self, args, config, is_adapter, save_root, model, train_dataloader, model_engine, pipeline_model):
        self.args = args
        self.config = config
        self.is_adapter = is_adapter
        self.save_root = Path(save_root)
        self.model = model
        self.train_dataloader = train_dataloader
        self.model_engine = model_engine
        self.pipeline_model = pipeline_model

    def save_adapter(self, name):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        tmp_dir = save_dir / "tmp"
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()
        if dp_id == 0:
            partial_state_dict = {}
            for pname, p in self.pipeline_model.named_parameters():
                if p.requires_grad:
                    if not hasattr(p, "original_name"):
                        logger.warning(
                            "WARNING: parameter %s requires_grad but does not have original_name. Not saving it.",
                            pname,
                        )
                        continue
                    key = p.original_name.replace(".default", "").replace(".modules_to_save", "")
                    partial_state_dict[key] = p.detach()
            if "save_dtype" in self.config:
                _convert_state_dict_dtype(partial_state_dict, self.config["save_dtype"])
            torch.save(partial_state_dict, tmp_dir / f"state_dict_{stage_id}.bin")
        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in sorted(tmp_dir.glob("*.bin")):
                state_dict.update(torch.load(path, weights_only=True, map_location="cpu"))
            self.model.save_adapter(save_dir, state_dict)
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)

    def save_full_model(self, name):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        tmp_dir = save_dir / "tmp"
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()
        if dp_id == 0:
            partial_state_dict = {
                p.original_name: p.detach()
                for p in self.pipeline_model.parameters()
                if hasattr(p, "original_name")
            }
            if "save_dtype" in self.config:
                _convert_state_dict_dtype(partial_state_dict, self.config["save_dtype"])
            torch.save(partial_state_dict, tmp_dir / f"state_dict_{stage_id}.bin")
        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in sorted(tmp_dir.glob("*.bin")):
                state_dict.update(torch.load(path, map_location="cpu", weights_only=True))
            self.model.save_model(save_dir, state_dict)
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)

    def save_model(self, name):
        if is_main_process():
            print(f"Saving model to directory {name}")
        if self.is_adapter:
            self.save_adapter(name)
        else:
            self.save_full_model(name)

    def save_checkpoint(self, step, examples):
        self.model_engine.save_checkpoint(
            str(self.save_root),
            client_state={
                "step": step,
                "examples": examples,
                "custom_loader": self.train_dataloader.state_dict(),
            },
            save_latest=True,
            exclude_frozen_parameters=True,
        )
        dist.barrier()
        if is_main_process():
            _prune_old_checkpoints(self.save_root, self.config.get("max_checkpoints_to_keep"))
        dist.barrier()

    def process_epoch(self, epoch, step, examples):
        checkpointed, saved = False, False
        if self.train_dataloader.epoch != epoch:
            if _need_to_checkpoint(self.config, epoch):
                self.save_checkpoint(step, examples)
                checkpointed = True
            if "save_every_n_epochs" in self.config and epoch % self.config["save_every_n_epochs"] == 0:
                self.save_model(f"epoch{epoch}")
                saved = True
            new_epoch = self.train_dataloader.epoch
            if new_epoch > self.config["epochs"]:
                return None, checkpointed, saved
            if is_main_process():
                print(f"Started new epoch: {new_epoch}")
            return new_epoch, checkpointed, saved
        return epoch, checkpointed, saved

    def process_step(self, step, examples):
        checkpointed, saved = False, False
        signals = process_signals(self.save_root)

        if "save_every_n_steps" in self.config and step % self.config["save_every_n_steps"] == 0:
            self.save_model(f"step{step}")
            saved = True

        if signals.should_export_model:
            self.save_model(f"signal_step{step}")
            saved = True

        if _need_to_checkpoint(self.config) or signals.should_checkpoint:
            self.save_checkpoint(step, examples)
            checkpointed = True

        if signals.should_quit or signals.should_export_quit:
            if is_main_process():
                reason = "save_quit" if signals.should_checkpoint else "export_model_quit"
                print(f"Manually quitting ({reason})")
            sys.exit(0)

        return checkpointed, saved, signals
