"""Build DeepSpeed / training subprocess commands from ``renga.local.toml``."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import which

from renga_flow.config.local_config import TrainingConfig, ensure_local_config_loaded


def _pick_master_port(requested: int) -> int:
    if requested > 0:
        return requested
    try:
        result = subprocess.run(
            ["ss", "-tln"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        bound = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 29500
    for port in range(29500, 29601):
        if f":{port} " not in bound:
            return port
    return 29500


def merge_training_env(
    base: dict[str, str] | None,
    training: TrainingConfig,
    *,
    respect_existing: bool = True,
) -> dict[str, str]:
    env = dict(base or os.environ)
    for key, value in training.env.items():
        if respect_existing and key in env:
            continue
        env[key] = value
    return env


def build_train_command(
    config_path: Path,
    *,
    num_gpus: int | None = None,
    master_port: int | None = None,
    resume_from: str | None = None,
    extra_args: list[str] | None = None,
    training: TrainingConfig | None = None,
) -> list[str]:
    cfg = ensure_local_config_loaded()
    t = training or cfg.training
    ngpus = num_gpus if num_gpus is not None else t.num_gpus
    port = _pick_master_port(master_port if master_port is not None else t.master_port)

    merged_extra: list[str] = []
    if t.extra_args:
        merged_extra.extend(shlex.split(t.extra_args))
    if extra_args:
        merged_extra.extend(extra_args)

    deepspeed = which("deepspeed")
    if deepspeed:
        cmd = [deepspeed, f"--num_gpus={ngpus}", f"--master_port={port}", "-m", "renga_flow.main", "--config", str(config_path)]
    else:
        cmd = [sys.executable, "-m", "renga_flow.main", "--config", str(config_path)]
    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])
    cmd.extend(merged_extra)
    return cmd


def training_subprocess_env(training: TrainingConfig | None = None) -> dict[str, str]:
    cfg = ensure_local_config_loaded()
    return merge_training_env(None, training or cfg.training)
