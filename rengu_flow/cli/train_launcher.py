"""Build DeepSpeed / training subprocess commands from ``rengu.local.toml``."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import which

from rengu_flow.config.local_config import TrainingConfig, ensure_local_config_loaded


def _pick_master_port(requested: int) -> int:
    if requested > 0:
        return requested
    # Cross-platform free-port probe (replaces Linux-only `ss` parsing).
    from rengu_flow.platform_compat import find_free_port

    return find_free_port(29500, 101)


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


def base_train_command(
    config_path: Path,
    *,
    num_gpus: int,
    master_port: int | None = None,
) -> list[str]:
    """Base argv to launch the trainer, shared by the CLI and the web UI.

    The DeepSpeed launcher is used only for engine='deepspeed' (multi-GPU pipeline). For
    engine='accelerate' (single-GPU, default on Windows) we run the module directly — no
    DeepSpeed launcher, no DeepSpeed needed. DeepSpeed's launcher needs ``--module`` (not
    ``-m``) for a module target; we fall back to ``python -m`` when deepspeed is absent too.
    """
    from rengu_flow.engine import resolve_backend

    deepspeed = which("deepspeed") if resolve_backend() == "deepspeed" else None
    if deepspeed:
        cmd = [deepspeed, f"--num_gpus={num_gpus}"]
        if master_port is not None:
            cmd.append(f"--master_port={master_port}")
        cmd += ["--module", "rengu_flow.main", "--config", str(config_path)]
    else:
        cmd = [sys.executable, "-m", "rengu_flow.main", "--config", str(config_path)]
    return cmd


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

    cmd = base_train_command(config_path, num_gpus=ngpus, master_port=port)
    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])
    cmd.extend(merged_extra)
    return cmd


def training_subprocess_env(training: TrainingConfig | None = None) -> dict[str, str]:
    cfg = ensure_local_config_loaded()
    t = training or cfg.training
    env = merge_training_env(None, t)
    # Select the engine backend for the child (rengu_flow.engine.resolve_backend reads this).
    # Empty -> auto (per-OS default). An explicit [training].engine wins; never override one
    # the user already exported.
    if t.engine:
        env.setdefault("RENGU_ENGINE", t.engine)
    # Unbuffered child stdout so @@RFPROG@@ progress markers (and logs) flush per line
    # instead of in block-buffered bursts — the CLI bar and UI log tail both depend on
    # markers arriving promptly. respect_existing semantics: don't override an explicit set.
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env
