"""Build DeepSpeed / training subprocess commands from ``rengu.local.toml``."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from rengu_flow.config.local_config import TrainingConfig, ensure_local_config_loaded
from rengu_flow.platform_compat import find_free_port


def _pick_master_port(requested: int) -> int:
    if requested > 0:
        return requested
    return find_free_port(start=29500, count=101)


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

    Delegates to the selected backend's ``launch_argv``. Backend is resolved from
    ``RENGU_ENGINE`` (set upstream by ``--engine``) → config ``engine`` key → OS default.
    Pass ``{}`` as config here because the loaded config is not available at this call site;
    the ``--engine`` flag already set ``RENGU_ENGINE`` in the env before this runs.
    """
    from rengu_flow.engine import select_backend

    backend = select_backend({})  # env/OS default; --engine already set RENGU_ENGINE upstream
    return backend.launch_argv({}, config_path=str(config_path), num_gpus=num_gpus, master_port=master_port)


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
