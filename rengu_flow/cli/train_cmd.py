"""``rengu train`` and training shortcuts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from rengu_flow.cli.train_launcher import build_train_command, training_subprocess_env
from rengu_flow.cli.training_extras import ensure_training_extras
from rengu_flow.config.local_config import ensure_local_config_loaded


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Launch training via DeepSpeed")
    p.add_argument("--config", required=True, help="Training TOML config path")
    p.add_argument("--num-gpus", type=int, default=None)
    p.add_argument("--master-port", type=int, default=None)
    p.add_argument("--resume-from-checkpoint", nargs="?", const=True, default=None)
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed to rengu_flow.main",
    )

    v = sub.add_parser("validate", help="Validate training config and exit")
    v.add_argument("--config", required=True)

    c = sub.add_parser("cache", help="Run --cache_only then exit")
    c.add_argument("--config", required=True)
    c.add_argument("extra", nargs=argparse.REMAINDER)

    d = sub.add_parser("dump-dataset", help="Dump dataset captions/metadata")
    d.add_argument("dataset", help="Dataset TOML path")


def _main_extra_from_remainder(extra: list[str]) -> list[str]:
    if extra and extra[0] == "--":
        return extra[1:]
    return extra


def run_train(args: argparse.Namespace) -> None:
    cfg = ensure_local_config_loaded()
    extra = _main_extra_from_remainder(getattr(args, "extra", []) or [])
    config_path = Path(args.config)
    ensure_training_extras(config_path, root=cfg.root)
    cmd = build_train_command(
        config_path,
        num_gpus=args.num_gpus,
        master_port=args.master_port,
        resume_from=args.resume_from_checkpoint if args.resume_from_checkpoint is not True else None,
        extra_args=extra,
        training=cfg.training,
    )
    env = training_subprocess_env(cfg.training)
    print(f"==> {' '.join(shlex.quote(c) for c in cmd)}")
    raise SystemExit(subprocess.run(cmd, env=env, cwd=str(cfg.root)).returncode)


def run_validate(args: argparse.Namespace) -> None:
    from rengu_flow.main import parse_args, run_prepared

    cfg = ensure_local_config_loaded()
    ensure_training_extras(Path(args.config), root=cfg.root)
    argv = ["--config", args.config, "--validate-only"]
    run_prepared(parse_args(argv))


def run_cache(args: argparse.Namespace) -> None:
    cfg = ensure_local_config_loaded()
    extra = ["--cache_only", *_main_extra_from_remainder(getattr(args, "extra", []) or [])]
    config_path = Path(args.config)
    ensure_training_extras(config_path, root=cfg.root)
    cmd = build_train_command(config_path, extra_args=extra, training=cfg.training)
    env = training_subprocess_env(cfg.training)
    raise SystemExit(subprocess.run(cmd, env=env, cwd=str(cfg.root)).returncode)


def run_dump_dataset(args: argparse.Namespace) -> None:
    from rengu_flow.main import parse_args, run_prepared

    run_prepared(parse_args(["--dump_dataset", args.dataset]))


def run(args: argparse.Namespace, command: str) -> None:
    if command == "train":
        run_train(args)
    elif command == "validate":
        run_validate(args)
    elif command == "cache":
        run_cache(args)
    elif command == "dump-dataset":
        run_dump_dataset(args)
    else:
        raise SystemExit(f"unknown train command: {command}")
