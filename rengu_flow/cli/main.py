"""``rengu`` CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rengu_flow.config.local_config import (
    apply_local_config_to_environ,
    ensure_local_config_file,
    load_local_config,
)
from rengu_flow.cli import init_cmd, platform, train_cmd, ui_cmd, update_cmd


def _warn_deprecated_invocation(argv: list[str]) -> None:
    exe = Path(sys.argv[0]).name if sys.argv else ""
    if exe in ("rengu-flow", "rengu-flow-ui"):
        print(
            f"warning: {exe} is deprecated; use the `rengu` command (see docs/user/cli.md)",
            file=sys.stderr,
        )
    if argv and argv[0] == "install":
        print(
            "warning: `rengu install` is deprecated; use `rengu init`",
            file=sys.stderr,
        )


def _looks_like_legacy_train(argv: list[str]) -> bool:
    if not argv:
        return False
    if argv[0] in ("init", "update", "train", "validate", "cache", "dump-dataset", "ui"):
        return False
    train_flags = (
        "--config",
        "--dump_dataset",
        "--validate-only",
        "--cache_only",
        "--local_rank",
        "--resume_from_checkpoint",
    )
    return any(a in argv for a in train_flags)


def _normalize_argv(argv: list[str]) -> list[str]:
    out = list(argv)
    if out and out[0] == "install":
        out = ["init", *out[1:]]
    exe = Path(sys.argv[0]).name
    if exe == "rengu-flow-ui" and out and out[0] != "ui":
        out = ["ui", *out]
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rengu", description="Rengu Flow command-line interface")
    # Handled by the early short-circuit in main() (before platform/config setup); declared here
    # only so it shows up in `rengu --help`.
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the rengu version, git commit, and installed koptim, then exit",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    init_cmd.add_parser(sub)
    update_cmd.add_parser(sub)
    train_cmd.add_parser(sub)
    ui_cmd.add_parser(sub)
    sub.add_parser("version", help="Show the rengu version, git commit, and installed koptim")
    return parser


def _print_version() -> None:
    from rengu_flow.version import version_info

    info = version_info()
    print(f"rengu-flow {info['version']}")
    if info["commit"]:
        print(f"commit:    {info['commit']}")
    print(f"koptim:    {info['koptim'] or 'not installed'}")


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    argv = _normalize_argv(raw_argv)
    _warn_deprecated_invocation(raw_argv)

    # Report version early — before platform/config setup — so it works everywhere (and is
    # available precisely when debugging an unsupported-platform or bad-config situation).
    if "--version" in raw_argv or (argv and argv[0] == "version"):
        _print_version()
        return

    platform.require_supported_platform()
    # Auto-generate rengu.local.toml for users who skipped `rengu init`. `init` does this
    # itself with its own messaging, so don't double up there.
    if not (argv and argv[0] == "init"):
        ensure_local_config_file()
    load_local_config()
    apply_local_config_to_environ()

    if _looks_like_legacy_train(argv):
        from rengu_flow.cli.training_extras import ensure_training_extras
        from rengu_flow.main import parse_args, run_prepared

        if "--config" in argv:
            idx = argv.index("--config")
            if idx + 1 < len(argv):
                ensure_training_extras(Path(argv[idx + 1]))
        run_prepared(parse_args(argv))
        return

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        raise SystemExit(2)

    if args.command == "init":
        init_cmd.run(args)
    elif args.command == "update":
        update_cmd.run(args)
    elif args.command in ("train", "validate", "cache", "dump-dataset"):
        train_cmd.run(args, args.command)
    elif args.command == "ui":
        ui_cmd.run(args)
    else:
        raise AssertionError(f"unhandled command: {args.command!r}")
