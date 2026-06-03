"""``rengu init`` — local config + uv sync."""

from __future__ import annotations

import argparse

from rengu_flow.config.local_config import (
    ensure_ui_data_dir,
    init_local_config_file,
    load_local_config,
    local_config_path,
    repo_root,
)
from rengu_flow.cli.project_venv import reexec_cli, sync_dependencies
from rengu_flow.install.profiles import normalize_profiles
from rengu_flow.install.state import record_installed_profiles


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "init",
        help="Create rengu.local.toml, UI data dir, and install deps into .venv",
    )
    p.add_argument(
        "profiles",
        nargs="*",
        default=["base"],
        help="Profiles: base, ui, cosmos, optim, lycoris, dev, all",
    )
    p.add_argument(
        "--only-config",
        action="store_true",
        help="Only copy rengu.local.toml and create dirs; skip dependency install",
    )


def run(args: argparse.Namespace) -> None:
    root = repo_root()
    created = not local_config_path(root).is_file()
    path = init_local_config_file(root=root)
    if created:
        print(f"==> Created {path}")
    else:
        print(f"==> Using existing {path}")
    load_local_config(root=root)
    data_dir = ensure_ui_data_dir()
    print(f"==> UI data dir: {data_dir}")
    if args.only_config:
        return
    profiles = normalize_profiles(list(args.profiles))
    sync_dependencies(profiles)
    record_installed_profiles([p for p in profiles if p != "base"])
    reexec_cli()
