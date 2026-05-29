"""``rengu update`` — uv sync only."""

from __future__ import annotations

import argparse

from rengu_flow.cli.project_venv import reexec_cli, sync_dependencies
from rengu_flow.install_profiles import normalize_profiles


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("update", help="Re-sync project dependencies with uv sync")
    p.add_argument(
        "profiles",
        nargs="*",
        default=["base"],
        help="Profiles: base, ui, cosmos, optim, lycoris, dev, all",
    )
    p.add_argument(
        "--all-extras",
        action="store_true",
        help="Sync with all documented optional extras",
    )


def run(args: argparse.Namespace) -> None:
    if args.all_extras:
        profiles = ["all"]
    else:
        profiles = normalize_profiles(list(args.profiles))
    sync_dependencies(profiles)
    reexec_cli()
