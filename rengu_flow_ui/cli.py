"""CLI entrypoint: rengu-flow-ui serve."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from rengu_flow_ui.app import create_app
from rengu_flow_ui.settings import ui_host, ui_port


def _ensure_schema_compatible() -> None:
    """Guard startup against an incompatible DB schema (no auto-migration yet).

    On a real version mismatch, ask the user to wipe-and-recreate (interactive) or abort
    so they can stay on the previous app version. Fresh and legacy-unstamped DBs pass.
    """
    from rengu_flow_ui.db import (
        SCHEMA_VERSION,
        reset_ui_database,
        schema_action,
        stored_schema_version,
    )

    stored = stored_schema_version()
    if schema_action(stored, SCHEMA_VERSION) == "ok":
        return

    print(
        f"The UI database schema changed (file is v{stored}, this build needs "
        f"v{SCHEMA_VERSION}). The existing library (configs, datasets, job history) is "
        "incompatible and there is no automatic migration yet."
    )
    if not sys.stdin.isatty():
        raise SystemExit(
            "Refusing to touch an incompatible database in non-interactive mode. "
            "Run `rengu-flow-ui reset-db` to wipe it, or use the previous app version."
        )
    answer = input(
        "Wipe and recreate the database now? All saved data is lost. [y/N] "
    ).strip().lower()
    if answer in ("y", "yes"):
        path = reset_ui_database()
        print(f"Recreated empty database: {path}")
    else:
        raise SystemExit("Aborted. Use the previous app version, or reset-db when ready.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rengu web control plane")
    sub = parser.add_subparsers(dest="command", required=True)
    serve_p = sub.add_parser("serve", help="Run FastAPI server")
    serve_p.add_argument("--host", default=None)
    serve_p.add_argument("--port", type=int, default=None)
    serve_p.add_argument(
        "--reload",
        action="store_true",
        help="Reload Python modules on change (development; use with Vite dev server)",
    )
    sub.add_parser(
        "reset-db",
        help="Delete jobs.db and recreate empty tables (configs, datasets, jobs)",
    )
    args = parser.parse_args(argv)

    if args.command == "reset-db":
        from rengu_flow_ui.db import reset_ui_database

        path = reset_ui_database()
        print(f"Reset UI database: {path}")
        return

    if args.command == "serve":
        _ensure_schema_compatible()
        host = args.host or ui_host()
        port = args.port or ui_port()
        if args.reload:
            from pathlib import Path

            pkg_root = Path(__file__).resolve().parent
            uvicorn.run(
                "rengu_flow_ui.app:create_app",
                factory=True,
                host=host,
                port=port,
                log_level="info",
                reload=True,
                reload_dirs=[str(pkg_root)],
            )
        else:
            app = create_app()
            uvicorn.run(app, host=host, port=port, log_level="info")
