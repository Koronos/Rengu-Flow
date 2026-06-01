"""CLI entrypoint: rengu-flow-ui serve."""

from __future__ import annotations

import argparse

import uvicorn

from rengu_flow_ui.app import create_app
from rengu_flow_ui.schema_guard import ensure_schema_compatible
from rengu_flow_ui.settings import ui_host, ui_port


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rengu Flow web control plane")
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
    export_p = sub.add_parser(
        "export-library",
        help="Export config & dataset library to <dir>/configs and <dir>/datasets as TOML",
    )
    export_p.add_argument("dir", help="Destination directory")
    import_p = sub.add_parser(
        "import-library",
        help="Import config & dataset library from a directory produced by export-library",
    )
    import_p.add_argument("dir", help="Source directory")
    import_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace rows whose id already exists (default: skip existing)",
    )
    args = parser.parse_args(argv)

    if args.command == "reset-db":
        from rengu_flow_ui.db import reset_ui_database

        path = reset_ui_database()
        print(f"Reset UI database: {path}")
        return

    if args.command == "export-library":
        from rengu_flow_ui.library_migration import export_library

        counts = export_library(args.dir)
        print(f"Exported {counts['configs']} configs, {counts['datasets']} datasets to {args.dir}")
        return

    if args.command == "import-library":
        from rengu_flow_ui.library_migration import import_library

        counts = import_library(args.dir, overwrite=args.overwrite)
        print(
            f"Imported {counts['configs']} configs, {counts['datasets']} datasets "
            f"({counts['skipped']} skipped) from {args.dir}"
        )
        return

    if args.command == "serve":
        ensure_schema_compatible()
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
