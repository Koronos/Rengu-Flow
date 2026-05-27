"""CLI entrypoint: renga-flow-ui serve."""

from __future__ import annotations

import argparse

import uvicorn

from renga_flow_ui.app import create_app
from renga_flow_ui.settings import ui_host, ui_port


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Renga Flow web control plane")
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
        from renga_flow_ui.db import reset_ui_database

        path = reset_ui_database()
        print(f"Reset UI database: {path}")
        return

    if args.command == "serve":
        host = args.host or ui_host()
        port = args.port or ui_port()
        if args.reload:
            from pathlib import Path

            pkg_root = Path(__file__).resolve().parent
            uvicorn.run(
                "renga_flow_ui.app:create_app",
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
