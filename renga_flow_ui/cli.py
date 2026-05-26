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
    args = parser.parse_args(argv)

    if args.command == "serve":
        host = args.host or ui_host()
        port = args.port or ui_port()
        app = create_app()
        uvicorn.run(app, host=host, port=port, log_level="info")
