"""Linux platform guard for the ``renga`` CLI."""

from __future__ import annotations

import sys


def require_linux() -> None:
    if sys.platform != "linux":
        raise SystemExit(
            "renga currently supports Linux only (including WSL). "
            "Use Linux or WSL for training and the web UI."
        )


def require_uv() -> None:
    from shutil import which

    if which("uv") is None:
        raise SystemExit(
            "renga: uv is required. Install from https://docs.astral.sh/uv/ then run: ./renga init"
        )
