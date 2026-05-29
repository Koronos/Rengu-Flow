"""Linux platform guard for the ``rengu`` CLI."""

from __future__ import annotations

import sys


def require_linux() -> None:
    if sys.platform != "linux":
        raise SystemExit(
            "rengu currently supports Linux only (including WSL). "
            "Use Linux or WSL for training and the web UI."
        )


def require_uv() -> None:
    from shutil import which

    if which("uv") is None:
        raise SystemExit(
            "rengu: uv is required. Install from https://docs.astral.sh/uv/ then run: ./rengu init"
        )
