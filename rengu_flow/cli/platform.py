"""Platform guard for the ``rengu`` CLI (Linux, WSL, and native Windows single-GPU)."""

from __future__ import annotations

import sys

_SUPPORTED = ("linux", "win32")


def require_supported_platform() -> None:
    """Allow Linux/WSL (training + multi-GPU) and native Windows (single-GPU); reject others."""
    if sys.platform not in _SUPPORTED:
        raise SystemExit(
            f"rengu supports Linux/WSL and Windows; unsupported platform {sys.platform!r}. "
            "Use Linux/WSL (recommended, multi-GPU) or native Windows (single-GPU)."
        )


def require_uv() -> None:
    from shutil import which

    if which("uv") is None:
        raise SystemExit(
            "rengu: uv is required. Install from https://docs.astral.sh/uv/ then run: ./rengu init"
        )
