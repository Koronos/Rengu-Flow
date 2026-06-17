"""Environment and path settings for the UI control server."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repository root (parent of rengu_flow_ui package)."""
    return Path(__file__).resolve().parent.parent


def ui_data_dir() -> Path:
    """Local UI state: configs, job DB, staging, logs.

    Set ``RENGU_FLOW_UI_DATA`` (e.g. from ``start-ui.sh``). If unset, falls back to
    ``<repo>/data`` for ``rengu-flow-ui serve`` without the launcher script — a
    non-hidden, git-ignored folder so users can easily find the DB, logs, and staging.
    """
    base = os.environ.get("RENGU_FLOW_UI_DATA")
    if base:
        return Path(base).expanduser().resolve()
    return repo_root() / "data"


def staging_dir() -> Path:
    return ui_data_dir() / "staging"


def logs_dir() -> Path:
    return ui_data_dir() / "logs"


def toolbox_dir() -> Path:
    return ui_data_dir() / "toolbox"


def db_path() -> Path:
    return ui_data_dir() / "jobs.db"


def web_dist_dir() -> Path:
    env = os.environ.get("RENGU_FLOW_UI_DIST")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "ui" / "web" / "dist"


def ui_host() -> str:
    return os.environ.get("RENGU_FLOW_UI_HOST", "127.0.0.1")


def ui_port() -> int:
    return int(os.environ.get("RENGU_FLOW_UI_PORT", "8765"))


def ui_token() -> str | None:
    return os.environ.get("RENGU_FLOW_UI_TOKEN") or None


def queue_poll_interval() -> float:
    """Seconds between background queue-poller ticks (advances the queue when a run ends).

    Set ``RENGU_FLOW_UI_QUEUE_POLL_SECS`` to tune how soon the next queued run starts after the
    current one finishes. The poller runs independently of UI activity, so this is the worst-case
    delay before an idle browser-less queue advances.
    """
    return float(os.environ.get("RENGU_FLOW_UI_QUEUE_POLL_SECS", "3"))


def ensure_data_dirs() -> None:
    for d in (ui_data_dir(), staging_dir(), logs_dir(), toolbox_dir()):
        d.mkdir(parents=True, exist_ok=True)
