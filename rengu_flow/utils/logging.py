"""Project logging: a configured, rank-aware logger decoupled from DeepSpeed.

Historically modules did ``from deepspeed.utils.logging import logger`` just to emit a line —
which dragged in DeepSpeed's ~17s eager import (and coupled plain logging to a training-only
dependency). This module provides an equivalent: a logger that is actually configured (its own
stdout handler + formatter, so ``.info()`` shows) and that only emits on rank 0.

Rank is read from the launcher-provided environment (``RANK``/``LOCAL_RANK``) so logging never
imports DeepSpeed or initializes torch.distributed. During real distributed training the launcher
sets these before any module logs; single-process runs default to rank 0 (logs enabled).
"""

from __future__ import annotations

import logging
import os
import sys

# A dedicated, isolated logger (its own tree, not the ``rengu_flow`` package-root ancestor) so it
# never intercepts or blocks the ``logging.getLogger(__name__)`` module loggers used elsewhere —
# exactly how DeepSpeed's own "DeepSpeed" logger stayed out of the way. Keeping it off the package
# hierarchy is what lets ``caplog`` still capture sibling modules' records via root propagation.
_LOGGER_NAME = "rengu_flow.console"
_RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK")


def _rank() -> int:
    for var in _RANK_ENV_VARS:
        value = os.environ.get(var)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


class _RankZeroFilter(logging.Filter):
    """Drop records on non-zero ranks so distributed runs don't duplicate every line."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _rank() == 0


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a configured, rank-0-only logger (idempotent across imports)."""
    log = logging.getLogger(name)
    if getattr(log, "_rengu_configured", False):
        return log
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [rengu_flow] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_RankZeroFilter())
    log.addHandler(handler)
    log.propagate = False
    log._rengu_configured = True  # type: ignore[attr-defined]
    return log


logger = get_logger()


class _DepPrefixFormatter(logging.Formatter):
    """Prefix root-propagated records from third-party loggers with ``[dep:<pkg>]``.

    Dependencies (lycoris, datasets, transformers, ...) log lines we must keep — muting
    would break log-file auditability — but that drown the trainer's own narrative in a
    captured log. Tagging instead of muting keeps every line in the file while making
    the source obvious at a glance (and trivially filterable in the UI).
    """

    def __init__(self, inner: logging.Formatter | None):
        super().__init__()
        self._inner = inner

    def format(self, record: logging.LogRecord) -> str:
        text = self._inner.format(record) if self._inner else super().format(record)
        name = record.name or "root"
        if name != "root" and not name.startswith("rengu_flow"):
            return f"[dep:{name.split('.')[0]}] {text}"
        return text


def tag_third_party_console_logs() -> None:
    """Wrap the root logger's handlers so third-party records get a ``[dep:...]`` prefix.

    Idempotent. Covers loggers that propagate to root (the logging-module default);
    libraries that attach their own private handlers keep their own format — those
    lines still land in the captured log, just untagged.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        root.addHandler(handler)
        if root.level == logging.NOTSET:
            root.setLevel(logging.WARNING)
    for handler in root.handlers:
        if isinstance(handler.formatter, _DepPrefixFormatter):
            continue
        handler.setFormatter(_DepPrefixFormatter(handler.formatter))
