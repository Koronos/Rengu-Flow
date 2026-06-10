"""Distributed boundary: the single module that knows about DeepSpeed's comm backend.

The rest of the codebase imports rank/world/barrier/etc. from here — never from ``deepspeed``
directly. This is a deliberate dependency boundary (Dependency Inversion): it lets importing the
core modules (data, utils, models) stay free of DeepSpeed's ~17s eager import, which only the
actual training/engine path needs.

Two tiers of primitives:

* **Cheap, hot, DeepSpeed-free** — ``get_rank`` / ``is_main_process`` / ``get_world_size`` read the
  launcher-provided environment (``RANK`` / ``WORLD_SIZE``). They never import DeepSpeed, so the
  ubiquitous ``is_main_process()`` checks cost nothing in tests / single-process runs.
* **Collectives, training-only** — ``barrier`` / ``broadcast`` / ``send`` / ``recv`` /
  ``get_world_group`` resolve ``deepspeed.comm`` lazily and are safe no-ops when the backend is not
  up (single-process / tests), so call sites need no guards.

``is_initialized()`` is the linchpin: it answers "is the dist backend actually running?" by looking
at ``sys.modules`` — if ``deepspeed.comm`` has never been imported (the test / single-process case),
the backend cannot be initialized, so it returns ``False`` *without* importing DeepSpeed.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_RANK_ENV_VARS = ("RANK", "OMPI_COMM_WORLD_RANK")
_WORLD_ENV_VARS = ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE")


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return default
    return default


def _comm() -> Any:
    """The ``deepspeed.comm`` module, imported lazily on first real collective op."""
    import deepspeed.comm as dist

    return dist


def is_initialized() -> bool:
    """True only if the distributed backend is actually up — never imports DeepSpeed itself.

    If ``deepspeed.comm`` is not already imported, the backend cannot be initialized, so the
    answer is ``False`` and we avoid DeepSpeed's heavy import in non-distributed contexts.
    """
    comm = sys.modules.get("deepspeed.comm")
    return bool(comm and comm.is_initialized())


def get_rank() -> int:
    """Current global rank (0 if single-process). From the launcher env; no DeepSpeed import."""
    if is_initialized():
        return _comm().get_rank()
    return _env_int(_RANK_ENV_VARS, 0)


def is_main_process() -> bool:
    return get_rank() == 0


def get_world_size() -> int:
    """Total process count (1 if single-process). From the launcher env; no DeepSpeed import."""
    if is_initialized():
        return _comm().get_world_size()
    return _env_int(_WORLD_ENV_VARS, 1)


def get_world_group() -> Any:
    """The default process group (only meaningful in a live distributed run)."""
    return _comm().get_world_group()


def barrier() -> None:
    """Synchronize all ranks. No-op when the backend is not up (single-process / tests)."""
    if is_initialized():
        _comm().barrier()


def broadcast(tensor: Any, src: int = 0, **kwargs: Any) -> None:
    """Broadcast ``tensor`` from ``src`` to all ranks. No-op when not distributed."""
    if is_initialized():
        _comm().broadcast(tensor, src=src, **kwargs)


def send(tensor: Any, dst: int, **kwargs: Any) -> None:
    """Point-to-point send. No-op when not distributed."""
    if is_initialized():
        _comm().send(tensor, dst, **kwargs)


def recv(tensor: Any, src: int, **kwargs: Any) -> None:
    """Point-to-point receive. No-op when not distributed."""
    if is_initialized():
        _comm().recv(tensor, src, **kwargs)
