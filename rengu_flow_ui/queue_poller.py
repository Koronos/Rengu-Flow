"""Background worker that advances the training queue independently of UI activity.

Queue advancement (reconcile a finished run -> start the next pending one) happens inside
``jobs.refresh_all_jobs`` -> ``poll_job`` -> ``try_start_next``. That used to be driven ONLY by
the ``GET /api/v1/jobs`` endpoint, so a finished run's successor stalled in ``pending`` until the
next HTTP poll — i.e. it only advanced while someone was navigating the UI. This module runs the
same reconciliation on a fixed interval from a daemon thread, so the queue drains whether or not a
browser is connected.

A plain thread (not an asyncio task) is used because ``poll_job`` does blocking file/DB I/O;
keeping it off the event loop matches the threading style already used elsewhere
(``tensorboard_server``, ``job_queue._start_lock``). ``try_start_next`` is serialized by
``job_queue._start_lock`` and ``poll_job`` is idempotent, so ticking here is safe alongside the
HTTP-triggered polls that still run on UI load.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from rengu_flow_ui import gpu_lease, jobs
from rengu_flow_ui.settings import queue_poll_interval

_logger = logging.getLogger("rengu_flow_ui.queue_poller")

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()


def _step(step: Callable[[], object], label: str) -> None:
    """Run one tick step, swallowing whatever it raises. Only ``KeyboardInterrupt`` gets through.

    ``BaseException``, not ``Exception``: ``ensure_profiles`` raises ``SystemExit`` when uv is
    missing or a profile stays unimportable, and that used to kill this thread silently. This
    thread owns ``reap_dead``, so a dead thread means no lease is ever freed again.
    """
    try:
        step()
    except KeyboardInterrupt:
        raise
    except BaseException:  # noqa: BLE001 - keep the poller alive across any single bad pass
        _logger.exception("queue poller step %s failed", label)


def _tick() -> None:
    """One reconciliation pass. Never raises — a transient failure must not kill the loop.

    Each step is guarded **separately**. Sharing one block meant a failure in ``reap_dead``
    (a ``sqlite3.OperationalError`` from a lock outliving ``busy_timeout``, say) aborted the
    whole tick, taking down the reconciliation of active runs — which works today, and worked
    long before any lease code existed.

    There is deliberately no ``try_start_next()`` here: a bare tick must never start an idle
    queue (see ``jobs.refresh_all_jobs`` and ``job_queue.enqueue_job``, and
    ``docs/spec/workflows.md`` "Both lanes participate"). The queue advances on a terminal
    transition, from the explicit Start endpoint, or from ``start_job_immediately``. The
    unconditional retry belongs to Phase 1, where the workflow lane can hold the GPU while no
    job is ``running`` and a failed acquire would otherwise have no retry path at all; in
    Phase 0 the only possible holder is the running training job, and then
    ``has_active_runner()`` already blocks.
    """
    _step(gpu_lease.reap_dead, "reap_dead")
    _step(jobs.refresh_all_jobs, "refresh_all_jobs")


def _run(interval: float) -> None:
    # Event.wait returns True once _stop is set, giving a prompt, busy-loop-free shutdown.
    while not _stop.wait(interval):
        _tick()


def start_poller(interval: float | None = None) -> None:
    """Start the background queue poller (idempotent: a second call is a no-op)."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        secs = queue_poll_interval() if interval is None else interval
        _thread = threading.Thread(
            target=_run, args=(secs,), name="queue-poller", daemon=True
        )
        _thread.start()
        _logger.info("queue poller started (interval=%ss)", secs)


def stop_poller(timeout: float = 5.0) -> None:
    """Signal the poller to stop and wait briefly for the thread to exit."""
    global _thread
    with _lock:
        thread = _thread
        _thread = None
    if thread is None:
        return
    _stop.set()
    thread.join(timeout=timeout)
