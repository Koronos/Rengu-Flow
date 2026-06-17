"""Fast scalar reads via TensorBoard's Rust data server — the path TensorBoard itself uses.

The Python ``EventAccumulator`` re-decodes a run's entire event stream to extract scalars; on a run
whose event file is fat with embedded preview-image bytes that is ~4.5s. TensorBoard is fast because
(with ``--load_fast=auto``, the default) it does NOT use that Python path: it spawns the compiled
``tensorboard-data-server`` binary and queries it over gRPC. The server parses in Rust, in parallel
across runs, reloads incrementally, and downsamples server-side. The same run reads in ~0.8s.

We delegate to that server. One server process per logdir (the run output dir) is spawned lazily and
kept alive — exactly how TensorBoard runs it. If the binary or its Python glue is missing (or fails),
``read_scalars`` raises :class:`DataServerUnavailable` and the caller falls back to EventAccumulator.

Everything tensorboard/grpc is imported lazily inside functions so importing this module stays cheap
and torch-free (the UI reader imports it on the request path).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

# Server-side reload cadence: the data server re-scans the logdir this often, so a live run's new
# points appear within this window (mirrors TensorBoard's default 5s reload).
_RELOAD_INTERVAL_SECS = 5

_lock = threading.Lock()
# logdir -> running SubprocessServerDataIngester (holds the gRPC data provider + the subprocess).
_ingesters: dict[str, Any] = {}


class DataServerUnavailable(RuntimeError):
    """The TensorBoard data server can't be used here; fall back to the EventAccumulator path."""


def _data_provider(logdir: str) -> Any:
    """Get (or lazily spawn) the data server for ``logdir`` and return its gRPC data provider."""
    with _lock:
        ingester = _ingesters.get(logdir)
        if ingester is not None:
            return ingester.data_provider
        try:
            from tensorboard.data import server_ingester
            from tensorboard.util import grpc_util
        except ImportError as exc:  # tensorboard too old / grpc missing
            raise DataServerUnavailable(f"tensorboard data-server glue unavailable: {exc}") from exc
        try:
            binary = server_ingester.get_server_binary()  # raises NoDataServerError if absent
        except Exception as exc:  # noqa: BLE001 - any discovery failure means "use the fallback"
            raise DataServerUnavailable(f"data-server binary not found: {exc}") from exc

        ingester = server_ingester.SubprocessServerDataIngester(
            server_binary=binary,
            logdir=logdir,
            reload_interval=_RELOAD_INTERVAL_SECS,
            channel_creds_type=grpc_util.ChannelCredsType.LOCAL,
            samples_per_plugin=None,  # server default reservoir; per-request downsample caps points
        )
        try:
            ingester.start()
        except Exception as exc:  # noqa: BLE001 - spawn/handshake failure -> fallback
            raise DataServerUnavailable(f"data server failed to start: {exc}") from exc
        # start() returns after the gRPC handshake but the run data loads asynchronously; block
        # until the initial scan settles so the very first read doesn't come back empty. Done under
        # _lock so the comparison view's concurrent first requests all wait for one warm-up, not
        # each spawn-and-race. Cached only after, so callers that queued on _lock get a ready server.
        _await_initial_load(ingester.data_provider)
        _ingesters[logdir] = ingester
        return ingester.data_provider


def _await_initial_load(
    provider: Any, *, deadline_secs: float = 8.0, min_empty_wait: float = 0.8
) -> None:
    """Wait until the server's initial scan settles, then return.

    Settled = the run-name set is stable across two polls. A non-empty stable set means the runs
    loaded; a stable *empty* set is only trusted after ``min_empty_wait`` so we don't mistake a scan
    that hasn't started for a logdir with no runs (e.g. a brand-new run with no events yet, or a
    cold empty dir in tests) — without that floor an empty logdir would block for the full deadline.
    """
    from tensorboard.context import RequestContext

    ctx = RequestContext()
    previous: set[str] | None = None
    start = time.monotonic()
    end = start + deadline_secs
    while time.monotonic() < end:
        try:
            current = {run.run_name for run in provider.list_runs(ctx, experiment_id="")}
        except Exception:  # noqa: BLE001 - let the subsequent read surface/await the error
            return
        if current == previous:
            if current:
                return  # runs found and stable
            if time.monotonic() - start >= min_empty_wait:
                return  # consistently empty past the floor → genuinely no runs
        previous = current
        time.sleep(0.2)


def _drop(logdir: str) -> None:
    """Forget (and stop) the server for ``logdir`` — called when a gRPC call shows it's dead."""
    with _lock:
        ingester = _ingesters.pop(logdir, None)
    _stop_ingester(ingester)


def read_scalars(
    logdir: str | Path,
    run_names: list[str],
    *,
    tags: list[str] | None = None,
    max_points: int = 600,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return ``{run_name: {tag: [{step, value, wall_time}, ...]}}`` via the data server.

    ``run_names`` are dir names under ``logdir`` (empty = all runs); ``tags`` filters metrics
    (None = all). Points are downsampled server-side to ``max_points`` and returned step-sorted.
    Raises :class:`DataServerUnavailable` when the server can't be used.
    """
    key = str(logdir)
    provider = _data_provider(key)

    from tensorboard.context import RequestContext
    from tensorboard.data import provider as dp

    rtf = dp.RunTagFilter(runs=list(run_names) or None, tags=list(tags) if tags else None)
    try:
        raw = provider.read_scalars(
            RequestContext(),
            experiment_id="",
            plugin_name="scalars",
            downsample=max_points,
            run_tag_filter=rtf,
        )
    except Exception as exc:  # noqa: BLE001 - server died / RPC error: drop it and fall back
        _drop(key)
        raise DataServerUnavailable(f"data server read failed: {exc}") from exc

    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run, tag_map in raw.items():
        out[run] = {
            tag: sorted(
                ({"step": d.step, "value": float(d.value), "wall_time": d.wall_time} for d in data),
                key=lambda p: p["step"],
            )
            for tag, data in tag_map.items()
        }
    return out


def _stop_ingester(ingester: Any) -> None:
    if ingester is None:
        return
    # The server is launched with --die-after-stdin; the ingester keeps its stdin open via
    # _stdin_handle, so closing that handle lets the subprocess exit cleanly.
    handle = getattr(ingester, "_stdin_handle", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


def shutdown() -> None:
    """Stop all running data servers (call on app shutdown)."""
    with _lock:
        ingesters = list(_ingesters.values())
        _ingesters.clear()
    for ingester in ingesters:
        _stop_ingester(ingester)
