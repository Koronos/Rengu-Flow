"""Regression: stopping the prefetch producer must not stall or leak the thread.

The val-gap probe calls loader.reset() per quantile; with the producer blocked in
put() (bounded queue full, consumer gone) the old join(timeout=30) burned 30 s per
reset and abandoned the still-blocked thread — a ~6-minute frozen log per probe and
one zombie producer per eval (py-spy verified, 2026-07-05)."""

from __future__ import annotations

import queue
import threading
import time

from rengu_flow.data.loader import PipelineDataLoader


def _bare_loader(items):
    loader = object.__new__(PipelineDataLoader)
    loader.dataloader = items
    loader.dataloader_prefetch = True
    loader.num_dataloader_workers = 0
    loader._prefetch_thread = None
    loader._prefetch_queue = None
    loader._prefetch_stop = threading.Event()
    loader._prefetch_error = []
    return loader


def test_stop_unblocks_producer_stuck_in_put():
    loader = _bare_loader(list(range(50)))
    gen = loader._iter_raw_batches()
    assert next(gen) == 0  # start the producer; it fills the size-2 queue and blocks
    deadline = time.time() + 5
    while loader._prefetch_queue.qsize() < 2 and time.time() < deadline:
        time.sleep(0.01)  # wait until the producer is parked in put()

    thread = loader._prefetch_thread
    t0 = time.time()
    loader._stop_prefetch_thread()
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"stop stalled {elapsed:.1f}s (old bug: 30s join timeout)"
    assert not thread.is_alive(), "producer thread leaked (zombie)"
    assert loader._prefetch_thread is None and loader._prefetch_queue is None


def test_normal_exhaustion_still_clean():
    loader = _bare_loader(list(range(3)))
    out = list(loader._iter_raw_batches())
    assert out == [0, 1, 2]
    loader._stop_prefetch_thread()  # idempotent after natural end
    assert loader._prefetch_thread is None
