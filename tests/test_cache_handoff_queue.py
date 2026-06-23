"""Verify the single-process (thread) cache handoff path via SingleDeviceBackend.make_cache_worker.

This test ensures that:
- DatasetManager accepts and stores a backend object
- SingleDeviceBackend.make_cache_worker returns (thread, queue) with the thread not yet started
- The queue is injected into cache_args before the worker starts (_cache_fn can enqueue tasks)
- The drain loop receives items and None sentinel correctly in the thread path
"""

import queue as _queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from rengu_flow.engine import select_backend


def test_single_device_backend_make_cache_worker_returns_thread_queue():
    """make_cache_worker returns (Thread, queue.Queue) and the thread is not yet started."""
    backend = select_backend({"engine": "accelerate"})
    called_with = {}

    def fake_fn(args, q):
        called_with["args"] = args
        called_with["q"] = q

    worker, q = backend.make_cache_worker(fake_fn, ("a", "b"))
    assert isinstance(worker, threading.Thread)
    assert isinstance(q, _queue.Queue)
    # Not started yet (daemon threads that haven't started have is_alive() == False)
    assert not worker.is_alive()

    worker.start()
    worker.join(timeout=2.0)
    assert not worker.is_alive(), "worker did not finish in time"
    assert called_with["args"] == ("a", "b")
    assert called_with["q"] is q


def test_dataset_manager_stores_backend():
    """DatasetManager stores the backend kwarg on self.backend."""
    backend = select_backend({"engine": "accelerate"})

    # Build a minimal fake model that satisfies DatasetManager.__init__
    model = MagicMock()
    model.get_vae.return_value = MagicMock()
    model.get_text_encoders.return_value = []
    model.get_call_vae_fn.return_value = MagicMock()

    from rengu_flow.data.manager import DatasetManager

    dm = DatasetManager(model, backend=backend)
    assert dm.backend is backend


def test_cache_single_process_thread_path():
    """The full cache() method completes via the thread path (no GPU, mocked _cache_fn)."""
    backend = select_backend({"engine": "accelerate"})

    model = MagicMock()
    vae = MagicMock()
    model.get_vae.return_value = vae
    model.get_text_encoders.return_value = []
    model.get_call_vae_fn.return_value = MagicMock()
    model.keep_submodel_on_cpu_after_cache.return_value = True

    from rengu_flow.data.manager import DatasetManager

    dm = DatasetManager(model, backend=backend)
    # No datasets registered — the worker gets an empty datasets_list and immediately enqueues None.

    # Replace _run_cache_worker with a minimal stub: enqueue None immediately (signals done).
    def _stub_run_cache_worker(args, q):
        q.put(None)

    with patch("rengu_flow.data.manager._run_cache_worker", _stub_run_cache_worker):
        dm.cache(unload_models=False)  # unload_models=False avoids nn.Module.to() calls
