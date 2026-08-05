"""--regenerate_text_cache must rebuild metadata + text embeddings but REUSE latents.

The latent (VAE) cache is the most expensive artifact to recompute, so the text-only flag must
never clear it. This drives _cache_fn with fake datasets and records the ``regenerate_cache``
each caching stage receives.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from rengu_flow.data import manager as manager_mod


class _FakeDataset:
    def __init__(self) -> None:
        self.directory_datasets = [object()]
        self.seen: dict[str, bool] = {}

    def cache_metadata(self, *, regenerate_cache, trust_cache, cache_num_proc):
        self.seen["metadata"] = regenerate_cache

    def cache_latents(self, map_fn, *, regenerate_cache, trust_cache,
                      caching_batch_size, cache_num_proc, cache_keep_in_memory):
        self.seen["latents"] = regenerate_cache

    def cache_text_embeddings(self, map_fn, idx, *, regenerate_cache,
                             caching_batch_size, cache_num_proc, cache_keep_in_memory):
        self.seen["text"] = regenerate_cache


def _run(monkeypatch, *, regenerate_cache, regenerate_text_cache):
    ds = _FakeDataset()

    fake_progress = SimpleNamespace(
        plan=lambda names: None,
        stage=lambda *a, **k: contextlib.nullcontext(),
    )
    monkeypatch.setattr(manager_mod, "is_main_process", lambda: False)
    monkeypatch.setattr(manager_mod, "_count_latent_units", lambda datasets: 0)
    monkeypatch.setattr(manager_mod, "_count_te_units", lambda datasets: 0)
    monkeypatch.setattr(
        manager_mod.caching_progress, "CachingProgress", lambda **kw: fake_progress
    )
    monkeypatch.setattr(manager_mod.caching_progress, "set_active", lambda p: None)

    queue = SimpleNamespace(put=lambda item: None)
    manager_mod._cache_fn(
        [ds],
        queue,
        lambda *a, **k: [],  # preprocess_media_file_fn (never invoked: fakes ignore map_fn)
        1,  # num_text_encoders
        regenerate_cache,
        regenerate_text_cache,
        False,  # trust_cache
        1,  # caching_batch_size
        None,  # cache_num_proc
        False,  # cache_keep_in_memory
    )
    return ds.seen


def test_text_flag_preserves_latents(monkeypatch) -> None:
    seen = _run(monkeypatch, regenerate_cache=False, regenerate_text_cache=True)
    assert seen["latents"] is False  # the expensive cache is reused
    assert seen["metadata"] is True
    assert seen["text"] is True


def test_full_flag_regenerates_everything(monkeypatch) -> None:
    seen = _run(monkeypatch, regenerate_cache=True, regenerate_text_cache=False)
    assert seen == {"metadata": True, "latents": True, "text": True}


def test_no_flag_regenerates_nothing(monkeypatch) -> None:
    seen = _run(monkeypatch, regenerate_cache=False, regenerate_text_cache=False)
    assert seen == {"metadata": False, "latents": False, "text": False}
