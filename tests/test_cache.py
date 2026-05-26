"""Tests for renga_flow.utils.cache.Cache."""

from pathlib import Path

import pytest
import torch

from renga_flow.utils.cache import Cache


def test_cache_add_and_read(tmp_path):
    """Add items to cache and read them back."""
    cache = Cache(tmp_path, "fp1", shard_size_gb=0.001)
    assert len(cache) == 0
    cache.add({"a": torch.tensor([1.0, 2.0])})
    cache.add({"a": torch.tensor([3.0, 4.0])})
    cache.finalize_current_shard()
    assert len(cache) == 2
    out0 = cache[0]
    out1 = cache[1]
    assert out0["a"].tolist() == [1.0, 2.0]
    assert out1["a"].tolist() == [3.0, 4.0]


def test_cache_fingerprint_mismatch_clears(tmp_path):
    """Different fingerprint clears existing cache."""
    cache = Cache(tmp_path, "fp1", shard_size_gb=0.001)
    cache.add({"x": torch.zeros(1)})
    cache.finalize_current_shard()
    assert len(cache) == 1
    cache2 = Cache(tmp_path, "fp2", shard_size_gb=0.001)
    assert len(cache2) == 0


def test_cache_default_shard_size_gb(tmp_path):
    cache = Cache(tmp_path, "fp")
    assert cache.shard_size_gb == 10.0


def test_cache_get_many_matches_single_reads(tmp_path):
    cache = Cache(tmp_path, "fp1", shard_size_gb=0.001)
    for i in range(5):
        cache.add({"v": torch.tensor([float(i)])})
    cache.finalize_current_shard()
    batch = cache.get_many([0, 2, 4, 1])
    assert [b["v"].item() for b in batch] == [0.0, 2.0, 4.0, 1.0]


def test_cache_clear(tmp_path):
    """clear() removes db and shards and re-inits."""
    cache = Cache(tmp_path, "fp1", shard_size_gb=0.001)
    cache.add({"y": torch.ones(2)})
    cache.finalize_current_shard()
    cache.clear()
    assert len(cache) == 0
    assert cache.fingerprint == "fp1"
