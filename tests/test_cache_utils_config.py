"""Tests for cache worker resolution and _map_and_cache keep_in_memory."""

from unittest.mock import MagicMock, patch

from rengu_flow.data.cache_utils import resolve_cache_num_proc


def test_resolve_cache_num_proc_default_capped():
    n = resolve_cache_num_proc(None)
    assert n >= 1
    assert n <= 8


def test_resolve_cache_num_proc_explicit():
    import sys

    if sys.platform == "win32":
        # Windows has no fork: a worker pool would spawn processes that can't share the in-process
        # GPU-encode queue handoff (deadlock), so caching is forced in-process (1) regardless.
        assert resolve_cache_num_proc(4) == 1
    else:
        assert resolve_cache_num_proc(4) == 4
    assert resolve_cache_num_proc(0) == 1


def test_ordered_parallel_map_preserves_input_order():
    # Results must come back in INPUT order even when later items finish first, and every item
    # must be processed exactly once (the cache loop relies on imap-like ordering).
    import time
    from concurrent.futures import ThreadPoolExecutor

    from rengu_flow.data.cache_utils import _ordered_parallel_map

    def fn(x):
        time.sleep((12 - x) * 0.005)  # later inputs complete sooner
        return x

    with ThreadPoolExecutor(max_workers=4) as ex:
        out = list(_ordered_parallel_map(ex, fn, range(12), max_inflight=8))
    assert out == list(range(12))


@patch("rengu_flow.data.cache_utils.ThreadPoolExecutor")
@patch("rengu_flow.data.cache_utils._ordered_parallel_map", return_value=iter([]))
def test_map_and_cache_passes_keep_in_memory_false(mock_omap, mock_executor):
    from rengu_flow.data.cache_utils import _map_and_cache

    ds = MagicMock()
    ds._fingerprint = "fp"
    ds.__len__ = MagicMock(return_value=2)

    cache = MagicMock()
    cache.fingerprint = "new_fp"
    cache.__len__ = MagicMock(return_value=0)

    subset = MagicMock()
    subset.__len__ = MagicMock(return_value=2)
    ds.select.return_value = subset

    with patch("rengu_flow.data.cache_utils.open_disk_cache", return_value=cache):
        with patch("rengu_flow.data.cache_utils.Hasher.hash", return_value="new_fp"):
            _map_and_cache(
                ds,
                lambda ex, rank: ex,
                "/tmp/cache_test",
                keep_in_memory=False,
                num_proc=2,
            )

    ds.select.assert_called_once()
    _, kwargs = ds.select.call_args
    assert kwargs.get("keep_in_memory") is False
