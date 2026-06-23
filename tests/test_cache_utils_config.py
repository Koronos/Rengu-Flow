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


@patch("rengu_flow.data.cache_utils.mp.Pool")
@patch("rengu_flow.data.cache_utils.mp.Manager")
def test_map_and_cache_passes_keep_in_memory_false(mock_manager, mock_pool):
    from rengu_flow.data.cache_utils import _map_and_cache

    mock_manager.return_value.Queue.return_value = MagicMock()
    mock_pool.return_value.imap.return_value = iter([])

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
