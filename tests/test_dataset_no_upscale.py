"""Size-bucket selection under no_upscale / drop_undersized.

drop_undersized is a sub-option of no_upscale (ignored when no_upscale is off).
Candidates are passed already sorted by AR closeness (here all square, so order
is preserved): largest first.
"""

import numpy as np

from rengu_flow.data.dataset import _select_size_bucket, size_bucket_ars

BUCKETS = [(1024, 1024, 1), (768, 768, 1), (512, 512, 1)]  # AR-sorted (all square)


def _pick(width, height, no_upscale, drop):
    return _select_size_bucket(BUCKETS, 1, False, width, height, no_upscale, drop)


def test_default_upscales_to_closest_bucket():
    assert _pick(900, 900, no_upscale=False, drop=False) == (1024, 1024, 1)


def test_drop_undersized_ignored_without_no_upscale():
    # no_upscale off -> drop_undersized has no effect: still the default bucket.
    assert _pick(900, 900, no_upscale=False, drop=True) == (1024, 1024, 1)


def test_no_upscale_rebuckets_down_to_largest_fitting():
    assert _pick(900, 900, no_upscale=True, drop=False) == (768, 768, 1)
    assert _pick(1200, 1200, no_upscale=True, drop=False) == (1024, 1024, 1)


def test_no_upscale_too_small_keeps_smallest_when_not_dropping():
    # 400 fits no bucket; without drop it lands in the smallest (least upscaling).
    assert _pick(400, 400, no_upscale=True, drop=False) == (512, 512, 1)


def test_no_upscale_plus_drop_discards_when_nothing_fits():
    assert _pick(400, 400, no_upscale=True, drop=True) is None
    # but an image that fits a smaller bucket is still kept
    assert _pick(900, 900, no_upscale=True, drop=True) == (768, 768, 1)


def test_size_bucket_ars_stay_parallel_to_config():
    """One AR per configured bucket, in config order — same-AR buckets must NOT collapse.

    Selection indexes size_buckets_config with positions derived from these ARs, so deduping or
    sorting them silently selects a different bucket. Adding 512x512 next to 1024x1024 used to
    collapse both to a single entry, leaving the added resolution unreachable.
    """
    config = [(1024, 1024, 1), (512, 512, 1)]
    assert list(size_bucket_ars(config)) == [1.0, 1.0]

    # Non-monotonic ARs keep config order (sorting would renumber the buckets).
    landscape_first = [(1024, 576, 1), (576, 1024, 1)]
    ars = size_bucket_ars(landscape_first)
    assert ars[0] > 1.0 and ars[1] < 1.0


def test_all_same_ar_buckets_are_reachable():
    """With 1024 and 512 both configured, each image lands in the bucket that actually fits."""
    from rengu_flow.data.dataset import DirectoryDataset

    dd = object.__new__(DirectoryDataset)
    dd.size_buckets_config = np.array([(1024, 1024, 1), (512, 512, 1)])
    dd.ars = size_bucket_ars(dd.size_buckets_config)
    dd.log_ars = np.log(dd.ars)
    dd.no_upscale = True
    dd.drop_undersized = False

    def pick(w, h):
        got = dd._find_closest_size_bucket(np.log(w / h), 1, False, width=w, height=h)
        return tuple(int(x) for x in got) if got is not None else None

    assert pick(1024, 1024) == (1024, 1024, 1)
    assert pick(512, 512) == (512, 512, 1)  # unreachable while ars was deduped


def test_renumbered_size_bucket_ars_are_rejected():
    """Deduped/sorted ARs must fail loudly, not silently select the wrong bucket.

    Reproduces the old bug: ars for [1024x1024, 512x512] collapsed to a single entry, so bucket
    selection ranked one AR and indexed a two-bucket config with it.
    """
    import numpy as np
    import pytest

    from rengu_flow.data.dataset import DirectoryDataset

    dd = object.__new__(DirectoryDataset)
    dd.size_buckets_config = np.array([(1024, 1024, 1), (512, 512, 1)])
    dd.ars = np.array([1.0])  # what dedup_and_sort used to produce
    dd.log_ars = np.log(dd.ars)
    dd.no_upscale = True
    dd.drop_undersized = False

    with pytest.raises(AssertionError, match="index-parallel"):
        dd._find_closest_size_bucket(0.0, 1, False, width=512, height=512)
