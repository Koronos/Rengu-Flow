"""Size-bucket selection under no_upscale / drop_undersized.

drop_undersized is a sub-option of no_upscale (ignored when no_upscale is off).
Candidates are passed already sorted by AR closeness (here all square, so order
is preserved): largest first.
"""

from rengu_flow.data.dataset import _select_size_bucket

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
