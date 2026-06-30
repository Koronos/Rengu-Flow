"""no_upscale discard rule: an image too small for its size bucket would upscale."""

from rengu_flow.data.dataset import _too_small_for_bucket


def test_image_at_or_above_bucket_is_kept():
    bucket = (1024, 1024, 1)  # (width, height, frames)
    assert _too_small_for_bucket(1024, 1024, bucket) is False  # exact fit
    assert _too_small_for_bucket(1200, 1100, bucket) is False  # larger -> downscale


def test_image_below_bucket_in_either_dimension_is_dropped():
    assert _too_small_for_bucket(1024, 768, (1024, 1024, 1)) is True   # height short
    assert _too_small_for_bucket(800, 1300, (832, 1216, 1)) is True    # width short
