"""PreprocessMediaFile bucket rounding (no I/O)."""

from renga_flow.data.preprocess_media import PreprocessMediaFile
from renga_flow.utils.common import round_down_to_multiple


def test_round_down_to_multiple():
    assert round_down_to_multiple(17, 16) == 16
    assert round_down_to_multiple(16, 16) == 16


def test_preprocess_media_rounding_defaults():
    cfg = {}
    fn = PreprocessMediaFile(cfg, support_video=False)
    assert fn.round_height == 16
    assert fn.round_width == 16
    assert fn.round_frames == 4
