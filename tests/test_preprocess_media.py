"""PreprocessMediaFile bucket rounding (no I/O)."""

from rengu_flow.data.preprocess_media import PreprocessMediaFile
from rengu_flow.utils.common import round_down_to_multiple


def test_round_down_to_multiple():
    assert round_down_to_multiple(17, 16) == 16
    assert round_down_to_multiple(16, 16) == 16


def test_preprocess_media_rounding_defaults():
    cfg = {}
    fn = PreprocessMediaFile(cfg, support_video=False)
    assert fn.round_height == 16
    assert fn.round_width == 16
    assert fn.round_frames == 4


def test_preprocess_frames_round_down_minus_one_pattern():
    """Matches SizeBucketDataset frame rounding: (n-1) rounded down to multiple + 1."""
    target = 5
    rounded = round_down_to_multiple(target - 1, 4) + 1
    assert rounded == 5
    assert round_down_to_multiple(6 - 1, 4) + 1 == 5


def test_animated_webp_decoded_as_video(tmp_path):
    """An animated WebP is treated as a native-frame video (not rejected)."""
    import imageio  # noqa: F401  (ensures the plugin is available)
    import numpy as np
    from PIL import Image

    from rengu_flow.data.preprocess_media import PreprocessMediaFile

    p = tmp_path / "clip.webp"
    frames = [
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype="uint8"))
        for i in range(6)
    ]
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=100, loop=0, format="WEBP")

    fn = PreprocessMediaFile(
        {"video_clip_mode": "single_beginning"}, support_video=True, framerate=8
    )
    # size_bucket = (width, height, frames); 5 -> frames_rounded 5 (the 4k+1 convention)
    out = fn((None, str(p)), None, size_bucket=(64, 48, 5))
    assert len(out) == 1
    video, _mask, _valid = out[0]
    assert tuple(video.shape) == (3, 5, 48, 64)  # (C, T, H, W), 5 native frames kept


def test_corrupt_image_is_tombstoned(tmp_path):
    """A corrupt still yields a zero-placeholder marked invalid instead of raising."""
    import torch
    from rengu_flow.data.preprocess_media import PreprocessMediaFile

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")  # JPEG SOI then garbage -> decode fails
    fn = PreprocessMediaFile({"video_clip_mode": "single_beginning"}, support_video=False)
    out = fn((None, str(bad)), None, size_bucket=(64, 48, 1))

    assert len(out) == 1  # exactly one row (keeps the cache 1:1), not skipped
    tensor, mask, valid = out[0]
    assert valid is False
    assert mask is None
    assert tuple(tensor.shape) == (3, 48, 64)  # bucket-shaped zero placeholder
    assert int(torch.count_nonzero(tensor)) == 0
