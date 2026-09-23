"""Control-image helpers for edit datasets (rengu_flow/data/control.py).

The size rule and the pairing rule are the contract both the VAE and the text-encoder side rely
on: one helper, one answer, so the latent grid and the text encoder see the same control image.
"""

from __future__ import annotations

import math

import pytest
from PIL import Image

from rengu_flow.data.control import (
    ControlPairingError,
    control_signature,
    control_size,
    index_control_dir,
    load_control_image,
    pair_control_files,
)

pytestmark = pytest.mark.no_ui_db


@pytest.mark.parametrize(
    "width, height, resolution, multiple, expected",
    [
        (1024, 1024, 1024, 32, (1024, 1024)),
        (2048, 2048, 1024, 32, (1024, 1024)),  # scale to the target area, not the source size
        (512, 512, 1024, 32, (1024, 1024)),  # upscales too: the area is the target
        (1600, 900, 1024, 32, (1344, 768)),  # 1365.3 x 768.0 -> floor to 32
        (900, 1600, 1024, 32, (768, 1344)),
        (1000, 750, 512, 16, (576, 432)),  # 591.2 x 443.4 -> floor to 16
        (4000, 100, 256, 32, (1600, 32)),  # extreme AR: never below one multiple
        (100, 4000, 256, 32, (32, 1600)),
    ],
)
def test_control_size_keeps_aspect_and_floors_to_multiple(width, height, resolution, multiple, expected):
    w, h = control_size(width, height, resolution, multiple)
    assert (w, h) == expected
    assert w % multiple == 0 and h % multiple == 0
    # Floor, never round up: the area never exceeds the target.
    assert w * h <= resolution * resolution or min(w, h) == multiple


def test_control_size_is_floor_not_nearest():
    # 1600x900 at 1024: exact w = 1365.33 -> nearest multiple of 32 would be 1376; floor is 1344.
    ratio = 1600 / 900
    exact_w = math.sqrt(1024 * 1024 * ratio)
    assert round(exact_w / 32) * 32 == 1376
    assert control_size(1600, 900, 1024, 32)[0] == 1344


def test_load_control_image_resizes_with_own_aspect(tmp_path):
    p = tmp_path / "c.png"
    Image.new("RGB", (1600, 900), (10, 20, 30)).save(p)
    img = load_control_image(p, 1024, 32)
    assert img.size == (1344, 768)
    assert img.mode == "RGB"


def test_load_control_image_keeps_alpha(tmp_path):
    p = tmp_path / "c.png"
    Image.new("RGBA", (64, 64), (10, 20, 30, 128)).save(p)
    img = load_control_image(p, 64, 32)
    assert img.mode == "RGBA"
    assert img.size == (64, 64)


def test_load_control_image_grayscale_becomes_rgb(tmp_path):
    p = tmp_path / "c.png"
    Image.new("L", (64, 32), 7).save(p)
    img = load_control_image(str(p), 64, 16)
    assert img.mode == "RGB"
    assert img.size == control_size(64, 32, 64, 16)


def test_control_signature():
    assert control_signature([], 1024, 32) == ()
    assert control_signature([[1600, 900], [1024, 1024]], 1024, 32) == ((1344, 768), (1024, 1024))


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"x")


def test_pair_single_control(tmp_path):
    _touch(tmp_path, "a.png", "b.jpg", "notes.txt")
    index = index_control_dir(tmp_path)
    assert pair_control_files("a", index) == [str(tmp_path / "a.png")]
    assert pair_control_files("b", index) == [str(tmp_path / "b.jpg")]


def test_pair_numbered_controls_in_numeric_order(tmp_path):
    names = [f"a_{i}.png" for i in range(12)]
    _touch(tmp_path, *reversed(names))
    index = index_control_dir(tmp_path)
    # Numeric order (a_2 before a_10), not lexicographic.
    assert pair_control_files("a", index) == [str(tmp_path / n) for n in names]


def test_pair_missing_control_is_an_error(tmp_path):
    _touch(tmp_path, "other.png")
    with pytest.raises(ControlPairingError, match="No control"):
        pair_control_files("a", index_control_dir(tmp_path))


def test_pair_both_forms_is_an_error(tmp_path):
    _touch(tmp_path, "a.png", "a_0.png")
    with pytest.raises(ControlPairingError, match="both"):
        pair_control_files("a", index_control_dir(tmp_path))


@pytest.mark.parametrize("names", [["a_1.png"], ["a_0.png", "a_2.png"], ["a_1.png", "a_2.png"]])
def test_pair_numbered_must_be_contiguous_from_zero(tmp_path, names):
    _touch(tmp_path, *names)
    with pytest.raises(ControlPairingError, match="contiguous"):
        pair_control_files("a", index_control_dir(tmp_path))


@pytest.mark.parametrize("names", [["a.png", "a.jpg"], ["a_0.png", "a_0.jpg"], ["a_0.png", "a_00.png"]])
def test_pair_duplicate_is_an_error(tmp_path, names):
    _touch(tmp_path, *names)
    with pytest.raises(ControlPairingError, match="ambiguous"):
        pair_control_files("a", index_control_dir(tmp_path))


def test_pair_ignores_non_images_and_other_stems(tmp_path):
    _touch(tmp_path, "a_0.png", "a_1.webp", "a_2.txt", "ab_0.png", "a_x.png")
    index = index_control_dir(tmp_path)
    assert pair_control_files("a", index) == [str(tmp_path / "a_0.png"), str(tmp_path / "a_1.webp")]
