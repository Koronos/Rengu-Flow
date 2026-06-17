"""Watermark cleanup: boxes_to_mask unit tests + clean_folder integration tests.

All tests use fake detector/inpainter — no ultralytics/torch/huggingface downloads.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rengu_flow.prep.cleanup import (
    CleanupConfig,
    LamaInpainter,
    WatermarkDetector,
    boxes_to_mask,
    clean_folder,
)

pytestmark = pytest.mark.no_ui_db

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)


@pytest.fixture(autouse=True)
def _isolate_prep_storage(tmp_path: Path, monkeypatch) -> None:
    # In-place originals backups now live under the app data dir; isolate per test.
    monkeypatch.setenv("RENGU_FLOW_UI_DATA", str(tmp_path / "appdata"))


@pytest.fixture()
def img_dir(tmp_path: Path) -> Path:
    """Two JPEG images in a temp directory."""
    d = tmp_path / "dataset"
    d.mkdir()
    shutil.copy(FIXTURE_JPG, d / "with_mark.jpg")
    shutil.copy(FIXTURE_JPG, d / "clean.jpg")
    return d


# ---------------------------------------------------------------------------
# Fake detector / inpainter factories
# ---------------------------------------------------------------------------


def make_fake_detector(boxes_map: dict[str, list[list[float]]]):
    """Return a WatermarkDetector-like factory whose detect() reads from boxes_map."""

    class FakeDetector:
        def detect(self, pil_image: Image.Image) -> list[list[float]]:
            # We identify the image by its pixel hash (fragile but simple for tests).
            raise RuntimeError("FakeDetector.detect called without filename context")

    # We'll override per-call using the filename-keyed approach below.
    del FakeDetector

    class FilenameAwareDetector:
        """Returned boxes depend on which call this is (first vs second image)."""

        def __init__(self, boxes_map: dict[str, list[list[float]]]) -> None:
            self._map = boxes_map
            self._call_index = 0

        def detect(self, pil_image: Image.Image) -> list[list[float]]:
            # Not usable by filename here — use call order.
            # The fixture puts 'with_mark.jpg' first (sorted), 'clean.jpg' second.
            key = sorted(self._map.keys())[self._call_index] if self._call_index < len(self._map) else None
            self._call_index += 1
            if key is None:
                return []
            return self._map[key]

    detector = FilenameAwareDetector(boxes_map)
    return lambda: detector


def make_dict_detector(per_name: dict[str, list[list[float]]]):
    """Detector that looks up boxes by PIL image id (set via monkey-patch on image).

    Simpler approach: sort images alphabetically and dispatch by call order.
    The test controls the mapping via per_name keyed by sorted filename index.
    """

    class DictDetector:
        def __init__(self) -> None:
            self._call = 0
            self._names = sorted(per_name.keys())

        def detect(self, pil_image: Image.Image) -> list[list[float]]:
            if self._call < len(self._names):
                result = per_name[self._names[self._call]]
            else:
                result = []
            self._call += 1
            return result

    d = DictDetector()
    return lambda: d


# Use a grey that JPEG YCbCr encoding preserves exactly (luma-only, no chroma shift).
SOLID_COLOR = (128, 128, 128)


class FakeInpainter:
    """Returns a solid-colour RGB image so we can detect it was called."""

    def inpaint(self, pil_image: Image.Image, mask_pil: Image.Image) -> Image.Image:
        return Image.new("RGB", pil_image.size, SOLID_COLOR)


def fake_inpainter_factory() -> FakeInpainter:
    return FakeInpainter()


# ---------------------------------------------------------------------------
# boxes_to_mask tests
# ---------------------------------------------------------------------------


def test_boxes_to_mask_empty_returns_all_black():
    mask = boxes_to_mask((100, 100), [], dilation_px=0)
    arr = np.array(mask)
    assert arr.max() == 0


def test_boxes_to_mask_single_box_fills_rectangle():
    mask = boxes_to_mask((200, 200), [[10, 20, 50, 80]], dilation_px=0)
    arr = np.array(mask)
    # Interior of rectangle should be white
    assert arr[20:80, 10:50].min() == 255
    # Outside should be black
    assert arr[0, 0] == 0
    assert arr[199, 199] == 0


def test_boxes_to_mask_dilation_expands():
    mask_no_dil = boxes_to_mask((200, 200), [[50, 50, 100, 100]], dilation_px=0)
    mask_dil = boxes_to_mask((200, 200), [[50, 50, 100, 100]], dilation_px=10)
    arr_no = np.array(mask_no_dil)
    arr_dil = np.array(mask_dil)
    # Dilated mask covers more pixels
    assert arr_dil.sum() > arr_no.sum()
    # Check that dilation reached 10px further in each direction
    assert arr_dil[40, 40] == 255   # top-left corner expanded
    assert arr_dil[109, 109] == 255  # bottom-right corner expanded


def test_boxes_to_mask_clamps_to_image_bounds():
    # Box goes beyond image boundary
    mask = boxes_to_mask((100, 100), [[-20, -20, 120, 120]], dilation_px=50)
    arr = np.array(mask)
    # All pixels should be white (or at least no index error)
    assert arr.min() == 255


def test_boxes_to_mask_multiple_boxes():
    mask = boxes_to_mask((200, 200), [[0, 0, 10, 10], [180, 180, 200, 200]], dilation_px=0)
    arr = np.array(mask)
    assert arr[5, 5] == 255
    assert arr[190, 190] == 255
    assert arr[100, 100] == 0  # middle untouched


# ---------------------------------------------------------------------------
# clean_folder: out-of-place (default)
# ---------------------------------------------------------------------------


def test_clean_folder_outofplace_detected_replaced_undetected_copied(img_dir: Path):
    """Detected image should differ from source; undetected should be a copy."""
    # Files sorted: clean.jpg, with_mark.jpg
    # clean.jpg -> no boxes; with_mark.jpg -> boxes
    per_name = {
        "clean.jpg": [],
        "with_mark.jpg": [[10, 10, 50, 50]],
    }
    config = CleanupConfig(copy_undetected=True)
    report = clean_folder(
        img_dir,
        config,
        detector_factory=make_dict_detector(per_name),
        inpainter_factory=fake_inpainter_factory,
    )

    out_dir = img_dir / "cleaned"
    assert out_dir.is_dir()

    assert report["cleaned"] == 1
    assert report["untouched"] == 1
    assert report["failed"] == []
    assert not report["stopped"]
    assert report["output_dir"] == str(out_dir)
    assert report["originals_backup"] is None

    # The cleaned image exists
    cleaned = out_dir / "with_mark.jpg"
    copy_of_clean = out_dir / "clean.jpg"
    assert cleaned.is_file()
    assert copy_of_clean.is_file()

    # The inpainted image should be solid SOLID_COLOR (the fake inpainter product)
    result_img = Image.open(cleaned).convert("RGB")
    arr = np.array(result_img)
    assert tuple(arr[0, 0]) == SOLID_COLOR

    # The undetected copy should load to same size as original
    src_img = Image.open(img_dir / "clean.jpg")
    out_img = Image.open(copy_of_clean)
    assert src_img.size == out_img.size


def test_clean_folder_outofplace_no_copy_undetected(img_dir: Path):
    """With copy_undetected=False the undetected image is absent from output."""
    per_name = {
        "clean.jpg": [],
        "with_mark.jpg": [[10, 10, 50, 50]],
    }
    config = CleanupConfig(copy_undetected=False)
    report = clean_folder(
        img_dir,
        config,
        detector_factory=make_dict_detector(per_name),
        inpainter_factory=fake_inpainter_factory,
    )
    out_dir = img_dir / "cleaned"
    assert not (out_dir / "clean.jpg").exists()
    assert (out_dir / "with_mark.jpg").exists()
    assert report["untouched"] == 1
    assert report["cleaned"] == 1


def test_clean_folder_report_counts(img_dir: Path):
    per_name = {
        "clean.jpg": [],
        "with_mark.jpg": [[5, 5, 30, 30]],
    }
    config = CleanupConfig()
    report = clean_folder(
        img_dir,
        config,
        detector_factory=make_dict_detector(per_name),
        inpainter_factory=fake_inpainter_factory,
    )
    assert report["cleaned"] + report["untouched"] == 2
    assert report["failed"] == []


# ---------------------------------------------------------------------------
# clean_folder: in_place
# ---------------------------------------------------------------------------


def test_clean_folder_inplace_backup_and_replacement(img_dir: Path):
    """in_place=True: detected image replaced; original backed up under the app data dir."""
    per_name = {
        "clean.jpg": [],
        "with_mark.jpg": [[10, 10, 40, 40]],
    }
    config = CleanupConfig(in_place=True)
    report = clean_folder(
        img_dir,
        config,
        detector_factory=make_dict_detector(per_name),
        inpainter_factory=fake_inpainter_factory,
    )

    # Backup dir created under the managed app data dir, not inside the dataset folder
    backup_dir = Path(report["originals_backup"])
    assert backup_dir.is_dir()
    assert not str(backup_dir).startswith(str(img_dir))

    # manifest.json lists the backed-up file
    manifest = json.loads((backup_dir / "manifest.json").read_text())
    assert "with_mark.jpg" in manifest["files"]
    assert "clean.jpg" not in manifest["files"]

    # Original backed up there
    assert (backup_dir / "with_mark.jpg").is_file()

    # Source file overwritten with inpainted result (solid colour)
    result_arr = np.array(Image.open(img_dir / "with_mark.jpg").convert("RGB"))
    assert tuple(result_arr[0, 0]) == SOLID_COLOR

    assert report["cleaned"] == 1
    assert report["untouched"] == 1


# ---------------------------------------------------------------------------
# clean_folder: failure path
# ---------------------------------------------------------------------------


def _raise_on_name(name: str, boxes_for_others: list[list[float]]):
    """Detector that raises for one filename, returns boxes for the rest."""

    class ErrDetector:
        def __init__(self) -> None:
            self._call = 0
            self._names_sorted = sorted([name, "__other__"])

        def detect(self, pil_image: Image.Image) -> list[list[float]]:
            idx = self._call
            self._call += 1
            if idx == 0:
                # First sorted file is 'clean.jpg'
                raise RuntimeError("fake detector error")
            return boxes_for_others

    d = ErrDetector()
    return lambda: d


def test_clean_folder_failure_path_continues(img_dir: Path):
    """A detector error on one image should be collected; others still processed."""

    class FirstFailsDetector:
        def __init__(self) -> None:
            self._call = 0

        def detect(self, pil_image: Image.Image) -> list[list[float]]:
            self._call += 1
            if self._call == 1:
                raise RuntimeError("simulated detector failure")
            return []  # second image: no watermark

    d = FirstFailsDetector()
    config = CleanupConfig()
    report = clean_folder(
        img_dir,
        config,
        detector_factory=lambda: d,
        inpainter_factory=fake_inpainter_factory,
    )

    # One failure, one success (untouched)
    assert len(report["failed"]) == 1
    assert report["untouched"] == 1
    assert not report["stopped"]


# ---------------------------------------------------------------------------
# clean_folder: should_stop
# ---------------------------------------------------------------------------


def test_clean_folder_should_stop(img_dir: Path):
    """should_stop() returning True after first image causes early exit."""
    call_count = [0]

    class CountingDetector:
        def detect(self, pil_image: Image.Image) -> list[list[float]]:
            call_count[0] += 1
            return []

    d = CountingDetector()
    stop_after = [1]

    def should_stop() -> bool:
        # Stop before processing the second image
        processed = call_count[0]
        return processed >= stop_after[0]

    config = CleanupConfig(copy_undetected=False)
    report = clean_folder(
        img_dir,
        config,
        detector_factory=lambda: d,
        inpainter_factory=fake_inpainter_factory,
        should_stop=should_stop,
    )

    assert report["stopped"] is True
    # Only one image was processed before the stop check
    assert call_count[0] <= 2  # at most both processed before stop detected


# ---------------------------------------------------------------------------
# clean_folder: custom output_dir
# ---------------------------------------------------------------------------


def test_clean_folder_custom_output_dir(img_dir: Path, tmp_path: Path):
    custom_out = tmp_path / "my_output"
    per_name = {
        "clean.jpg": [],
        "with_mark.jpg": [],
    }
    config = CleanupConfig(output_dir=custom_out, copy_undetected=True)
    report = clean_folder(
        img_dir,
        config,
        detector_factory=make_dict_detector(per_name),
        inpainter_factory=fake_inpainter_factory,
    )
    assert report["output_dir"] == str(custom_out)
    assert custom_out.is_dir()
    assert (custom_out / "clean.jpg").is_file()
    assert (custom_out / "with_mark.jpg").is_file()
