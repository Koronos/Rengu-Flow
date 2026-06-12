"""Watermark detection + inpainting for dataset preparation.

Detects watermarks in images using a YOLO model, then inpaints them away with
LaMa. All heavy imports (ultralytics, simple_lama_inpainting, cv2, torch,
huggingface_hub) are deferred inside methods so the module is safe to import
on CPU-only environments.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from rengu_flow.prep.caption_store import IMAGE_EXTENSIONS, PREP_DIR_NAME
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HF_REPO_ID = "fancyfeast/joycaption-watermark-detection"
_HF_FILENAME = "yolo11x-train28-best.pt"
_HF_REPO_TYPE = "space"

# LaMa inpainting as ONNX: same inference runtime as the taggers, no extra package
# (the simple-lama-inpainting wrapper pins a Pillow 9.x that doesn't build on 3.13).
_LAMA_REPO_ID = "Carve/LaMa-ONNX"
_LAMA_FILENAME = "lama_fp32.onnx"

CLEANUP_ORIGINALS_DIR = "cleanup_originals"


@dataclass
class CleanupConfig:
    """Configuration for the watermark cleanup pipeline."""

    detector_model: str = "yolo11-watermark"
    confidence: float = 0.35
    mask_dilation_px: int = 8
    in_place: bool = False
    output_dir: Path | None = None
    copy_undetected: bool = True


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def boxes_to_mask(
    size_wh: tuple[int, int],
    boxes: list[list[float]],
    dilation_px: int = 0,
) -> Image.Image:
    """Return a black L-mode PIL image with white rectangles at each box.

    Parameters
    ----------
    size_wh:
        (width, height) of the image.
    boxes:
        List of xyxy float boxes (as returned by WatermarkDetector.detect).
    dilation_px:
        Expand each rectangle by this many pixels on every side (clamped to
        image bounds).
    """
    w, h = size_wh
    mask = Image.new("L", (w, h), 0)
    if not boxes:
        return mask
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        x0, y0, x1, y1 = box[:4]
        x0 = max(0, x0 - dilation_px)
        y0 = max(0, y0 - dilation_px)
        x1 = min(w, x1 + dilation_px)
        y1 = min(h, y1 + dilation_px)
        draw.rectangle([x0, y0, x1, y1], fill=255)
    return mask


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class WatermarkDetector:
    """YOLO-based watermark detector.  ultralytics is imported lazily."""

    def __init__(self, model_path: str | Path | None = None, confidence: float = 0.5) -> None:
        self._model_path = model_path
        self.confidence = confidence
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._model_path is None:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            local = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=_HF_FILENAME,
                repo_type=_HF_REPO_TYPE,
            )
            self._model_path = local
        from ultralytics import YOLO  # noqa: PLC0415

        self._model = YOLO(str(self._model_path))

    def detect(self, pil_image: Image.Image) -> list[list[float]]:
        """Return xyxy boxes (float) whose confidence >= threshold."""
        self._load()
        arr = np.array(pil_image.convert("RGB"))
        results = self._model(arr, conf=self.confidence, verbose=False)
        boxes: list[list[float]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                if conf >= self.confidence:
                    boxes.append(list(box))
        return boxes


# ---------------------------------------------------------------------------
# Inpainter
# ---------------------------------------------------------------------------


class LamaInpainter:
    """LaMa inpainting via ONNX (Carve/LaMa-ONNX); onnxruntime is imported lazily.

    LaMa's FFC blocks need mod-8 spatial dims, so image and mask are padded
    (reflect) to the next multiple of 8 and the output is cropped back.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = model_path
        self._session: Any = None

    def _load(self) -> None:
        if self._session is not None:
            return
        if self._model_path is None:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            self._model_path = hf_hub_download(repo_id=_LAMA_REPO_ID, filename=_LAMA_FILENAME)
        import onnxruntime as ort  # noqa: PLC0415

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        else:
            logger.warning("CUDAExecutionProvider unavailable for LaMa — running on CPU.")
        providers.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(str(self._model_path), providers=providers)

    def inpaint(self, pil_image: Image.Image, mask_pil: Image.Image) -> Image.Image:
        """Inpaint *pil_image* at the white regions of *mask_pil*."""
        self._load()

        rgb = pil_image.convert("RGB")
        w, h = rgb.size
        pad_w = (8 - w % 8) % 8
        pad_h = (8 - h % 8) % 8

        img = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3,H,W) 0..1
        mask = np.asarray(mask_pil.convert("L"), dtype=np.float32)[None, :, :]
        mask = (mask > 127).astype(np.float32)  # (1,H,W) {0,1}
        if pad_w or pad_h:
            img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            mask = np.pad(mask, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        inputs = self._session.get_inputs()
        feed = {inputs[0].name: img[None], inputs[1].name: mask[None]}
        out = np.asarray(self._session.run(None, feed)[0])[0]  # (3,H',W')

        # Exports differ on output scale (0..1 vs 0..255); normalize to 0..255.
        if out.max() <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        return Image.fromarray(out[:h, :w])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _save_image(img: Image.Image, dest: Path, original_suffix: str) -> None:
    """Save *img* to *dest* respecting format and quality."""
    suffix = original_suffix.lower()
    if suffix in {".jpg", ".jpeg", ".jpe", ".jfif"}:
        img.save(dest, format="JPEG", quality=95)
    elif suffix in {".webp"}:
        img.save(dest, format="WEBP", quality=100)
    else:
        img.save(dest)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clean_folder(
    src: str | Path,
    config: CleanupConfig,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    detector_factory: Callable[[], WatermarkDetector] | None = None,
    inpainter_factory: Callable[[], LamaInpainter] | None = None,
) -> dict:
    """Run watermark cleanup over all images in *src*.

    Parameters
    ----------
    src:
        Directory containing the dataset images.
    config:
        CleanupConfig controlling behaviour.
    on_progress:
        Called with (done, total, message) after each image.
    should_stop:
        If it returns True between images, stops early and sets
        report["stopped"] = True.
    detector_factory:
        Test seam; defaults to WatermarkDetector(confidence=config.confidence).
    inpainter_factory:
        Test seam; defaults to LamaInpainter().

    Returns
    -------
    dict with keys: cleaned, untouched, failed, stopped, output_dir,
    originals_backup.
    """
    src = Path(src).resolve()

    images = sorted(
        p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    total = len(images)

    report: dict = {
        "cleaned": 0,
        "untouched": 0,
        "failed": [],
        "stopped": False,
        "output_dir": None,
        "originals_backup": None,
    }

    # --- resolve output dir -----------------------------------------------
    if not config.in_place:
        out_dir = Path(config.output_dir) if config.output_dir else src / "cleaned"
        out_dir.mkdir(parents=True, exist_ok=True)
        report["output_dir"] = str(out_dir)
    else:
        out_dir = None  # written back in-place; per-image logic handles it

    # --- in_place: prepare timestamped backup dir --------------------------
    originals_backup_dir: Path | None = None
    originals_manifest: list[str] = []
    if config.in_place:
        ts = _utc_stamp()
        originals_backup_dir = src / PREP_DIR_NAME / CLEANUP_ORIGINALS_DIR / ts
        originals_backup_dir.mkdir(parents=True, exist_ok=True)
        report["originals_backup"] = str(originals_backup_dir)

    # --- lazy-init factories -----------------------------------------------
    _detector: WatermarkDetector | None = None
    _inpainter: LamaInpainter | None = None

    def get_detector() -> WatermarkDetector:
        nonlocal _detector
        if _detector is None:
            _detector = (detector_factory or (lambda: WatermarkDetector(confidence=config.confidence)))()
        return _detector

    def get_inpainter() -> LamaInpainter:
        nonlocal _inpainter
        if _inpainter is None:
            _inpainter = (inpainter_factory or LamaInpainter)()
        return _inpainter

    # --- main loop ---------------------------------------------------------
    for done, img_path in enumerate(images):
        if should_stop is not None and should_stop():
            report["stopped"] = True
            if on_progress:
                on_progress(done, total, "stopped")
            return report

        try:
            pil = Image.open(img_path)
            pil.load()

            boxes = get_detector().detect(pil)

            if not boxes:
                # No watermark detected
                if config.in_place:
                    # Nothing to do; image stays as-is; backup not needed
                    report["untouched"] += 1
                else:
                    if config.copy_undetected:
                        dest = out_dir / img_path.name  # type: ignore[operator]
                        shutil.copy2(img_path, dest)
                    report["untouched"] += 1
            else:
                # Watermark detected -> inpaint
                mask = boxes_to_mask(pil.size, boxes, config.mask_dilation_px)
                result = get_inpainter().inpaint(pil, mask)

                if config.in_place:
                    # Back up original first
                    shutil.copy2(img_path, originals_backup_dir / img_path.name)  # type: ignore[operator]
                    originals_manifest.append(img_path.name)
                    _save_image(result, img_path, img_path.suffix)
                else:
                    dest = out_dir / img_path.name  # type: ignore[operator]
                    _save_image(result, dest, img_path.suffix)

                report["cleaned"] += 1

        except Exception as exc:
            logger.warning("cleanup: failed to process %s: %s", img_path.name, exc)
            report["failed"].append(img_path.name)

        if on_progress:
            on_progress(done + 1, total, img_path.name)

    # --- write in_place manifest -------------------------------------------
    if config.in_place and originals_backup_dir is not None:
        if originals_manifest:
            manifest = {
                "created": datetime.now(timezone.utc).isoformat(),
                "files": originals_manifest,
            }
            (originals_backup_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        else:
            # Nothing was rewritten — drop the empty backup dir instead of accumulating
            # one per dry run.
            shutil.rmtree(originals_backup_dir, ignore_errors=True)
            report["originals_backup"] = None

    return report
