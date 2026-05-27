"""Scan dataset directories for UI previews (bounded, fast on huge folders)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from renga_flow.data.dataset import CAPTIONS_JSON_FILE, VIDEO_EXTENSIONS

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

# Stop counting / listing after this many media files per folder (UI only).
UI_SCAN_COUNT_CAP = 10_000
UI_LIST_COUNT_CAP = 10_000
DEFAULT_MAX_SAMPLES = 12

_VIDEO_EXT_LOWER = {v.lower() for v in VIDEO_EXTENSIONS}


def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _is_video_name(name: str) -> bool:
    return Path(name).suffix.lower() in _VIDEO_EXT_LOWER


def list_image_files_page(
    root: Path,
    *,
    offset: int = 0,
    limit: int = 24,
    count_cap: int = UI_LIST_COUNT_CAP,
) -> dict[str, Any]:
    """
    List one page of image filenames using ``os.scandir`` (no full sort, no full scan).

    Stops after ``offset + limit`` images are collected for the page, or after
    ``count_cap`` image files were seen (whichever comes first).
    """
    offset = max(0, int(offset))
    limit = max(1, int(limit))
    count_cap = max(limit + offset, int(count_cap))

    if not root.is_dir():
        return {"names": [], "total": 0, "count_capped": False}

    names: list[str] = []
    image_index = 0
    count_capped = False

    try:
        with os.scandir(root) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name = entry.name
                if not _is_image_name(name):
                    continue

                if image_index < offset:
                    image_index += 1
                    if image_index >= count_cap:
                        count_capped = True
                        break
                    continue

                if len(names) < limit:
                    names.append(name)
                image_index += 1
                if image_index >= count_cap:
                    count_capped = True
                    break
    except OSError:
        return {"names": [], "total": 0, "count_capped": False}

    return {
        "names": names,
        "total": image_index,
        "count_capped": count_capped,
    }


def scan_folder(
    path: str | Path,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    count_cap: int = UI_SCAN_COUNT_CAP,
) -> dict[str, Any]:
    """
    Summarize one directory for the UI without reading every file on disk.

    Uses a single ``os.scandir`` pass (unordered). Counts and samples stop at
    ``count_cap`` media files; sets ``count_capped`` when the folder may be larger.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        return {
            "ok": False,
            "path": str(root),
            "error": "Path is not a directory or does not exist.",
        }

    count_cap = max(1, int(count_cap))
    max_samples = max(1, int(max_samples))

    image_count = 0
    video_count = 0
    caption_txt = 0
    samples: list[str] = []
    sample_stems: set[str] = set()
    count_capped = False
    media_seen = 0

    try:
        with os.scandir(root) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name = entry.name
                suffix = Path(name).suffix.lower()
                stem = Path(name).stem

                if suffix in IMAGE_EXTENSIONS:
                    image_count += 1
                    media_seen += 1
                    if len(samples) < max_samples:
                        samples.append(name)
                        sample_stems.add(stem)
                elif suffix in _VIDEO_EXT_LOWER:
                    video_count += 1
                    media_seen += 1
                    if len(samples) < max_samples:
                        samples.append(name)
                        sample_stems.add(stem)
                elif suffix == ".txt" and stem not in sample_stems:
                    caption_txt += 1

                if media_seen >= count_cap:
                    count_capped = True
                    break
    except OSError as e:
        return {"ok": False, "path": str(root.resolve()), "error": str(e)}

    has_captions_json = (root / CAPTIONS_JSON_FILE).is_file()

    out: dict[str, Any] = {
        "ok": True,
        "path": str(root.resolve()),
        "image_count": image_count,
        "video_count": video_count,
        "caption_txt_files": caption_txt,
        "has_captions_json": has_captions_json,
        "sample_files": samples,
        "total_media": image_count + video_count,
        "count_capped": count_capped,
    }
    if count_capped:
        out["image_count_display"] = f"{count_cap}+"
    return out


def preview_dataset_config(dataset_config: dict) -> dict:
    """Scan all [[directory]] entries (each scan is capped)."""
    directories = dataset_config.get("directory") or []
    if not isinstance(directories, list):
        directories = []
    per_dir = []
    total_images = 0
    total_videos = 0
    any_capped = False
    for i, entry in enumerate(directories):
        if not isinstance(entry, dict):
            per_dir.append({"index": i, "ok": False, "error": "Invalid directory entry"})
            continue
        path = entry.get("path", "")
        scan = scan_folder(path) if path else {"ok": False, "error": "Missing path"}
        scan["index"] = i
        scan["num_repeats"] = entry.get("num_repeats")
        per_dir.append(scan)
        if scan.get("ok"):
            total_images += scan.get("image_count", 0)
            total_videos += scan.get("video_count", 0)
            if scan.get("count_capped"):
                any_capped = True
    return {
        "directories": per_dir,
        "total_images": total_images,
        "total_videos": total_videos,
        "directory_count": len(directories),
        "total_images_capped": any_capped,
    }
