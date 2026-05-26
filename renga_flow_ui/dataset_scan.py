"""Scan dataset directories for image counts and caption hints."""

from __future__ import annotations

from pathlib import Path

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


def scan_folder(path: str | Path, *, max_samples: int = 12) -> dict:
    """Summarize one directory path for UI preview."""
    root = Path(path).expanduser()
    if not root.is_dir():
        return {
            "ok": False,
            "path": str(root),
            "error": "Path is not a directory or does not exist.",
        }

    image_count = 0
    video_count = 0
    caption_txt = 0
    samples: list[str] = []
    sample_stems: set[str] = set()

    try:
        for entry in sorted(root.iterdir()):
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                image_count += 1
                if len(samples) < max_samples:
                    samples.append(entry.name)
                    sample_stems.add(entry.stem)
            elif suffix in {v.lower() for v in VIDEO_EXTENSIONS}:
                video_count += 1
                if len(samples) < max_samples:
                    samples.append(entry.name)
                    sample_stems.add(entry.stem)
            elif suffix == ".txt" and entry.stem not in sample_stems:
                caption_txt += 1
    except OSError as e:
        return {"ok": False, "path": str(root.resolve()), "error": str(e)}

    has_captions_json = (root / CAPTIONS_JSON_FILE).is_file()

    return {
        "ok": True,
        "path": str(root.resolve()),
        "image_count": image_count,
        "video_count": video_count,
        "caption_txt_files": caption_txt,
        "has_captions_json": has_captions_json,
        "sample_files": samples,
        "total_media": image_count + video_count,
    }


def preview_dataset_config(dataset_config: dict) -> dict:
    """Scan all [[directory]] entries in a parsed dataset config."""
    directories = dataset_config.get("directory") or []
    if not isinstance(directories, list):
        directories = []
    per_dir = []
    total_images = 0
    total_videos = 0
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
    return {
        "directories": per_dir,
        "total_images": total_images,
        "total_videos": total_videos,
        "directory_count": len(directories),
    }
