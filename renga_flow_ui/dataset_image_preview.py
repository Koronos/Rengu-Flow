"""List and serve images from dataset [[directory]] paths for UI gallery preview."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import toml

from renga_flow_ui.dataset_scan import IMAGE_EXTENSIONS
from renga_flow_ui.settings import ui_data_dir, ui_token

DEFAULT_LIST_LIMIT = 24
MAX_LIST_LIMIT = 48
MAX_CATALOG_PER_DIR = 2000


def _signing_key() -> bytes:
    tok = ui_token()
    if tok:
        return tok.encode("utf-8")
    return hashlib.sha256(f"renga-flow-preview:{ui_data_dir()}".encode()).digest()


def _safe_filename(name: str) -> bool:
    if not name or name != Path(name).name:
        return False
    if "/" in name or "\\" in name:
        return False
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _directory_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    directories = config.get("directory") or []
    if not isinstance(directories, list):
        return []
    return [e for e in directories if isinstance(e, dict)]


def _directory_root(entry: dict[str, Any]) -> Path | None:
    path = entry.get("path", "")
    if not path or not isinstance(path, str):
        return None
    return Path(path).expanduser().resolve()


def list_images_in_directory(
    root: Path,
    *,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
) -> tuple[list[str], int]:
    """Return (page of filenames, total image count) for one folder (non-recursive)."""
    if not root.is_dir():
        return [], 0
    names: list[str] = []
    try:
        for entry in sorted(root.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            names.append(entry.name)
            if len(names) >= MAX_CATALOG_PER_DIR:
                break
    except OSError:
        return [], 0
    total = len(names)
    start = max(0, offset)
    end = start + max(1, min(limit, MAX_LIST_LIMIT))
    return names[start:end], total


def issue_image_token(directory_index: int, filename: str, root: Path) -> str:
    payload = json.dumps(
        {"i": directory_index, "n": filename, "r": str(root)},
        separators=(",", ":"),
        sort_keys=True,
    )
    sig = hmac.new(_signing_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    raw = f"{payload}.{sig}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode().rstrip("=")


def resolve_image_token(token: str) -> Path:
    """Resolve a signed preview token to an on-disk image path."""
    if not token:
        raise ValueError("Missing token")
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
        payload, sig = raw.rsplit(".", 1)
        expected = hmac.new(
            _signing_key(), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("Invalid token signature")
        data = json.loads(payload)
        directory_index = int(data["i"])
        filename = str(data["n"])
        root = Path(str(data["r"])).resolve()
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise ValueError("Invalid preview token") from e

    if directory_index < 0 or not _safe_filename(filename):
        raise ValueError("Invalid preview token payload")
    if not root.is_dir():
        raise ValueError("Dataset directory no longer exists")

    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError("Path escapes dataset directory") from e
    if not candidate.is_file():
        raise ValueError("Image file not found")
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Not an image file")
    return candidate


def list_dataset_preview_images(
    content: str,
    *,
    directory_index: int | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """List image files under configured dataset directories."""
    limit = max(1, min(int(limit), MAX_LIST_LIMIT))
    offset = max(0, int(offset))

    try:
        config = toml.loads(content)
    except Exception as e:
        return {"ok": False, "error": f"TOML parse error: {e}"}

    entries = _directory_entries(config)
    if not entries:
        return {
            "ok": True,
            "images": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "directories": [],
        }

    catalog: list[tuple[int, str, Path]] = []
    dir_meta: list[dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        if directory_index is not None and idx != directory_index:
            continue
        root = _directory_root(entry)
        path_str = str(entry.get("path", ""))
        if root is None:
            dir_meta.append(
                {
                    "index": idx,
                    "path": path_str,
                    "ok": False,
                    "error": "Missing path",
                    "image_count": 0,
                }
            )
            continue
        if not root.is_dir():
            dir_meta.append(
                {
                    "index": idx,
                    "path": str(root),
                    "ok": False,
                    "error": "Path is not a directory or does not exist",
                    "image_count": 0,
                }
            )
            continue

        names, total = list_images_in_directory(root, limit=MAX_CATALOG_PER_DIR, offset=0)
        dir_meta.append(
            {
                "index": idx,
                "path": str(root),
                "ok": True,
                "image_count": total,
                "catalog_capped": total >= MAX_CATALOG_PER_DIR,
            }
        )
        for name in names:
            catalog.append((idx, name, root))

    page = catalog[offset : offset + limit]
    images = [
        {
            "directory_index": idx,
            "name": name,
            "token": issue_image_token(idx, name, root),
        }
        for idx, name, root in page
    ]

    total_images = len(catalog)
    return {
        "ok": True,
        "images": images,
        "total": total_images,
        "limit": limit,
        "offset": offset,
        "directories": dir_meta,
    }
