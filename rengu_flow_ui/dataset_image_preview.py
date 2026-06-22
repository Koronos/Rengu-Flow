"""List and serve images from dataset [[directory]] paths for UI gallery preview."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import toml

from rengu_flow_ui.dataset_scan import (
    IMAGE_EXTENSIONS as _IMAGE_SUFFIXES,
    UI_LIST_COUNT_CAP,
    list_image_files_page,
)
from rengu_flow_ui.settings import ui_data_dir, ui_token

DEFAULT_LIST_LIMIT = 24
MAX_LIST_LIMIT = 48


def _signing_key() -> bytes:
    tok = ui_token()
    if tok:
        return tok.encode("utf-8")
    return hashlib.sha256(f"rengu-flow-preview:{ui_data_dir()}".encode()).digest()


def _safe_filename(name: str) -> bool:
    if not name or name != Path(name).name:
        return False
    if "/" in name or "\\" in name:
        return False
    return Path(name).suffix.lower() in _IMAGE_SUFFIXES


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
    if candidate.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError("Not an image file")
    return candidate


def list_dataset_preview_images(
    content: str,
    *,
    directory_index: int | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """List one page of image files (bounded scan per folder)."""
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
            "total_capped": False,
            "limit": limit,
            "offset": offset,
            "directories": [],
        }

    dir_meta: list[dict[str, Any]] = []
    page_items: list[tuple[int, str, Path]] = []
    total_for_response = 0
    total_capped = False

    if directory_index is not None:
        if directory_index < 0 or directory_index >= len(entries):
            return {"ok": False, "error": "directory_index out of range"}
        indices = [directory_index]
    else:
        indices = list(range(len(entries)))

    remaining_offset = offset
    remaining_limit = limit

    for idx in indices:
        if remaining_limit <= 0:
            break
        entry = entries[idx]
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

        listed = list_image_files_page(
            root,
            offset=remaining_offset,
            limit=remaining_limit,
            count_cap=UI_LIST_COUNT_CAP,
        )
        names = listed["names"]
        dir_meta.append(
            {
                "index": idx,
                "path": str(root),
                "ok": True,
                "image_count": listed["total"],
                "count_capped": listed["count_capped"],
            }
        )
        if listed["count_capped"]:
            total_capped = True

        for name in names:
            page_items.append((idx, name, root))

        if directory_index is not None:
            total_for_response = listed["total"]
            break

        remaining_offset = max(0, remaining_offset - listed["total"])
        remaining_limit = limit - len(page_items)
        if len(page_items) >= limit:
            total_for_response = sum(
                m.get("image_count", 0) for m in dir_meta if m.get("ok")
            )
            break

    if directory_index is None and page_items and total_for_response == 0:
        total_for_response = sum(m.get("image_count", 0) for m in dir_meta if m.get("ok"))

    images = [
        {
            "directory_index": idx,
            "name": name,
            "token": issue_image_token(idx, name, root),
        }
        for idx, name, root in page_items
    ]

    return {
        "ok": True,
        "images": images,
        "total": total_for_response,
        "total_capped": total_capped,
        "limit": limit,
        "offset": offset,
        "directories": dir_meta,
    }
