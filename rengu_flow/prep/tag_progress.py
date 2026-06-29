"""Resumable tag-job progress.

Chunk-outer tagging persists each fully-processed chunk to the caption files immediately,
so the captions are usable mid-run and a stopped job already has what it finished. This
side file records which image keys are *done* for the current model set — the bit the
caption files alone can't carry (e.g. an image whose tags came back empty still counts as
processed, and a changed model set should re-tag). It lives under the managed prep storage
(``<data_dir>/prep/<folder-id>``), never inside the dataset, so it can't collide with image
or caption entries. Deleted once the job completes.
"""

from __future__ import annotations

import json
from pathlib import Path

from rengu_flow.prep.storage import prep_storage_dir
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

_FILE = "tag_progress.json"


def _path(folder: str | Path) -> Path:
    return prep_storage_dir(folder) / _FILE


def load_done(folder: str | Path, models: list[str]) -> set[str]:
    """Image keys already tagged in a prior run — only if it used the SAME model set
    (a different set means different output, so re-tag from scratch)."""
    p = _path(folder)
    if not p.is_file():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — corrupt/partial progress file: ignore, re-tag
        logger.warning("tag_progress: ignoring unreadable %s: %s", p, exc)
        return set()
    if data.get("models") != list(models):
        return set()
    return set(data.get("done", []))


def save_done(folder: str | Path, models: list[str], done: set[str]) -> None:
    """Atomically persist the set of completed image keys for this model set."""
    p = _path(folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"models": list(models), "done": sorted(done)}), encoding="utf-8")
    tmp.replace(p)


def clear(folder: str | Path) -> None:
    """Remove the progress file (job finished — nothing to resume)."""
    _path(folder).unlink(missing_ok=True)
