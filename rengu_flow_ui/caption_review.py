"""Per-image caption review for the web UI's caption editor (Studio → Caption editor).

The tag editor stages bulk ops over a session; this is the other half: read one folder's captions
page by page (every line, plus each target's control images in an edit set) and write ONE image's
caption back. Both go through ``rengu_flow.prep.caption_store`` — the same reader/writer the prep
stages and the tag editor use — so ``caption_format``/``caption_ext`` and the trainer's pairing rule
mean the same thing here as everywhere else.

Stateless on purpose: every call re-opens the folder. A save therefore always starts from what is
on disk now (a ``captions.json`` rewrite never resurrects a stale copy of the other entries), and
``expected`` lets the editor refuse to overwrite a caption that changed after it was loaded.

A folder that a prep job or a workflow prep step is writing right now is read-only: the stage would
overwrite a hand fix, or the fix would clobber the stage's output (:func:`active_writers`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rengu_flow.prep.caption_store import (
    CAPTIONS_JSON_FILE,
    FORMAT_JSON,
    FORMAT_SIDECAR,
    CaptionSet,
    CaptionStore,
)
from rengu_flow_ui.dataset_image_preview import issue_image_token

LIST_FILTERS = ("all", "uncaptioned", "unpaired")
FORMAT_AUTO = "auto"  # the trainer's rule: captions.json when the folder has one, else sidecars
MAX_PAGE_SIZE = 200

# One plain suffix: no separators, no second dot — the extension is joined onto image paths.
_SAFE_EXT = re.compile(r"^\.[A-Za-z0-9_-]{1,16}$")

_ACTIVE_JOB_STATES = ("running", "stopping")
_ACTIVE_WORKFLOW_STATUSES = ("running", "cancelling")
_ACTIVE_NODE_STATUSES = ("launching", "running", "stopping")


class FolderBusyError(RuntimeError):
    """A prep job or workflow step is writing this folder's captions (HTTP 409)."""


class CaptionConflictError(RuntimeError):
    """The caption on disk is no longer what the editor loaded (HTTP 409)."""


def _norm(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _open(
    path: str, fmt: str, ext: str, control_path: str | None = None
) -> CaptionSet:
    folder = Path(path).expanduser()
    if fmt == FORMAT_AUTO:
        fmt = FORMAT_JSON if (folder / CAPTIONS_JSON_FILE).is_file() else FORMAT_SIDECAR
    ext = ext if ext.startswith(".") else f".{ext}"
    if not _SAFE_EXT.match(ext):
        raise ValueError(f"Invalid caption extension: {ext!r}")
    control = Path(control_path).expanduser() if (control_path or "").strip() else None
    return CaptionStore.open(folder, fmt=fmt, ext=ext, control_path=control)


def _config_path(content: str) -> str | None:
    import toml

    try:
        path = toml.loads(content or "").get("path")
    except (toml.TomlDecodeError, TypeError):
        return None
    return _norm(path) if isinstance(path, str) and path.strip() else None


def active_writers(folder: str | Path) -> list[dict]:
    """Prep runs writing *folder* right now: running prep jobs, and workflow prep steps.

    A workflow step has no job row; the folder it processes is the ``path`` of the ``prep.toml``
    it launched with, in its node directory. A pending job has written nothing yet — it only
    blocks once it starts, which the save-time check catches.
    """
    from rengu_flow_ui import db, workflow_db

    target = _norm(folder)
    found: list[dict] = []
    for job in db.list_jobs(limit=500):
        if job.kind != "prep" or job.state not in _ACTIVE_JOB_STATES:
            continue
        if _config_path(job.config_content) == target:
            found.append(
                {"kind": "job", "id": str(job.id), "stage": job.extra_args or "",
                 "label": f"job #{job.id}"}
            )
    for wf in workflow_db.list_workflows():
        try:
            state = json.loads(wf.state_json or "{}")
        except json.JSONDecodeError:
            continue
        if state.get("status") not in _ACTIVE_WORKFLOW_STATUSES:
            continue
        for node_id, info in (state.get("nodes") or {}).items():
            if not isinstance(info, dict) or info.get("status") not in _ACTIVE_NODE_STATUSES:
                continue
            try:
                config = workflow_db.node_dir(wf.id, node_id) / "prep.toml"
            except KeyError:  # a node id that escapes the workflow dir: not a real step
                continue
            if config.is_file() and _config_path(config.read_text(encoding="utf-8")) == target:
                found.append(
                    {"kind": "workflow", "id": str(wf.id), "stage": node_id,
                     "label": f"workflow '{wf.name}' step {node_id}"}
                )
    return found


def _item(captions: CaptionSet, key: str, folder: Path, control_root: Path | None) -> dict:
    return {
        "key": key,
        "lines": captions.get_lines(key),
        "token": issue_image_token(0, key, folder),
        "controls": [
            {"name": p.name, "token": issue_image_token(0, p.name, control_root)}
            for p in captions.controls.get(key, [])
        ] if control_root is not None else [],
        "unpaired": captions.unpaired.get(key),
    }


def list_captions(
    path: str,
    *,
    fmt: str = FORMAT_SIDECAR,
    ext: str = ".txt",
    control_path: str | None = None,
    q: str = "",
    filter: str = "all",
    limit: int = 60,
    offset: int = 0,
) -> dict:
    """One page of a folder's captions, after the search/filter; counts are folder-wide."""
    if filter not in LIST_FILTERS:
        raise ValueError(f"Unknown filter {filter!r}; expected one of {LIST_FILTERS}")
    limit = max(1, min(int(limit), MAX_PAGE_SIZE))
    offset = max(0, int(offset))
    captions = _open(path, fmt, ext, control_path)
    folder = captions.folder.resolve()
    control_root = captions.control_path.resolve() if captions.control_path else None

    needle = q.strip().lower()
    keys = []
    for key in captions.keys():
        lines = captions.get_lines(key)
        if filter == "uncaptioned" and lines:
            continue
        if filter == "unpaired" and key not in captions.unpaired:
            continue
        if needle and needle not in key.lower() and not any(needle in l.lower() for l in lines):
            continue
        keys.append(key)

    active = active_writers(folder)
    return {
        "path": str(folder),
        "format": captions.fmt,
        "ext": captions.ext,
        "control_path": str(control_root) if control_root else None,
        "image_count": len(captions.images),
        "uncaptioned_count": sum(1 for k in captions.keys() if not captions.get_lines(k)),
        "unpaired_count": len(captions.unpaired),
        "total": len(keys),
        "offset": offset,
        "limit": limit,
        "read_only": bool(active),
        "active": active,
        "items": [
            _item(captions, key, folder, control_root) for key in keys[offset : offset + limit]
        ],
    }


def save_caption(
    path: str,
    key: str,
    lines: list[str],
    *,
    fmt: str = FORMAT_SIDECAR,
    ext: str = ".txt",
    expected: list[str] | None = None,
    backup: bool = False,
) -> dict:
    """Write one image's caption lines (blank lines dropped, as the trainer reads them).

    ``key`` must be an image of the folder — a name the store itself discovered, so nothing
    outside it (``../x.jpg``, ``sub/x.jpg``) can be addressed. ``expected``: the lines the editor
    loaded; a different caption on disk raises :class:`CaptionConflictError` instead of being
    overwritten. ``backup``: snapshot every caption file first (``CaptionSet.snapshot``, the tag
    editor's backups — restorable from its Backups dialog); the editor asks for it once per folder.
    """
    captions = _open(path, fmt, ext)
    if key not in captions.images:
        raise KeyError(key)
    busy = active_writers(captions.folder)
    if busy:
        raise FolderBusyError(
            f"{busy[0]['label']} is writing this folder; wait for it to finish before editing."
        )
    current = captions.get_lines(key)
    if expected is not None and current != [l.strip() for l in expected if l.strip()]:
        raise CaptionConflictError(
            f"The caption of {key} changed on disk since it was loaded; reload to see it."
        )
    backup_dir = captions.snapshot() if backup else None
    captions.set_lines(key, lines)
    written = captions.save()
    return {
        "key": key,
        "lines": captions.get_lines(key),
        "written": written,
        "backup": backup_dir.name if backup_dir else None,
    }
