"""Disk I/O helpers for checkpoints and model export (atomic writes, ENOSPC handling)."""

from __future__ import annotations

import errno
import os
import re
import shutil
from pathlib import Path
from typing import Any

import safetensors.torch

_STEP_DIR_RE = re.compile(r"^step(\d+)$")
_EPOCH_DIR_RE = re.compile(r"^epoch(\d+)$")


def is_disk_full_error(exc: BaseException) -> bool:
    """True when *exc* indicates the filesystem is out of space."""
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
        return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    return isinstance(cause, OSError) and cause.errno in (errno.ENOSPC, errno.EDQUOT)


def global_step_sort_key(dir_name: str) -> int:
    suffix = dir_name.removeprefix("global_step")
    return int(suffix) if suffix.isdigit() else 0


def snapshot_global_step_dirs(save_root: Path) -> set[str]:
    """Return names of ``global_step*`` directories under *save_root*."""
    if not save_root.is_dir():
        return set()
    return {
        p.name
        for p in save_root.iterdir()
        if p.is_dir() and p.name.startswith("global_step")
    }


def rollback_failed_checkpoint(save_root: Path, before: set[str], after: set[str]) -> None:
    """Remove checkpoint dirs created during a failed save and fix ``latest`` if needed."""
    new_dirs = sorted(after - before)
    for name in new_dirs:
        path = save_root / name
        if path.is_dir():
            print(f"Rolling back incomplete checkpoint directory {name}")
            shutil.rmtree(path)
    latest = save_root / "latest"
    if not latest.exists():
        return
    try:
        target = latest.resolve()
    except OSError:
        return
    if target.is_dir() and target.name in new_dirs:
        remaining = sorted(
            (save_root / n for n in before if (save_root / n).is_dir()),
            key=lambda p: global_step_sort_key(p.name),
        )
        if remaining:
            print(f"Restoring latest pointer to {remaining[-1].name}")
            if latest.is_symlink():
                latest.unlink()
            elif latest.is_file():
                latest.unlink()
            latest.symlink_to(remaining[-1].name)
        else:
            print("Removing broken latest checkpoint pointer")
            if latest.is_symlink() or latest.is_file():
                latest.unlink()


def atomic_save_safetensors(
    path: Path | str,
    state_dict: dict[str, Any],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write *state_dict* atomically via a sibling ``.tmp`` file and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        safetensors.torch.save_file(state_dict, tmp, metadata=metadata or {"format": "pt"})
        os.replace(tmp, path)
    except OSError:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def prepare_export_tmp(save_dir: Path) -> Path:
    """Ensure ``save_dir/tmp`` exists, removing a leftover tmp tree from a crashed save."""
    tmp_dir = save_dir / "tmp"
    if tmp_dir.exists():
        print(f"Removing leftover export tmp directory {tmp_dir}")
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


def cleanup_export_dir(save_dir: Path) -> None:
    """Remove partial export artifacts under *save_dir* after a failed write."""
    if not save_dir.exists():
        return
    tmp_dir = save_dir / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    for pattern in ("*.safetensors", "*.safetensors.tmp"):
        for path in save_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def parse_export_sort_key(dir_name: str, steps_per_epoch: int) -> int | None:
    """Sort key for export retention; ``None`` if not a scheduled export folder name."""
    m = _STEP_DIR_RE.match(dir_name)
    if m:
        return int(m.group(1))
    m = _EPOCH_DIR_RE.match(dir_name)
    if m:
        epoch = int(m.group(1))
        return epoch * max(steps_per_epoch, 1)
    return None


def list_prunable_export_dirs(save_root: Path) -> list[Path]:
    """Export directories eligible for automatic retention pruning."""
    if not save_root.is_dir():
        return []
    result = []
    for p in save_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("signal_step"):
            continue
        if _STEP_DIR_RE.match(p.name) or _EPOCH_DIR_RE.match(p.name):
            result.append(p)
    return result
