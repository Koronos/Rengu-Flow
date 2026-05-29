"""Tests for rengu_flow.utils.save_io."""

import errno
from pathlib import Path

import pytest
import torch

from rengu_flow.utils.save_io import (
    atomic_save_safetensors,
    cleanup_export_dir,
    is_disk_full_error,
    parse_export_sort_key,
    prepare_export_tmp,
    rollback_failed_checkpoint,
    snapshot_global_step_dirs,
)


def test_is_disk_full_error_enospc():
    assert is_disk_full_error(OSError(errno.ENOSPC, "No space"))


def test_is_disk_full_error_wrapped():
    try:
        raise OSError(errno.ENOSPC, "disk") from ValueError("x")
    except OSError as exc:
        assert is_disk_full_error(exc)


def test_atomic_save_safetensors(tmp_path: Path):
    path = tmp_path / "out.safetensors"
    atomic_save_safetensors(path, {"w": torch.zeros(2)})
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_prepare_export_tmp_removes_stale(tmp_path: Path):
    save_dir = tmp_path / "step1"
    stale = save_dir / "tmp"
    stale.mkdir(parents=True)
    (stale / "old.bin").write_bytes(b"x")
    tmp = prepare_export_tmp(save_dir)
    assert tmp.is_dir()
    assert not (stale / "old.bin").exists()


def test_cleanup_export_dir(tmp_path: Path):
    save_dir = tmp_path / "step2"
    save_dir.mkdir()
    (save_dir / "model.safetensors").write_bytes(b"partial")
    (save_dir / "tmp").mkdir()
    cleanup_export_dir(save_dir)
    assert not (save_dir / "model.safetensors").exists()
    assert not (save_dir / "tmp").exists()


def test_snapshot_and_rollback(tmp_path: Path):
    (tmp_path / "global_step100").mkdir()
    before = snapshot_global_step_dirs(tmp_path)
    (tmp_path / "global_step200").mkdir()
    after = snapshot_global_step_dirs(tmp_path)
    rollback_failed_checkpoint(tmp_path, before, after)
    assert not (tmp_path / "global_step200").exists()
    assert (tmp_path / "global_step100").exists()


def test_parse_export_sort_key():
    assert parse_export_sort_key("step500", 100) == 500
    assert parse_export_sort_key("epoch2", 50) == 100
    assert parse_export_sort_key("signal_step9", 50) is None
