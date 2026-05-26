"""Tests for optional status.json writer."""

import json
from pathlib import Path

from renga_flow.control.status_file import read_status_file, write_status_file


def test_write_status_file(tmp_path: Path) -> None:
    write_status_file(
        tmp_path,
        step=10,
        examples=1000,
        epoch=2,
        loss=0.42,
    )
    data = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert data["step"] == 10
    assert data["examples"] == 1000
    assert data["epoch"] == 2
    assert data["loss"] == 0.42
    assert "updated_at" in data


def test_read_status_file(tmp_path: Path) -> None:
    write_status_file(tmp_path, step=1, examples=2, epoch=0, loss=0.1)
    data = read_status_file(tmp_path)
    assert data is not None
    assert data["step"] == 1
    assert read_status_file(tmp_path / "missing") is None
