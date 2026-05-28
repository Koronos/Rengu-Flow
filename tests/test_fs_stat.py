"""Tests for filesystem stat used by UI path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from renga_flow_ui.fs_stat import resolve_validated_path, stat_path
from renga_flow_ui.settings import repo_root


def test_stat_path_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("renga_flow_ui.fs_stat.repo_root", lambda: tmp_path)
    f = tmp_path / "model.safetensors"
    f.write_text("x", encoding="utf-8")

    result = stat_path("model.safetensors", expect="file")
    assert result["exists"] is True
    assert result["is_file"] is True
    assert result["is_dir"] is False
    assert "error" not in result


def test_stat_path_existing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("renga_flow_ui.fs_stat.repo_root", lambda: tmp_path)
    d = tmp_path / "data"
    d.mkdir()

    result = stat_path("data", expect="dir")
    assert result["exists"] is True
    assert result["is_dir"] is True
    assert "error" not in result


def test_stat_path_missing() -> None:
    result = stat_path("__definitely_missing_renga_flow_path__")
    assert result["exists"] is False
    assert result["error"] == "Path does not exist"


def test_stat_path_wrong_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("renga_flow_ui.fs_stat.repo_root", lambda: tmp_path)
    d = tmp_path / "folder"
    d.mkdir()

    result = stat_path("folder", expect="file")
    assert result["exists"] is True
    assert result["error"] == "Expected a file"


def test_stat_path_blocks_parent_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("renga_flow_ui.fs_stat.repo_root", lambda: tmp_path)
    result = stat_path("../outside")
    assert result["exists"] is False
    assert ".." in (result.get("error") or "")


def test_resolve_validated_path_relative_under_repo() -> None:
    p = resolve_validated_path("output")
    assert p == (repo_root() / "output").resolve()


def test_fs_stat_api(ui_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("renga_flow_ui.fs_stat.repo_root", lambda: tmp_path)
    sample = tmp_path / "examples"
    sample.mkdir()
    (sample / "minimal_dataset.toml").write_text("x", encoding="utf-8")

    r = ui_client.post(
        "/api/v1/fs/stat",
        json={"path": "examples/minimal_dataset.toml", "expect": "file"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is True
    assert body["is_file"] is True

    r2 = ui_client.get(
        "/api/v1/fs/stat",
        params={"path": "examples", "expect": "dir"},
    )
    assert r2.status_code == 200
    assert r2.json()["is_dir"] is True
