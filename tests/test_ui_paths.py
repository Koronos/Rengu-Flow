"""Tests for repo-relative path resolution in the UI."""

from __future__ import annotations

from rengu_flow_ui.paths import resolve_repo_path
from rengu_flow_ui.settings import repo_root


def test_resolve_repo_path_relative() -> None:
    p = resolve_repo_path("output")
    assert p == (repo_root() / "output").resolve()
