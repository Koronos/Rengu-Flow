"""Tests for rengu_flow.utils.paths."""

from __future__ import annotations

from rengu_flow.utils.paths import path_is_under


def test_path_is_under(tmp_path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    child = root / "img" / "a.jpg"
    child.parent.mkdir(parents=True)
    child.touch()
    other = tmp_path / "data_extra"
    other.mkdir()

    assert path_is_under(child, root)
    assert path_is_under(root, root)
    assert not path_is_under(other, root)
