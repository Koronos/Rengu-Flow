"""Tests for UI documentation reader."""

import pytest

from renga_flow_ui.docs_reader import DocNotFoundError, DocPathError, read_doc


def test_read_web_ui_doc() -> None:
    doc = read_doc("docs/user/web-ui.md")
    assert "Web UI" in doc["content"]
    assert doc["path"] == "docs/user/web-ui.md"


def test_reject_path_traversal() -> None:
    with pytest.raises(DocPathError):
        read_doc("docs/../../../etc/passwd")


def test_missing_doc() -> None:
    with pytest.raises(DocNotFoundError):
        read_doc("docs/user/does-not-exist-xyz.md")
