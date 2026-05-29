"""Tests for UI documentation reader."""

from pathlib import Path

import pytest

from rengu_flow_ui.docs_reader import DocNotFoundError, DocPathError, list_docs_index, read_doc, resolve_doc_path


def test_read_web_ui_doc() -> None:
    doc = read_doc("docs/user/web-ui.md")
    assert "Web UI" in doc["content"]
    assert doc["path"] == "docs/user/web-ui.md"


def test_reject_path_traversal() -> None:
    with pytest.raises(DocNotFoundError):
        read_doc("docs/../../../etc/passwd")


def test_reject_encoded_path_traversal() -> None:
    with pytest.raises(DocNotFoundError):
        read_doc("docs/user/%2e%2e/%2e%2e/README.md")


def test_reject_double_encoded_path_traversal() -> None:
    with pytest.raises(DocNotFoundError):
        read_doc("docs/user/%252e%252e/%252e%252e/README.md")


def test_reject_null_byte_in_path() -> None:
    with pytest.raises(DocNotFoundError):
        read_doc("docs/user/web-ui.md\x00")


def test_reject_absolute_path() -> None:
    with pytest.raises(DocNotFoundError):
        read_doc("/docs/user/web-ui.md")


def test_reject_windows_absolute_path() -> None:
    with pytest.raises(DocNotFoundError):
        read_doc("C:/docs/user/web-ui.md")


def test_reject_non_md_extension() -> None:
    with pytest.raises(DocPathError, match="Only .md files"):
        read_doc("docs/user/web-ui.txt")


def test_missing_doc() -> None:
    with pytest.raises(DocNotFoundError):
        read_doc("docs/user/does-not-exist-xyz.md")


def test_symlink_escape_outside_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs_user = tmp_path / "docs" / "user"
    docs_user.mkdir(parents=True)
    (docs_user / "visible.md").write_text("# Visible\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (docs_user / "escape.md").symlink_to(outside)

    monkeypatch.setattr("rengu_flow_ui.docs_reader.repo_root", lambda: tmp_path)

    with pytest.raises(DocNotFoundError):
        resolve_doc_path("docs/user/escape.md", repo=tmp_path)


def test_list_docs_index() -> None:
    items = list_docs_index()
    assert len(items) >= 1
    paths = {item["path"] for item in items}
    assert "docs/user/web-ui.md" in paths
    web_ui = next(i for i in items if i["path"] == "docs/user/web-ui.md")
    assert web_ui["title"]
