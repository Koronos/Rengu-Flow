"""Tests for dataset image gallery preview."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="ui extra not installed (uv sync --extra ui)")
from fastapi.testclient import TestClient

from rengu_flow_ui.dataset_image_preview import (
    issue_image_token,
    list_dataset_preview_images,
    resolve_image_token,
)


def test_list_preview_images(tmp_path: Path) -> None:
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (d / "b.png").write_bytes(b"\x89PNG")
    content = f'resolutions = [512]\n\n[[directory]]\npath = "{d}"\nnum_repeats = 1\n'
    r = list_dataset_preview_images(content, limit=10)
    assert r["ok"] is True
    assert r["total"] == 2
    assert len(r["images"]) == 2
    assert all(img["token"] for img in r["images"])


def test_resolve_token_serves_file(tmp_path: Path) -> None:
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "pic.webp").write_bytes(b"RIFF")
    root = d.resolve()
    token = issue_image_token(0, "pic.webp", root)
    path = resolve_image_token(token)
    assert path.name == "pic.webp"
    assert path.is_file()


def test_resolve_rejects_traversal(tmp_path: Path) -> None:
    d = tmp_path / "imgs"
    d.mkdir()
    root = d.resolve()
    with pytest.raises(ValueError, match="Invalid"):
        resolve_image_token(issue_image_token(0, "../etc/passwd", root))


def test_preview_image_api(ui_client: TestClient, tmp_path: Path) -> None:
    d = tmp_path / "gallery"
    d.mkdir()
    (d / "one.jpg").write_bytes(b"\xff\xd8\xff")
    content = f'[[directory]]\npath = "{d}"\n'
    r = ui_client.post(
        "/api/v1/datasets/preview-images",
        json={"content": content, "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert len(body["images"]) == 1
    token = body["images"][0]["token"]
    img = ui_client.get(f"/api/v1/datasets/preview-image?t={token}")
    assert img.status_code == 200
    assert img.headers["content-type"].startswith("image/")
