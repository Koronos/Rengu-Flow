"""Tests for dataset folder scanning."""

from pathlib import Path

from renga_flow_ui.dataset_scan import (
    list_image_files_page,
    preview_dataset_config,
    scan_folder,
)


def test_scan_folder_samples_multiple_images(tmp_path: Path) -> None:
    for i in range(6):
        (tmp_path / f"img_{i}.jpg").write_bytes(b"x")
    r = scan_folder(tmp_path, max_samples=12)
    assert r["ok"] is True
    assert len(r["sample_files"]) >= 4


def test_scan_folder_counts_images(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("caption for a")
    (tmp_path / "note.txt").write_text("orphan caption file")
    r = scan_folder(tmp_path)
    assert r["ok"] is True
    assert r["image_count"] == 2
    assert r["caption_txt_files"] == 1
    assert r["count_capped"] is False


def test_scan_folder_paired_caption_txt(tmp_path: Path) -> None:
    (tmp_path / "img.jpg").write_bytes(b"x")
    (tmp_path / "img.txt").write_text("line one")
    r = scan_folder(tmp_path)
    assert r["caption_txt_files"] == 1
    (tmp_path / "img.txt").unlink()
    r2 = scan_folder(tmp_path)
    assert r2["caption_txt_files"] == 0


def test_scan_missing_dir() -> None:
    r = scan_folder("/nonexistent/path/for/renga_flow_test")
    assert r["ok"] is False


def test_scan_folder_count_cap(tmp_path: Path) -> None:
    for i in range(20):
        (tmp_path / f"img_{i:03d}.jpg").write_bytes(b"x")
    r = scan_folder(tmp_path, count_cap=5, max_samples=3)
    assert r["ok"] is True
    assert r["image_count"] == 5
    assert r["count_capped"] is True
    assert len(r["sample_files"]) <= 3


def test_list_image_files_page_offset_limit(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"{i:02d}.jpg").write_bytes(b"x")
    page0 = list_image_files_page(tmp_path, offset=0, limit=3)
    assert len(page0["names"]) == 3
    assert page0["total"] >= 3
    page1 = list_image_files_page(tmp_path, offset=3, limit=3)
    assert len(page1["names"]) == 3
    assert page0["names"] != page1["names"]


def test_list_image_files_page_respects_count_cap(tmp_path: Path) -> None:
    for i in range(30):
        (tmp_path / f"f{i}.jpg").write_bytes(b"x")
    listed = list_image_files_page(tmp_path, offset=0, limit=2, count_cap=5)
    assert len(listed["names"]) == 2
    assert listed["total"] == 5
    assert listed["count_capped"] is True


def test_preview_aggregates(tmp_path: Path) -> None:
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "x.jpg").write_bytes(b"x")
    (d2 / "y.jpg").write_bytes(b"x")
    (d2 / "z.jpg").write_bytes(b"x")
    cfg = {
        "directory": [
            {"path": str(d1), "num_repeats": 1},
            {"path": str(d2), "num_repeats": 1},
        ]
    }
    p = preview_dataset_config(cfg)
    assert p["directory_count"] == 2
    assert p["total_images"] == 3
