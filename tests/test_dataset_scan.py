"""Tests for dataset folder scanning."""

from pathlib import Path

from renga_flow_ui.dataset_scan import preview_dataset_config, scan_folder


def test_scan_folder_counts_images(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("caption")
    r = scan_folder(tmp_path)
    assert r["ok"] is True
    assert r["image_count"] == 2
    assert r["caption_txt_files"] == 1


def test_scan_missing_dir() -> None:
    r = scan_folder("/nonexistent/path/for/renga_flow_test")
    assert r["ok"] is False


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
