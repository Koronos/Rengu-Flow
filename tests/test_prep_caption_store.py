"""CaptionStore: sidecar/JSON round-trips, atomic save, snapshot/restore, quarantine."""

import json
import shutil
from pathlib import Path

import pytest

from rengu_flow.prep.caption_store import CaptionStore

pytestmark = pytest.mark.no_ui_db

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)


@pytest.fixture
def img_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        shutil.copy(FIXTURE_JPG, d / name)
    return d


def test_sidecar_roundtrip_preserves_multiline(img_dir):
    (img_dir / "a.txt").write_text("1girl, long hair\nA girl with long hair.\n")
    (img_dir / "b.txt").write_text("tag only\n")
    cs = CaptionStore.open(img_dir)
    assert cs.get_lines("a.jpg") == ["1girl, long hair", "A girl with long hair."]
    assert cs.get_lines("b.jpg") == ["tag only"]
    assert cs.get_lines("c.jpg") == []  # no sidecar
    assert cs.get_tags("a.jpg") == ["1girl", "long hair"]

    cs.set_line("a.jpg", 0, "1girl, short hair")
    written = cs.save()
    assert written == ["a.txt"]
    assert (img_dir / "a.txt").read_text() == "1girl, short hair\nA girl with long hair.\n"
    # No-op save writes nothing.
    assert cs.save() == []


def test_sidecar_custom_extension(img_dir):
    (img_dir / "a.caption").write_text("solo\n")
    cs = CaptionStore.open(img_dir, ext=".caption")
    assert cs.get_lines("a.jpg") == ["solo"]
    cs.append_line("a.jpg", "second line")
    cs.save()
    assert (img_dir / "a.caption").read_text() == "solo\nsecond line\n"
    # ext without leading dot also accepted
    cs2 = CaptionStore.open(img_dir, ext="caption")
    assert cs2.get_lines("a.jpg") == ["solo", "second line"]


def test_sidecar_empty_lines_skipped_like_trainer(img_dir):
    (img_dir / "a.txt").write_text("first\n\n  \nsecond\n")
    cs = CaptionStore.open(img_dir)
    assert cs.get_lines("a.jpg") == ["first", "second"]


def test_sidecar_clearing_lines_deletes_file(img_dir):
    (img_dir / "a.txt").write_text("doomed\n")
    cs = CaptionStore.open(img_dir)
    cs.set_lines("a.jpg", [])
    written = cs.save()
    assert written == ["a.txt"]
    assert not (img_dir / "a.txt").exists()


def test_json_roundtrip_list_and_legacy_string(img_dir):
    (img_dir / "captions.json").write_text(
        json.dumps({"a.jpg": ["tags here", "a caption"], "b.jpg": "legacy string"})
    )
    cs = CaptionStore.open(img_dir, fmt="json")
    assert cs.get_lines("a.jpg") == ["tags here", "a caption"]
    assert cs.get_lines("b.jpg") == ["legacy string"]
    assert cs.get_lines("c.jpg") == []  # missing key

    cs.set_line("c.jpg", 0, "1girl")
    cs.save()
    data = json.loads((img_dir / "captions.json").read_text())
    assert data["a.jpg"] == ["tags here", "a caption"]
    assert data["b.jpg"] == ["legacy string"]  # coerced to list (trainer requires lists)
    assert data["c.jpg"] == ["1girl"]


def test_json_never_writes_empty_caption_list(img_dir):
    cs = CaptionStore.open(img_dir, fmt="json")
    cs.set_lines("a.jpg", ["x"])
    cs.save()
    cs.set_lines("a.jpg", [])
    cs.save()
    data = json.loads((img_dir / "captions.json").read_text())
    # Empty list would silently drop the image from training; [""] keeps one empty caption.
    assert data["a.jpg"] == [""]


def test_snapshot_and_restore_sidecar(img_dir):
    (img_dir / "a.txt").write_text("original a\n")
    (img_dir / "b.txt").write_text("original b\n")
    cs = CaptionStore.open(img_dir)
    backup = cs.snapshot()
    assert backup.is_dir()
    assert (backup / "a.txt").read_text() == "original a\n"

    # Mutate: edit a, delete b's caption, add caption for c.
    cs.set_lines("a.jpg", ["edited"])
    cs.set_lines("b.jpg", [])
    cs.set_lines("c.jpg", ["new file"])
    cs.save()
    assert (img_dir / "c.txt").exists()

    backups = CaptionStore.list_backups(img_dir)
    assert [b["name"] for b in backups] == [backup.name]
    CaptionStore.restore_snapshot(img_dir, backup.name)
    assert (img_dir / "a.txt").read_text() == "original a\n"
    assert (img_dir / "b.txt").read_text() == "original b\n"
    assert not (img_dir / "c.txt").exists()  # extra caption files removed


def test_restore_unknown_backup_raises(img_dir):
    with pytest.raises(FileNotFoundError):
        CaptionStore.restore_snapshot(img_dir, "nope")


def test_quarantine_moves_images_and_sidecars(img_dir):
    (img_dir / "a.txt").write_text("bad tag\n")
    cs = CaptionStore.open(img_dir)
    qdir = cs.quarantine(["a.jpg"])
    assert not (img_dir / "a.jpg").exists()
    assert not (img_dir / "a.txt").exists()
    assert (qdir / "a.jpg").exists()
    assert (qdir / "a.txt").exists()
    assert "a.jpg" not in cs.images

    batches = CaptionStore.list_quarantine(img_dir)
    assert batches and batches[0]["images"] == ["a.jpg"]
    restored = CaptionStore.restore_quarantine(img_dir, batches[0]["name"])
    assert restored == ["a.jpg"]
    assert (img_dir / "a.jpg").exists()
    assert (img_dir / "a.txt").read_text() == "bad tag\n"
    assert CaptionStore.list_quarantine(img_dir) == []


def test_quarantine_json_format_rewrites_captions_json(img_dir):
    (img_dir / "captions.json").write_text(json.dumps({"a.jpg": ["x"], "b.jpg": ["y"]}))
    cs = CaptionStore.open(img_dir, fmt="json")
    cs.quarantine(["a.jpg"])
    data = json.loads((img_dir / "captions.json").read_text())
    assert "a.jpg" not in data and data["b.jpg"] == ["y"]
    batches = CaptionStore.list_quarantine(img_dir)
    CaptionStore.restore_quarantine(img_dir, batches[0]["name"])
    data = json.loads((img_dir / "captions.json").read_text())
    assert data["a.jpg"] == ["x"]


def test_prep_dir_invisible_to_discovery(img_dir):
    cs = CaptionStore.open(img_dir)
    cs.snapshot()
    cs2 = CaptionStore.open(img_dir)
    # Backup dir lives under .rengu_prep and never shows up as images.
    assert sorted(cs2.images) == ["a.jpg", "b.jpg", "c.jpg"]


def test_trainer_parity_txt_semantics(img_dir):
    """CaptionStore reads sidecars exactly like the trainer's per-line reader."""
    from rengu_flow.data.dataset import _read_captions_from_txt_per_line

    (img_dir / "a.txt").write_text("  one  \n\ntwo, three\n")
    cs = CaptionStore.open(img_dir)
    trainer = _read_captions_from_txt_per_line(str(img_dir / "a.txt"))
    assert cs.get_lines("a.jpg") == trainer == ["one", "two, three"]
