"""The one-pass ``os.scandir`` folder scans give exactly what the ``Path.glob`` + ``is_file``
scans they replaced gave: same files, same order, same sidecar lookups — on a tree full of
edge cases (extension case, hidden files, dotted names, folders named like images, sidecars
without an image, ``.bak``/``.json`` neighbours, symlinks when the OS allows them)."""

import json
import os
from pathlib import Path

import pytest

from rengu_flow.data.control import _NUMBERED_STEM, CONTROL_IMAGE_EXTENSIONS, index_control_dir
from rengu_flow.prep import aesthetic_scorer, iqa_scorer, quality_index
from rengu_flow.prep.caption_store import IMAGE_EXTENSIONS, CaptionStore
from rengu_flow.utils.paths import dir_files, glob_names, name_stem, name_suffix

pytestmark = pytest.mark.no_ui_db

CASE_INSENSITIVE_FS = os.name == "nt"


@pytest.fixture(autouse=True)
def _isolate_prep_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("RENGU_FLOW_UI_DATA", str(tmp_path / "appdata"))


# -- the replaced implementations (verbatim from before the scandir change) -----------------


def old_open_images(folder: Path) -> dict:
    return {
        p.name: p
        for p in sorted(folder.glob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def old_sidecar_captions(images: dict, ext: str) -> dict:
    out = {}
    for key, image in images.items():
        sidecar = image.with_suffix(ext)
        out[key] = sidecar.read_text(encoding="utf-8") if sidecar.is_file() else None
    return out


def old_snapshot_names(folder: Path, ext: str) -> list:
    return sorted(p.name for p in folder.glob(f"*{ext}") if p.is_file())


def old_index_control_dir(control_path) -> dict:
    exact: dict = {}
    numbered: dict = {}
    for p in sorted(Path(control_path).glob("*")):
        if not p.is_file() or p.suffix.lower() not in CONTROL_IMAGE_EXTENSIONS:
            continue
        exact.setdefault(p.stem, []).append(p)
        m = _NUMBERED_STEM.match(p.stem)
        if m:
            numbered.setdefault(m.group("base"), {}).setdefault(int(m.group("n")), []).append(p)
    return {"exact": exact, "numbered": numbered}


def old_iterdir_images(src: Path, exts) -> list:
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts)


# -- the tree --------------------------------------------------------------------------------

IMAGES = [
    "a.png", "B.PNG", "c.JpG", "d.jpeg", "e.tar.png", "..png", ".hidden.png", "Zeta.png",
    "alpha.png", "_under.png", "10.png", "9.png", "ñ.png", "É.webp", "a_0.png", "a_1.png",
    "a_2.JPG", "m.gif", "n.avif", "o.heic", "x2.png",
]
OTHERS = [
    ".png",  # a dotfile: no suffix for pathlib
    "noext", "f.", "g.txt.bak", "d.json", "notes.md", "orphan.txt", "a.txt", "B.txt",
    "e.tar.txt", "..txt", ".hidden.txt", "Zeta.TXT", "alpha.caption", "_under.txt.bak",
    "captions.json",
]
if not CASE_INSENSITIVE_FS:
    OTHERS += ["c.TXT", "c.txt"]  # two sidecars differing only by case (POSIX only)
else:
    OTHERS += ["c.TXT"]  # found for c.JpG through the case-insensitive lookup


@pytest.fixture
def tree(tmp_path) -> tuple[Path, bool]:
    d = tmp_path / "ds"
    d.mkdir()
    for name in IMAGES:
        (d / name).write_bytes(b"img")
    for name in OTHERS:
        (d / name).write_text(f"caption of {name}\n", encoding="utf-8")
    (d / "captions.json").write_text(json.dumps({"a.png": ["json a"]}), encoding="utf-8")
    # Folders that look like images/sidecars, and a nested image that must not be picked up.
    (d / "dir.png").mkdir()
    (d / "x2.txt").mkdir()  # the "sidecar" of x2.png is a folder -> no caption
    (d / "sub").mkdir()
    (d / "sub" / "nested.png").write_bytes(b"img")
    (d / "sub" / "nested.txt").write_text("nested\n", encoding="utf-8")
    symlinks = True
    try:
        os.symlink(d / "a.png", d / "link.png")
        os.symlink(d / "a.txt", d / "link.txt")
        os.symlink(d / "missing.png", d / "broken.png")
        os.symlink(d / "sub", d / "dirlink.png", target_is_directory=True)
    except (OSError, NotImplementedError):
        symlinks = False
    return d, symlinks


def test_tree_exercises_the_edge_cases(tree):
    folder, symlinks = tree
    names = dir_files(folder)
    assert "dir.png" not in names and "sub" not in names and "x2.txt" not in names
    assert ".hidden.png" in names and "..png" in names
    if symlinks:
        assert "link.png" in names and "broken.png" not in names and "dirlink.png" not in names


def test_dir_files_matches_sorted_glob(tree):
    folder, _ = tree
    assert dir_files(folder) == [p.name for p in sorted(folder.glob("*")) if p.is_file()]
    for exts in (IMAGE_EXTENSIONS, CONTROL_IMAGE_EXTENSIONS, {".txt"}):
        assert dir_files(folder, exts) == [
            p.name for p in sorted(folder.glob("*")) if p.is_file() and p.suffix.lower() in exts
        ]


@pytest.mark.parametrize("pattern", ["*.txt", "*.TXT", "*.caption", "*.json", "*.txt.bak", "*"])
def test_glob_names_matches_path_glob(tree, pattern):
    folder, _ = tree
    got = glob_names(dir_files(folder), pattern)
    assert sorted(got) == sorted(p.name for p in folder.glob(pattern) if p.is_file())


@pytest.mark.parametrize(
    "name", ["a.png", "e.tar.png", "..png", ".png", ".hidden.png", "noext", "f.", "a..b"]
)
def test_name_suffix_and_stem_match_pathlib(name):
    assert name_suffix(name) == Path(name).suffix
    assert name_stem(name) == Path(name).stem


@pytest.mark.parametrize("ext", [".txt", ".caption", ".TXT"])
def test_open_sidecar_matches_old_scan(tree, ext):
    folder, _ = tree
    cs = CaptionStore.open(folder, fmt="sidecar", ext=ext)
    old = old_open_images(folder)
    assert list(cs.images.items()) == list(old.items())  # same keys, paths AND order
    expected = {
        key: [] if text is None else [line.strip() for line in text.splitlines() if line.strip()]
        for key, text in old_sidecar_captions(old, ext).items()
    }
    assert cs.captions == expected
    assert list(cs.captions) == list(old)


def test_open_json_matches_old_scan(tree):
    folder, _ = tree
    cs = CaptionStore.open(folder, fmt="json")
    assert list(cs.images.items()) == list(old_open_images(folder).items())
    assert cs.captions["a.png"] == ["json a"]


@pytest.mark.parametrize("ext", [".txt", ".TXT", ".caption"])
def test_snapshot_and_restore_match_old_scan(tree, ext):
    folder, _ = tree
    cs = CaptionStore.open(folder, fmt="sidecar", ext=ext)
    backup = cs.snapshot()
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == old_snapshot_names(folder, ext) + ["captions.json"]

    # Extra caption files appear after the snapshot; restore removes exactly the ones the old
    # glob found (and still restores every snapshotted file).
    (folder / f"late{ext}").write_text("late\n", encoding="utf-8")
    (folder / f"LATE2{ext.upper()}").write_text("late\n", encoding="utf-8")
    snapshotted = set(manifest["files"])
    extras = {n for n in old_snapshot_names(folder, ext) if n not in snapshotted}
    restored = CaptionStore.restore_snapshot(folder, backup.name)
    assert restored == sorted(extras | snapshotted)
    assert not any((folder / n).exists() for n in extras)


def test_index_control_dir_matches_old_scan(tree):
    folder, _ = tree
    new, old = index_control_dir(folder), old_index_control_dir(folder)
    assert new == old
    assert list(new["exact"]) == list(old["exact"])  # insertion order too
    assert new["numbered"]["a"] == {0: [folder / "a_0.png"], 1: [folder / "a_1.png"], 2: [folder / "a_2.JPG"]}


def test_prep_stage_scans_match_old_iterdir(tree):
    folder, _ = tree
    assert quality_index._scan(folder) == old_iterdir_images(folder, IMAGE_EXTENSIONS)
    for mod in (aesthetic_scorer, iqa_scorer):
        assert mod._list_images(folder) == old_iterdir_images(folder, mod.IMAGE_EXTENSIONS)

