"""Caption storage for dataset preparation.

Reads/writes the two caption layouts the trainer understands (rengu_flow/data/dataset.py):
per-image sidecar text files (one caption variant per line, customizable extension) and a
composite ``captions.json`` (``{image_filename: [captions]}``). All mutations stay in memory
until ``save()``; writes are atomic, and ``snapshot()``/``restore_snapshot()`` give a full
caption backup under the managed app data dir (see ``prep_storage_dir``) so a bad bulk
edit is always recoverable.

An edit dataset (targets + a ``control_path`` folder of condition images) opens with
``CaptionStore.open(..., control_path=...)``: each target is paired with its controls by the
trainer's own rules (``rengu_flow.data.control``), so prep and training always agree on the
pairs. Targets that do not pair are recorded in ``CaptionSet.unpaired``, never fatal.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rengu_flow.prep.storage import prep_storage_dir
from rengu_flow.utils.paths import dir_files, glob_names, name_key, name_stem, name_suffix

CAPTIONS_JSON_FILE = "captions.json"
BACKUPS_DIR_NAME = "backups"
QUARANTINE_DIR_NAME = "quarantine"
CONTROLS_DIR_NAME = "controls"  # quarantined control images, inside a quarantine batch
MANIFEST_FILE = "manifest.json"

FORMAT_SIDECAR = "sidecar"
FORMAT_JSON = "json"

# Prep stages feed PIL directly, so unlike the training scanner (which takes anything that
# is not a known sidecar suffix) discovery is restricted to decodable image extensions.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".heic",
    ".heif",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_caption_lines(text: str) -> list[str]:
    """Same semantics as the trainer's _read_captions_from_txt_per_line: empty lines skipped."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_caption_lines_positional(text: str) -> list[str]:
    """A sidecar's lines **by position**: leading/interior empty lines are kept, trailing ones go.

    Prep addresses a caption by line number (tags on line 1, a caption on line 3...), so a blank
    line is a placeholder that must stay where it is. The trainer still skips blank lines
    (``read_caption_lines``), so a padded file trains on exactly its non-empty lines.
    """
    return _trim_trailing([line.strip() for line in text.splitlines()])


def _trim_trailing(lines: list[str]) -> list[str]:
    out = list(lines)
    while out and not out[-1].strip():
        out.pop()
    return out


# How a prep step writes into its target line. ``skip`` leaves lines that already have content
# alone (the historical default), ``replace`` overwrites, ``append`` adds to the existing text.
WRITE_SKIP = "skip"
WRITE_REPLACE = "replace"
WRITE_APPEND = "append"
WRITE_MODES = (WRITE_SKIP, WRITE_REPLACE, WRITE_APPEND)


def effective_write_mode(write_mode: str | None, overwrite: bool = False) -> str:
    """The mode a stage runs with. ``write_mode`` unset (``""``) falls back to the legacy
    ``overwrite`` flag (True == replace), so every config saved before modes existed behaves
    identically."""
    mode = str(write_mode or "").strip().lower()
    if not mode:
        return WRITE_REPLACE if overwrite else WRITE_SKIP
    if mode not in WRITE_MODES:
        raise ValueError(f"Unknown write_mode {write_mode!r}; expected one of {WRITE_MODES}")
    return mode


def merge_tags(existing: str, new: str) -> str:
    """``existing, new`` as one tag line, without repeating a tag (case-insensitive)."""
    have = [t.strip() for t in existing.split(",") if t.strip()]
    seen = {t.lower() for t in have}
    for tag in (t.strip() for t in new.split(",")):
        if tag and tag.lower() not in seen:
            have.append(tag)
            seen.add(tag.lower())
    return ", ".join(have)


@dataclass
class CaptionSet:
    """In-memory captions for one dataset folder. Nothing touches disk until save()."""

    folder: Path
    fmt: str = FORMAT_SIDECAR
    ext: str = ".txt"
    images: dict[str, Path] = field(default_factory=dict)
    captions: dict[str, list[str]] = field(default_factory=dict)
    # Edit datasets only (open(..., control_path=...)): the control folder, each paired
    # target's control images in order, and why each unpaired target did not pair.
    control_path: Path | None = None
    controls: dict[str, list[Path]] = field(default_factory=dict)
    unpaired: dict[str, str] = field(default_factory=dict)
    _loaded: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    # -- accessors ---------------------------------------------------------------

    def keys(self) -> list[str]:
        return list(self.images.keys())

    def get_lines(self, key: str) -> list[str]:
        return list(self.captions.get(key, []))

    def set_lines(self, key: str, lines: list[str]) -> None:
        if key not in self.images:
            raise KeyError(f"Unknown image: {key}")
        self.captions[key] = [line.strip() for line in lines if line.strip()]

    def set_line(self, key: str, index: int, text: str) -> None:
        lines = self.get_lines(key)
        while len(lines) <= index:
            lines.append("")
        lines[index] = text
        self.set_lines(key, lines)

    def line_has_content(self, key: str, index: int) -> bool:
        lines = self.captions.get(key, [])
        return index < len(lines) and bool(lines[index].strip())

    def write_line(
        self,
        key: str,
        index: int,
        text: str,
        mode: str = WRITE_REPLACE,
        *,
        sep: str = " ",
        tags: bool = False,
    ) -> None:
        """Write *text* on line *index* (0-based), padding shorter captions with empty lines.

        ``append`` adds to the line's existing content (joined with *sep*; with ``tags=True`` as a
        de-duplicated tag list); on an empty line it is the same as ``replace``. ``skip`` is a
        decision for the caller (:meth:`line_has_content`) - here it writes like ``replace``.
        """
        if key not in self.images:
            raise KeyError(f"Unknown image: {key}")
        lines = self.get_lines(key)
        while len(lines) <= index:
            lines.append("")
        text = text.strip()
        current = lines[index].strip()
        if mode == WRITE_APPEND and current:
            if tags:
                text = merge_tags(current, text)
            elif text and (text == current or current.endswith(f"{sep}{text}")):
                # Exact-duplicate guard: a re-run (there is no resume tracking for appended
                # captions) must not stack the same text on the line again.
                text = current
            elif text:
                text = f"{current}{sep}{text}"
            else:
                text = current
        lines[index] = text
        self.captions[key] = _trim_trailing(lines)

    def get_tags(self, key: str, line_index: int = 0) -> list[str]:
        lines = self.captions.get(key, [])
        if line_index >= len(lines):
            return []
        return [t.strip() for t in lines[line_index].split(",") if t.strip()]

    def caption_path(self, key: str) -> Path:
        return self.images[key].with_suffix(self.ext)

    def dirty_keys(self) -> list[str]:
        return [
            key
            for key in self.images
            if tuple(self.captions.get(key, [])) != self._loaded.get(key, ())
        ]

    # -- persistence -------------------------------------------------------------

    def save(self) -> list[str]:
        """Write changed captions to disk atomically. Returns the affected file names."""
        dirty = self.dirty_keys()
        if not dirty:
            return []
        written: list[str] = []
        if self.fmt == FORMAT_JSON:
            payload = {
                key: (lines if lines else [""])
                for key, lines in sorted(self.captions.items())
            }
            _atomic_write_text(
                self.folder / CAPTIONS_JSON_FILE,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            written.append(CAPTIONS_JSON_FILE)
        else:
            for key in dirty:
                path = self.caption_path(key)
                lines = self.captions.get(key, [])
                if lines:
                    _atomic_write_text(path, "\n".join(lines) + "\n")
                    written.append(path.name)
                elif path.exists():
                    path.unlink()
                    written.append(path.name)
        for key in dirty:
            self._loaded[key] = tuple(self.captions.get(key, []))
        return written

    # -- backup / restore ----------------------------------------------------------

    def snapshot(self) -> Path:
        """Copy every caption file (sidecars and captions.json) to a timestamped backup."""
        backups_root = prep_storage_dir(self.folder) / BACKUPS_DIR_NAME
        backup_dir = backups_root / _utc_stamp()
        suffix = 0
        while backup_dir.exists():
            suffix += 1
            backup_dir = backups_root / f"{backup_dir.name.split('-')[0]}-{suffix}"
        backup_dir.mkdir(parents=True)

        files: list[str] = []
        for name in sorted(glob_names(dir_files(self.folder), f"*{self.ext}")):
            shutil.copy2(self.folder / name, backup_dir / name)
            files.append(name)
        captions_json = self.folder / CAPTIONS_JSON_FILE
        if captions_json.is_file():
            shutil.copy2(captions_json, backup_dir / CAPTIONS_JSON_FILE)
            files.append(CAPTIONS_JSON_FILE)

        manifest = {
            "created": datetime.now(timezone.utc).isoformat(),
            "format": self.fmt,
            "ext": self.ext,
            "files": files,
        }
        _atomic_write_text(
            backup_dir / MANIFEST_FILE, json.dumps(manifest, indent=2) + "\n"
        )
        return backup_dir

    # -- quarantine ----------------------------------------------------------------

    def quarantine(self, keys: list[str]) -> Path:
        """Move images (and their sidecars) out of the dataset — never delete.

        In an edit set (opened with ``control_path``) a target's paired control images move
        with it, into the batch's ``controls/`` subfolder, so the pair stays whole and
        ``restore_quarantine`` puts both back. A control that a remaining target also pairs
        with (``a_1.png`` is both the control of ``a_1`` and control #1 of ``a``) stays in
        place: moving it would silently unpair that other target.
        """
        qdir = prep_storage_dir(self.folder) / QUARANTINE_DIR_NAME / _utc_stamp()
        qdir.mkdir(parents=True, exist_ok=True)
        removed = {key for key in keys if key in self.images}
        kept_controls = {
            p for key, paths in self.controls.items() if key not in removed for p in paths
        }
        entries = {}
        for key in keys:
            image = self.images.get(key)
            if image is None:
                continue
            entry: dict = {"captions": self.captions.get(key, [])}
            shutil.move(str(image), qdir / image.name)
            sidecar = image.with_suffix(self.ext)
            if sidecar.is_file():
                shutil.move(str(sidecar), qdir / sidecar.name)
            moved = []
            for control in self.controls.get(key, []):
                if control in kept_controls or not control.is_file():
                    continue
                (qdir / CONTROLS_DIR_NAME).mkdir(exist_ok=True)
                shutil.move(str(control), qdir / CONTROLS_DIR_NAME / control.name)
                moved.append(control.name)
            if moved:
                entry["controls"] = moved
            entries[key] = entry
            self.images.pop(key, None)
            self.captions.pop(key, None)
            self._loaded.pop(key, None)
            self.controls.pop(key, None)
            self.unpaired.pop(key, None)
        manifest = {
            "created": datetime.now(timezone.utc).isoformat(),
            "format": self.fmt,
            "ext": self.ext,
            "entries": entries,
        }
        if self.control_path is not None:
            manifest["control_path"] = str(self.control_path)
        _atomic_write_text(
            qdir / MANIFEST_FILE,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        if self.fmt == FORMAT_JSON:
            # Rewrite captions.json without the removed keys even if nothing else changed.
            payload = {
                key: (lines if lines else [""])
                for key, lines in sorted(self.captions.items())
            }
            _atomic_write_text(
                self.folder / CAPTIONS_JSON_FILE,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
        return qdir


def _pair_controls(
    images: dict[str, Path], control_path: Path
) -> tuple[dict[str, list[Path]], dict[str, str]]:
    """Pair each image with its control images by the trainer's rules (``stem`` or
    ``stem_0..N``). Returns ``(controls, unpaired)``: a pairing error is recorded, not raised."""
    from rengu_flow.data.control import (
        ControlPairingError,
        index_control_dir,
        pair_control_files,
    )

    index = index_control_dir(control_path)  # one directory scan for the whole set
    controls: dict[str, list[Path]] = {}
    unpaired: dict[str, str] = {}
    for key, image in images.items():
        try:
            controls[key] = [Path(p) for p in pair_control_files(image.stem, index, target=key)]
        except ControlPairingError as exc:
            unpaired[key] = str(exc)
    return controls, unpaired


class CaptionStore:
    """Entry points for opening caption sets and managing backups/quarantine."""

    @staticmethod
    def open(
        folder: str | Path,
        fmt: str = FORMAT_SIDECAR,
        ext: str = ".txt",
        control_path: str | Path | None = None,
        positional: bool = False,
    ) -> CaptionSet:
        """Load a folder's captions. By default blank lines are skipped, exactly as the trainer reads
        them; ``positional=True`` keeps leading/interior blank lines where they are (trailing ones
        go), which is what a prep step that writes "line 3" needs - padded lines must round-trip.
        With ``control_path`` (an edit dataset) each image is also
        paired with its control images — see :attr:`CaptionSet.controls` and ``unpaired``."""
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        if fmt not in (FORMAT_SIDECAR, FORMAT_JSON):
            raise ValueError(f"Unknown caption format: {fmt}")
        if not ext.startswith("."):
            ext = f".{ext}"

        # One scandir pass lists images and sidecars alike: no stat per file (see dir_files).
        names = dir_files(folder)
        images = {
            name: folder / name
            for name in names
            if name_suffix(name).lower() in IMAGE_EXTENSIONS
        }
        captions: dict[str, list[str]] = {}
        if fmt == FORMAT_JSON:
            captions_json = folder / CAPTIONS_JSON_FILE
            data = {}
            if captions_json.is_file():
                with open(captions_json, encoding="utf-8") as f:
                    data = json.load(f)
            for key in images:
                raw = data.get(key)
                if raw is None:
                    captions[key] = []
                elif isinstance(raw, str):
                    captions[key] = (
                        read_caption_lines_positional(raw) if positional else read_caption_lines(raw)
                    )
                else:
                    cleaned = [str(c).strip() for c in raw]
                    captions[key] = (
                        _trim_trailing(cleaned) if positional else [c for c in cleaned if c]
                    )
        else:
            # Same lookup as ``image.with_suffix(ext).is_file()``, answered from the listing.
            present = {name_key(name) for name in names}
            for key, image in images.items():
                sidecar = f"{name_stem(key)}{ext}"
                if name_key(sidecar) in present:
                    text = (folder / sidecar).read_text(encoding="utf-8")
                    captions[key] = (
                        read_caption_lines_positional(text) if positional else read_caption_lines(text)
                    )
                else:
                    captions[key] = []

        controls: dict[str, list[Path]] = {}
        unpaired: dict[str, str] = {}
        if control_path is not None:
            control_path = Path(control_path)
            if not control_path.is_dir():
                raise FileNotFoundError(f"Control folder not found: {control_path}")
            controls, unpaired = _pair_controls(images, control_path)

        return CaptionSet(
            folder=folder,
            fmt=fmt,
            ext=ext,
            images=images,
            captions=captions,
            control_path=control_path,
            controls=controls,
            unpaired=unpaired,
            _loaded={key: tuple(lines) for key, lines in captions.items()},
        )

    @staticmethod
    def list_backups(folder: str | Path) -> list[dict]:
        backups_root = prep_storage_dir(folder) / BACKUPS_DIR_NAME
        results = []
        if not backups_root.is_dir():
            return results
        for backup_dir in sorted(backups_root.iterdir(), reverse=True):
            manifest_file = backup_dir / MANIFEST_FILE
            if not manifest_file.is_file():
                continue
            with open(manifest_file, encoding="utf-8") as f:
                manifest = json.load(f)
            results.append(
                {
                    "name": backup_dir.name,
                    "created": manifest.get("created"),
                    "format": manifest.get("format"),
                    "ext": manifest.get("ext"),
                    "file_count": len(manifest.get("files", [])),
                }
            )
        return results

    @staticmethod
    def restore_snapshot(folder: str | Path, backup_name: str) -> list[str]:
        """Restore caption files exactly as snapshotted (extra caption files are removed)."""
        folder = Path(folder)
        backup_dir = prep_storage_dir(folder) / BACKUPS_DIR_NAME / backup_name
        manifest_file = backup_dir / MANIFEST_FILE
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Backup not found: {backup_dir}")
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
        files = set(manifest.get("files", []))
        ext = manifest.get("ext", ".txt")

        restored: list[str] = []
        for name in glob_names(dir_files(folder), f"*{ext}"):
            if name not in files:
                (folder / name).unlink()
                restored.append(name)
        captions_json = folder / CAPTIONS_JSON_FILE
        if CAPTIONS_JSON_FILE not in files and captions_json.is_file():
            captions_json.unlink()
            restored.append(CAPTIONS_JSON_FILE)
        for name in files:
            shutil.copy2(backup_dir / name, folder / name)
            restored.append(name)
        return sorted(set(restored))

    @staticmethod
    def list_quarantine(folder: str | Path) -> list[dict]:
        qroot = prep_storage_dir(folder) / QUARANTINE_DIR_NAME
        results = []
        if not qroot.is_dir():
            return results
        for qdir in sorted(qroot.iterdir(), reverse=True):
            manifest_file = qdir / MANIFEST_FILE
            if not manifest_file.is_file():
                continue
            with open(manifest_file, encoding="utf-8") as f:
                manifest = json.load(f)
            results.append(
                {
                    "name": qdir.name,
                    "created": manifest.get("created"),
                    "images": sorted(manifest.get("entries", {})),
                }
            )
        return results

    @staticmethod
    def restore_quarantine(folder: str | Path, batch_name: str) -> list[str]:
        """Move a quarantine batch back into the dataset folder."""
        folder = Path(folder)
        qdir = prep_storage_dir(folder) / QUARANTINE_DIR_NAME / batch_name
        manifest_file = qdir / MANIFEST_FILE
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Quarantine batch not found: {qdir}")
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
        restored = []
        for key, entry in manifest.get("entries", {}).items():
            image = qdir / key
            if image.is_file():
                shutil.move(str(image), folder / key)
                restored.append(key)
            sidecar = (qdir / key).with_suffix(manifest.get("ext", ".txt"))
            if sidecar.is_file():
                shutil.move(str(sidecar), folder / sidecar.name)
            if entry.get("controls") and manifest.get("control_path"):
                control_dir = Path(manifest["control_path"])
                control_dir.mkdir(parents=True, exist_ok=True)
                for name in entry["controls"]:
                    src = qdir / CONTROLS_DIR_NAME / name
                    if src.is_file():
                        shutil.move(str(src), control_dir / name)
            if manifest.get("format") == FORMAT_JSON and entry.get("captions"):
                captions_json = folder / CAPTIONS_JSON_FILE
                data = {}
                if captions_json.is_file():
                    with open(captions_json, encoding="utf-8") as f:
                        data = json.load(f)
                data[key] = entry["captions"]
                _atomic_write_text(
                    captions_json,
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                )
        shutil.rmtree(qdir)
        return restored


CONVERSIONS_DIR_NAME = "conversions"

# What the trainer never treats as a media file (rengu_flow/data/dataset.py, file scan).
_NON_MEDIA_SUFFIXES = {".txt", ".npz", ".json", ".bak", ".tar", ".parquet"}


# Only these can carry a sidecar caption for the layout guard (a notes.md next to notes.txt is not one).
_CAPTIONED_MEDIA = IMAGE_EXTENSIONS | {".mp4", ".avi", ".mov", ".webm", ".mkv"}


class LayoutMismatchError(ValueError):
    """A stage would work in a caption layout the folder is not actually in."""


def _norm_ext(ext: str) -> str:
    ext = str(ext or ".txt").strip() or ".txt"
    return ext if ext.startswith(".") else f".{ext}"


def _media_files(folder: Path, *sidecar_exts: str) -> dict[str, Path]:
    """Every file the trainer would train on as a standalone item: images **and videos** (any file
    that is not a sidecar/side file). ``.tar`` / ``.parquet`` carry their captions inline and are
    not addressable by a sidecar, so they are not listed."""
    skip = _NON_MEDIA_SUFFIXES | {e.lower() for e in sidecar_exts}
    out: dict[str, Path] = {}
    for name in dir_files(folder):
        if name_suffix(name).lower() in skip:
            continue
        if (folder / name).is_file():
            out[name] = folder / name
    return out


def _read_json_captions(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name} is not valid JSON ({exc}); nothing was converted") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must hold an object of caption lists; nothing was converted")
    return raw


def _json_lines(value) -> list[str]:
    """One captions.json entry as positional lines (blank placeholders kept, trailing dropped)."""
    if isinstance(value, str):
        return read_caption_lines_positional(value)
    if isinstance(value, list):
        return _trim_trailing([str(c).strip() for c in value])
    return []


def _backup_files(folder: Path, files: list[Path], label: str) -> Path | None:
    """Copy *files* to ``<prep storage>/conversions/<stamp>-<label>/`` before they are removed."""
    files = [f for f in files if f.is_file()]
    if not files:
        return None
    root = prep_storage_dir(folder) / CONVERSIONS_DIR_NAME
    target = root / f"{_utc_stamp()}-{label}"
    n = 0
    while target.exists():
        n += 1
        target = root / f"{_utc_stamp()}-{label}-{n}"
    target.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, target / f.name)
    _atomic_write_text(
        target / MANIFEST_FILE,
        json.dumps({"folder": str(folder), "files": [f.name for f in files]}, indent=2) + "\n",
    )
    return target


def convert_captions(
    folder: str | Path,
    from_fmt: str,
    from_ext: str,
    to_fmt: str,
    to_ext: str,
) -> dict:
    """Move a folder's captions from one layout to another, then remove the old files.

    Entries are enumerated the way the trainer does: **every** media file of the folder (videos
    too), not just images. Lines are carried by position (a padded empty line stays put).

    * ``json -> sidecar`` is **refused before anything is written** when ``captions.json`` has keys
      that cannot become a sidecar (a tar member, a file that is not in the folder): they would be
      lost. Otherwise one sidecar per captioned file is written (overwriting an existing one: the
      json was the layout in force), verified, ``captions.json`` is backed up and removed.
    * ``sidecar -> json`` **merges into the existing** ``captions.json`` (keys not covered by a
      sidecar are kept untouched, including ones this folder no longer has), writes it atomically,
      verifies it, backs the sidecars up and removes them.
    * ``sidecar -> sidecar`` with another extension renames.

    Every file that is removed **or overwritten** (old layout files, and destination files the
    conversion replaces) first lands in ``<prep storage>/conversions/<stamp>-<from>/`` (see
    :func:`~rengu_flow.prep.storage.prep_storage_dir`); the report's ``backup`` names it. New files
    are always written and verified **before** any old file is touched. Running it again once
    converted does nothing. Returns ``{"converted", "removed", "from", "to", "backup"}``.
    """
    folder = Path(folder)
    from_ext, to_ext = _norm_ext(from_ext), _norm_ext(to_ext)
    for fmt in (from_fmt, to_fmt):
        if fmt not in (FORMAT_SIDECAR, FORMAT_JSON):
            raise ValueError(f"Unknown caption format: {fmt}")
    report = {
        "converted": 0,
        "removed": 0,
        "from": f"{from_fmt}{from_ext if from_fmt == FORMAT_SIDECAR else ''}",
        "to": f"{to_fmt}{to_ext if to_fmt == FORMAT_SIDECAR else ''}",
        "backup": None,
    }
    if from_fmt == to_fmt and (from_fmt == FORMAT_JSON or from_ext == to_ext):
        return report
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    media = _media_files(folder, from_ext, to_ext)
    json_path = folder / CAPTIONS_JSON_FILE

    # ---- gather what the source layout holds
    carried: dict[str, list[str]] = {}
    old_files: list[Path] = []
    if from_fmt == FORMAT_JSON:
        if not json_path.is_file():
            return report
        raw_json = _read_json_captions(json_path)
        stray = sorted(k for k in raw_json if k not in media)
        if stray:
            shown = ", ".join(repr(k) for k in stray[:8]) + (" ..." if len(stray) > 8 else "")
            raise ValueError(
                f"captions.json has {len(stray)} entr{'y' if len(stray) == 1 else 'ies'} that cannot "
                f"become sidecar files ({shown}): tar members or files that are not in this folder. "
                "Converting would lose them, so nothing was changed; keep this folder on "
                "captions.json or remove those entries first."
            )
        for key, value in raw_json.items():
            lines = _json_lines(value)
            if lines:
                carried[key] = lines
        old_files = [json_path]
    else:
        for name, path in media.items():
            sidecar = path.with_suffix(from_ext)
            if sidecar.is_file():
                old_files.append(sidecar)
                lines = read_caption_lines_positional(sidecar.read_text(encoding="utf-8-sig"))
                if lines:
                    carried[name] = lines

    # ---- back up everything that will be removed OR overwritten, before touching anything
    overwritten: list[Path] = []
    if carried:
        if to_fmt == FORMAT_JSON:
            overwritten = [json_path] if json_path.is_file() else []
        else:
            overwritten = [
                media[name].with_suffix(to_ext)
                for name in carried
                if media[name].with_suffix(to_ext).is_file()
            ]
    backup = _backup_files(folder, list(dict.fromkeys([*old_files, *overwritten])), from_fmt)
    report["backup"] = str(backup) if backup else None

    # ---- write the destination (and verify) before anything old is removed
    if carried:
        if to_fmt == FORMAT_JSON:
            merged = _read_json_captions(json_path) if json_path.is_file() else {}
            for name, lines in carried.items():
                merged[name] = lines
            _atomic_write_text(json_path, json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
            check = _read_json_captions(json_path)
            for name, lines in carried.items():
                if _json_lines(check.get(name)) != lines:
                    raise RuntimeError(f"Caption conversion could not verify {name}; old files kept")
        else:
            for name, lines in carried.items():
                _atomic_write_text(media[name].with_suffix(to_ext), "\n".join(lines) + "\n")
            for name, lines in carried.items():
                got = read_caption_lines_positional(
                    media[name].with_suffix(to_ext).read_text(encoding="utf-8")
                )
                if got != lines:
                    raise RuntimeError(f"Caption conversion could not verify {name}; old files kept")
        report["converted"] = len(carried)

    for path in old_files:
        if path.is_file():
            path.unlink()
            report["removed"] += 1
    return report


def check_layout(folder: str | Path, fmt: str, ext: str) -> None:
    """Refuse to run a caption stage in a layout the folder is not in.

    The trainer reads ``captions.json`` whenever the file exists and then ignores every sidecar. A
    sidecar-mode stage in a folder holding ``captions.json`` would write captions training never
    reads; a json-mode stage in a folder with sidecars would leave them stale. Either is a bug the
    user cannot see, so it fails clearly, naming the layout the folder is really in.
    """
    folder = Path(folder)
    ext = _norm_ext(ext)
    json_path = folder / CAPTIONS_JSON_FILE
    if fmt == FORMAT_SIDECAR and json_path.is_file():
        raise LayoutMismatchError(
            f"This step works in sidecar mode ({ext}) but {folder} holds a captions.json, so that "
            "folder is in the captions.json layout and training would ignore any sidecar written "
            "now. Set the step's caption format to captions.json (or convert the folder), or "
            "remove captions.json."
        )
    # Leftover sidecars next to an existing captions.json are ignored by the trainer and harmless.
    if fmt == FORMAT_JSON and not json_path.is_file():
        stale = [
            name
            for name, path in _media_files(folder, ext).items()
            if name_suffix(name).lower() in _CAPTIONED_MEDIA and path.with_suffix(ext).is_file()
        ]
        if stale:
            raise LayoutMismatchError(
                f"This step works in captions.json mode but {len(stale)} file(s) in {folder} "
                f"(e.g. {stale[0]}) still have {ext} sidecars, so the folder is in the sidecar "
                "layout: those captions would be left behind and ignored. Set the step's caption "
                "format to sidecar files (or convert the folder first)."
            )
