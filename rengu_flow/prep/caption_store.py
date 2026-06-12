"""Caption storage for dataset preparation.

Reads/writes the two caption layouts the trainer understands (rengu_flow/data/dataset.py):
per-image sidecar text files (one caption variant per line, customizable extension) and a
composite ``captions.json`` (``{image_filename: [captions]}``). All mutations stay in memory
until ``save()``; writes are atomic, and ``snapshot()``/``restore_snapshot()`` give a full
caption backup under ``<folder>/.rengu_prep/`` so a bad bulk edit is always recoverable.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CAPTIONS_JSON_FILE = "captions.json"
PREP_DIR_NAME = ".rengu_prep"
BACKUPS_DIR_NAME = "backups"
QUARANTINE_DIR_NAME = "quarantine"
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


@dataclass
class CaptionSet:
    """In-memory captions for one dataset folder. Nothing touches disk until save()."""

    folder: Path
    fmt: str = FORMAT_SIDECAR
    ext: str = ".txt"
    images: dict[str, Path] = field(default_factory=dict)
    captions: dict[str, list[str]] = field(default_factory=dict)
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

    def append_line(self, key: str, text: str) -> None:
        lines = self.get_lines(key)
        lines.append(text)
        self.set_lines(key, lines)

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
        backups_root = self.folder / PREP_DIR_NAME / BACKUPS_DIR_NAME
        backup_dir = backups_root / _utc_stamp()
        suffix = 0
        while backup_dir.exists():
            suffix += 1
            backup_dir = backups_root / f"{backup_dir.name.split('-')[0]}-{suffix}"
        backup_dir.mkdir(parents=True)

        files: list[str] = []
        for name in sorted(
            p.name for p in self.folder.glob(f"*{self.ext}") if p.is_file()
        ):
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
        """Move images (and their sidecars) out of the dataset — never delete."""
        qdir = self.folder / PREP_DIR_NAME / QUARANTINE_DIR_NAME / _utc_stamp()
        qdir.mkdir(parents=True, exist_ok=True)
        entries = {}
        for key in keys:
            image = self.images.get(key)
            if image is None:
                continue
            entries[key] = {"captions": self.captions.get(key, [])}
            shutil.move(str(image), qdir / image.name)
            sidecar = image.with_suffix(self.ext)
            if sidecar.is_file():
                shutil.move(str(sidecar), qdir / sidecar.name)
            self.images.pop(key, None)
            self.captions.pop(key, None)
            self._loaded.pop(key, None)
        _atomic_write_text(
            qdir / MANIFEST_FILE,
            json.dumps(
                {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "format": self.fmt,
                    "ext": self.ext,
                    "entries": entries,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
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


class CaptionStore:
    """Entry points for opening caption sets and managing backups/quarantine."""

    @staticmethod
    def open(
        folder: str | Path, fmt: str = FORMAT_SIDECAR, ext: str = ".txt"
    ) -> CaptionSet:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        if fmt not in (FORMAT_SIDECAR, FORMAT_JSON):
            raise ValueError(f"Unknown caption format: {fmt}")
        if not ext.startswith("."):
            ext = f".{ext}"

        images = {
            p.name: p
            for p in sorted(folder.glob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
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
                    captions[key] = read_caption_lines(raw) or []
                else:
                    captions[key] = [c.strip() for c in raw if str(c).strip()]
        else:
            for key, image in images.items():
                sidecar = image.with_suffix(ext)
                if sidecar.is_file():
                    captions[key] = read_caption_lines(
                        sidecar.read_text(encoding="utf-8")
                    )
                else:
                    captions[key] = []

        return CaptionSet(
            folder=folder,
            fmt=fmt,
            ext=ext,
            images=images,
            captions=captions,
            _loaded={key: tuple(lines) for key, lines in captions.items()},
        )

    @staticmethod
    def list_backups(folder: str | Path) -> list[dict]:
        backups_root = Path(folder) / PREP_DIR_NAME / BACKUPS_DIR_NAME
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
        backup_dir = folder / PREP_DIR_NAME / BACKUPS_DIR_NAME / backup_name
        manifest_file = backup_dir / MANIFEST_FILE
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Backup not found: {backup_dir}")
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
        files = set(manifest.get("files", []))
        ext = manifest.get("ext", ".txt")

        restored: list[str] = []
        for current in folder.glob(f"*{ext}"):
            if current.is_file() and current.name not in files:
                current.unlink()
                restored.append(current.name)
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
        qroot = Path(folder) / PREP_DIR_NAME / QUARANTINE_DIR_NAME
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
        qdir = folder / PREP_DIR_NAME / QUARANTINE_DIR_NAME / batch_name
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
