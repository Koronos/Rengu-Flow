"""Filesystem path helpers."""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Collection
from pathlib import Path

# pathlib compares, sorts and globs case-insensitively on Windows (WindowsPath) and
# case-sensitively elsewhere; the scandir helpers below reproduce that so their results are
# identical to the ``sorted(folder.glob("*"))`` / ``folder.glob("*.txt")`` they replace.
_CASE_INSENSITIVE = os.name == "nt"


def path_is_under(path: str | Path, root: str | Path) -> bool:
    """True when ``path`` is ``root`` or a file/dir under ``root`` (resolved)."""
    try:
        p = Path(path).resolve()
        r = Path(root).resolve()
        return p == r or p.is_relative_to(r)
    except (ValueError, OSError):
        return False


def name_suffix(name: str) -> str:
    """``PurePath(name).suffix`` without building a path (``".png"`` has none, ``"a.b.png"`` is
    ``".png"``)."""
    i = name.rfind(".")
    return name[i:] if 0 < i < len(name) - 1 else ""


def name_stem(name: str) -> str:
    """``PurePath(name).stem`` without building a path."""
    suffix = name_suffix(name)
    return name[: -len(suffix)] if suffix else name


def name_key(name: str) -> str:
    """Case key pathlib uses for a file name here: the lowercased name on Windows."""
    return name.lower() if _CASE_INSENSITIVE else name


def dir_files(folder: str | Path, suffixes: Collection[str] | None = None) -> list[str]:
    """Names of the files directly in ``folder``, in ``sorted(Path(folder).glob("*"))`` order.

    Same set as ``[p for p in sorted(folder.glob("*")) if p.is_file()]``: hidden files included,
    symlinks followed (a link to a file counts, a broken link or a link to a folder does not),
    subfolders skipped. With ``suffixes`` (lowercase, e.g. ``{".png"}``) only names whose
    lowercased :func:`name_suffix` is in it are kept, like ``p.suffix.lower() in suffixes``.

    One ``os.scandir`` pass: ``DirEntry.is_file()`` answers from the directory listing, where
    ``Path.is_file()`` costs a ``stat`` per file (~0.4 ms each on NTFS — minutes on a
    million-image folder).
    """
    with os.scandir(folder) as it:
        names = [
            e.name
            for e in it
            if e.is_file()
            and (suffixes is None or name_suffix(e.name).lower() in suffixes)
        ]
    names.sort(key=name_key)
    return names


def glob_names(names: list[str], pattern: str) -> list[str]:
    """The ``names`` a ``Path.glob(pattern)`` on their folder would match (a single-component
    pattern, with pathlib's platform case rule), keeping ``names`` order."""
    flags = re.IGNORECASE if _CASE_INSENSITIVE else 0
    match = re.compile(fnmatch.translate(pattern), flags).fullmatch
    return [n for n in names if match(n)]
