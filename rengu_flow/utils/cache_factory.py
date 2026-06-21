"""Open on-disk cache (v2 mmap tensor stacks only)."""

from __future__ import annotations

from pathlib import Path

from rengu_flow.utils.cache_v2 import FORMAT_VERSION, MANIFEST_NAME, CacheV2

CACHE_FORMAT_V2 = "v2"


def detect_cache_format(path: Path) -> str | None:
    """Return ``v2`` if a v2 manifest exists; ``None`` if empty; raise if legacy v1."""
    path = Path(path)
    manifest = path / MANIFEST_NAME
    if manifest.is_file():
        import json

        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("format_version") == FORMAT_VERSION:
            return CACHE_FORMAT_V2
    if (path / "metadata.db").is_file():
        raise ValueError(
            f"Legacy cache v1 at {path}; regenerate cache (v2 only). "
            "Delete the old cache directory or run with --regenerate_cache."
        )
    return None


def open_disk_cache(
    path: str | Path,
    fingerprint: str,
):
    """Return a v2 ``CacheV2`` instance for *path*."""
    path = Path(path)
    detect_cache_format(path)  # raises on legacy v1; returns v2 or None
    return CacheV2(path, fingerprint)
