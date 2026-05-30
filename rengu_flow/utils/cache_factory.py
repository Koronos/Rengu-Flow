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
    *,
    shard_size_gb: float = 10.0,
    cache_format: str | None = None,
):
    """Return a v2 ``CacheV2`` instance for *path*."""
    del shard_size_gb  # v1-only parameter; kept for call-site compatibility
    del cache_format  # v1/v2 selector; only v2 exists now, kept for call-site compatibility
    path = Path(path)
    detected = detect_cache_format(path)
    if detected is not None and detected != CACHE_FORMAT_V2:
        raise ValueError(f"Unsupported cache format at {path}")
    return CacheV2(path, fingerprint)
