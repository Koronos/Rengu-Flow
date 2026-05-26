"""Open on-disk cache in v1 (pickle shards) or v2 (mmap tensor stacks)."""

from __future__ import annotations

from pathlib import Path

from renga_flow.utils.cache import Cache
from renga_flow.utils.cache_v2 import FORMAT_VERSION, MANIFEST_NAME, CacheV2

CACHE_FORMAT_V1 = "v1"
CACHE_FORMAT_V2 = "v2"


def detect_cache_format(path: Path) -> str | None:
    path = Path(path)
    manifest = path / MANIFEST_NAME
    if manifest.is_file():
        import json

        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("format_version") == FORMAT_VERSION:
            return CACHE_FORMAT_V2
    if (path / "metadata.db").is_file():
        return CACHE_FORMAT_V1
    return None


def open_disk_cache(
    path: str | Path,
    fingerprint: str,
    *,
    cache_format: str = CACHE_FORMAT_V2,
    shard_size_gb: float = 10.0,
):
    """Return a cache instance (v1 ``Cache`` or v2 ``CacheV2``) for *path*."""
    path = Path(path)
    detected = detect_cache_format(path)
    if detected is not None:
        cache_format = detected
    elif cache_format not in (CACHE_FORMAT_V1, CACHE_FORMAT_V2):
        raise ValueError(f"Unknown cache_format: {cache_format}")

    if cache_format == CACHE_FORMAT_V2:
        return CacheV2(path, fingerprint)
    return Cache(path, fingerprint, shard_size_gb=shard_size_gb)
