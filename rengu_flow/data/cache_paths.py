"""Resolve dataset cache directories under cache_root (v2 only)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _stable_id(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def caption_cache_key(caption: str | list[str]) -> str:
    """Full SHA-256 hex for caption dedup during text-embedding cache.

    Captions may be a single string or a list of strings (multi-caption images);
    a NUL separator keeps ["a", "b"] distinct from ["ab"].
    """
    if isinstance(caption, (list, tuple)):
        payload = "\x00".join(str(c) for c in caption)
    else:
        payload = str(caption)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_cache_root() -> Path:
    """Untracked cache root at rengu-flow installation (repo) root."""
    return Path(__file__).resolve().parents[2] / "cache"


def resolve_cache_root(
    training_config: dict,
    *,
    dataset_config: dict | None = None,
) -> Path:
    """Return resolved cache root from training config, with legacy dataset fallback."""
    raw = training_config.get("cache_root")
    legacy = dataset_config.get("cache_root") if dataset_config else None
    if legacy is not None and (raw is None or (isinstance(raw, str) and not str(raw).strip())):
        logger.warning(
            "cache_root in dataset TOML is deprecated; set cache_root in the training config. "
            "Using dataset TOML value for this run."
        )
        raw = legacy
    elif legacy is not None and raw is not None:
        logger.warning(
            "cache_root in dataset TOML is ignored; set cache_root in the training config "
            "(Training loop section)."
        )
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        root = default_cache_root()
    else:
        root = Path(str(raw)).expanduser()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_cache_id(dataset_config: dict) -> str:
    """Namespace id for a dataset config's caches, keyed on the data it points AT.

    Keyed on the sorted [[directory]] paths — the dataset's stable identity — never on
    where the TOML file happens to live: the UI stages a copy of the dataset TOML into a
    per-job folder, so a path-keyed id changed every run and silently regenerated every
    cache. Settings changes (resolutions, captions, augmentation) don't need to move the
    namespace: bucket dirs and content fingerprints already invalidate exactly what they
    affect.
    """
    dirs = dataset_config.get("directory") or []
    paths = sorted(
        str(Path(d["path"]).resolve())
        for d in dirs
        if isinstance(d, dict) and d.get("path")
    )
    if paths:
        return _stable_id("\x00".join(paths))
    return _legacy_dataset_cache_id(dataset_config)


def _legacy_dataset_cache_id(dataset_config: dict) -> str:
    """Pre-directory-keyed id (TOML path hash); kept only to relocate old caches once."""
    path = dataset_config.get("_dataset_toml_path")
    if path:
        return _stable_id(str(Path(path).resolve()))
    return _stable_id(repr(sorted(dataset_config.keys())))


def directory_cache_id(directory_path: str | Path) -> str:
    return _stable_id(str(Path(directory_path).resolve()))


def resolve_directory_cache_dir(
    dataset_config: dict,
    directory_path: str | Path,
    model_name: str,
    *,
    training_config: dict | None = None,
) -> Path:
    cfg = training_config if training_config is not None else {}
    root = resolve_cache_root(cfg, dataset_config=dataset_config)
    dataset_dir = root / dataset_cache_id(dataset_config)
    if not dataset_dir.exists():
        # One-time relocation of caches built under the pre-directory-keyed id (TOML
        # path hash). Best-effort: a concurrent worker may win the rename; both end up
        # at the same destination.
        legacy = root / _legacy_dataset_cache_id(dataset_config)
        if legacy != dataset_dir and legacy.is_dir():
            try:
                legacy.rename(dataset_dir)
                logger.info("Relocated dataset cache %s -> %s", legacy.name, dataset_dir.name)
            except OSError:
                pass
    return dataset_dir / directory_cache_id(directory_path) / model_name
