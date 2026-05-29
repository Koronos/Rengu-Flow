"""Resolve dataset cache directories under cache_root (v2 only)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _stable_id(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def caption_cache_key(caption: str) -> str:
    """Full SHA-256 hex for caption dedup during text-embedding cache."""
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()


def default_cache_root() -> Path:
    """Untracked cache root at rengu-flow installation (repo) root."""
    return Path(__file__).resolve().parents[2] / "cache"


def warn_legacy_dataset_cache_root(dataset_config: dict) -> None:
    """Warn when dataset TOML still sets cache_root (training config owns this key)."""
    if dataset_config.get("cache_root") is None:
        return
    logger.warning(
        "cache_root in dataset TOML is ignored; set cache_root in the training config "
        "(Training loop section)."
    )


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
        warn_legacy_dataset_cache_root(dataset_config)
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        root = default_cache_root()
    else:
        root = Path(str(raw)).expanduser()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_cache_id(dataset_config: dict) -> str:
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
    return (
        root
        / dataset_cache_id(dataset_config)
        / directory_cache_id(directory_path)
        / model_name
    )
