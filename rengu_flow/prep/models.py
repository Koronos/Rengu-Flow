"""Model registry and HuggingFace download helpers for the prep pipeline.

Centralises every inference model the prep stages can use across tagging,
captioning, and cleanup. Heavy imports (huggingface_hub) are always lazy so
importing this module at parse time has no cost.

``list_models(stage)`` is the primary UI-facing entry point: it returns a list
of dicts with enough metadata to show users what is available and what has
already been downloaded. ``ensure_model(spec_or_id, stage)`` downloads on demand
and returns the local path.
"""

from __future__ import annotations

from pathlib import Path

from rengu_flow.prep.tagger import KNOWN_TAGGERS, TaggerModelSpec
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Caption model registry
# ---------------------------------------------------------------------------

# Each entry: id -> {repo_id, repo_type, filename (optional), notes}
CAPTION_MODELS: dict[str, dict] = {
    "joycaption-beta-one": {
        "repo_id": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "repo_type": "model",
        "filename": None,
        "notes": (
            "LLaVA-style multimodal caption model by fancyfeast. "
            "Full model directory download required; used via Transformers."
        ),
    },
    "toriigate-0.5": {
        "repo_id": "Minthy/ToriiGate-0.5",
        "repo_type": "model",
        "filename": None,
        "notes": (
            "ToriiGate 0.5 — anime-focused captioner. "
            "Full model directory download required; used via Transformers."
        ),
    },
}

# ---------------------------------------------------------------------------
# Cleanup / watermark model registry
# ---------------------------------------------------------------------------

CLEANUP_MODELS: dict[str, dict] = {
    "yolo11-watermark": {
        "repo_id": "fancyfeast/joycaption-watermark-detection",
        "repo_type": "space",
        "filename": "yolo11x-train28-best.pt",
        "notes": (
            "YOLOv11-x watermark detection model from the JoyCaption project. "
            "Loaded as a single .pt file; used via ultralytics."
        ),
    },
    "lama-onnx": {
        "repo_id": "Carve/LaMa-ONNX",
        "repo_type": "model",
        "filename": "lama_fp32.onnx",
        "notes": (
            "Big-LaMa inpainting exported to ONNX — runs on the same onnxruntime "
            "as the taggers (no extra inpainting package)."
        ),
    },
}

# ---------------------------------------------------------------------------
# Cache-check helpers
# ---------------------------------------------------------------------------

def _is_downloaded(repo_id: str, filename: str | None, repo_type: str = "model") -> bool:
    """Return True if the model (or its primary file) appears to be cached locally.

    Uses ``huggingface_hub.try_to_load_from_cache`` for single-file checks.
    For directory models (filename=None) uses ``scan_cache_dir`` to see whether
    the repo has any cached revision. Returns False if huggingface_hub is not
    installed rather than crashing.
    """
    try:
        import huggingface_hub as hfh  # lazy
    except ImportError:
        logger.warning(
            "huggingface_hub not installed — cannot check download status for %s.",
            repo_id,
        )
        return False

    try:
        if filename is not None:
            result = hfh.try_to_load_from_cache(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
            )
            # Cached -> path string; not cached -> None; known-missing -> _CACHED_NO_EXIST
            # sentinel. Only the string means we actually have the file.
            return isinstance(result, str)
        else:
            # For full-directory models, check whether any revision is cached
            info = hfh.scan_cache_dir()
            for repo in info.repos:
                if repo.repo_id == repo_id and repo.repo_type == repo_type:
                    return len(repo.revisions) > 0
            return False
    except Exception as exc:
        logger.warning("Cache check failed for %s: %s", repo_id, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_models(stage: str) -> list[dict]:
    """Return registry entries for ``stage`` enriched with a ``downloaded`` flag.

    Args:
        stage: One of ``"tag"``, ``"caption"``, ``"clean"``.

    Returns:
        List of dicts, one per model. Tagger entries include full
        ``TaggerModelSpec`` fields; caption/cleanup entries include the static
        metadata plus ``downloaded`` and ``available``.

    Raises:
        ValueError: If ``stage`` is not recognised.
    """
    if stage == "tag":
        results = []
        for spec in KNOWN_TAGGERS.values():
            downloaded = _is_downloaded(spec.repo_id, spec.filename, repo_type="model")
            results.append(
                {
                    "id": spec.id,
                    "repo_id": spec.repo_id,
                    "filename": spec.filename,
                    "tags_filename": spec.tags_filename,
                    "subdir": spec.subdir,
                    "input_size": spec.input_size,
                    "general_threshold": spec.general_threshold,
                    "character_threshold": spec.character_threshold,
                    "rating_threshold": spec.rating_threshold,
                    "source": spec.source,
                    "downloaded": downloaded,
                    "available": True,
                }
            )
        return results

    if stage == "quality":
        return []  # quality filter is pure numpy — no downloadable models

    registry = {"caption": CAPTION_MODELS, "clean": CLEANUP_MODELS}.get(stage)
    if registry is not None:
        results = []
        for mid, entry in registry.items():
            downloaded = _is_downloaded(
                entry["repo_id"],
                entry.get("filename"),
                repo_type=entry.get("repo_type", "model"),
            )
            results.append(
                {
                    "id": mid,
                    "repo_id": entry["repo_id"],
                    "repo_type": entry.get("repo_type", "model"),
                    "filename": entry.get("filename"),
                    "notes": entry.get("notes", ""),
                    "downloaded": downloaded,
                    "available": True,
                }
            )
        return results

    raise ValueError(
        f"Unknown stage {stage!r}. Expected one of 'tag', 'caption', 'clean'."
    )


def ensure_model(
    spec_or_id: str | TaggerModelSpec,
    stage: str,
) -> Path:
    """Download a model if not cached and return its local path.

    For taggers returns the path to the ONNX model file. For caption/cleanup
    models with ``filename=None`` (full directories), returns the local snapshot
    directory.

    Args:
        spec_or_id: A model id string.
        stage: One of ``"tag"``, ``"caption"``, ``"clean"``.

    Returns:
        Local ``Path`` to the downloaded file or directory.

    Raises:
        ValueError: If the id is not found in the registry for the given stage.
        ImportError: If huggingface_hub is not installed.
    """
    import huggingface_hub as hfh  # lazy — must be available here

    model_id = str(spec_or_id) if not isinstance(spec_or_id, str) else spec_or_id

    if stage == "tag":
        spec = KNOWN_TAGGERS.get(model_id)
        if spec is None:
            raise ValueError(
                f"Unknown tagger id {model_id!r}. "
                f"Known ids: {list(KNOWN_TAGGERS)}"
            )
        kwargs: dict = dict(repo_id=spec.repo_id, filename=spec.filename)
        if spec.subdir:
            kwargs["subfolder"] = spec.subdir
        local = hfh.hf_hub_download(**kwargs)
        logger.info("Tagger model ready: %s -> %s", spec.id, local)
        return Path(local)

    registry = {"caption": CAPTION_MODELS, "clean": CLEANUP_MODELS}.get(stage)
    if registry is None:
        raise ValueError(
            f"Unknown stage {stage!r}. Expected one of 'tag', 'caption', 'clean'."
        )
    entry = registry.get(model_id)
    if entry is None:
        raise ValueError(
            f"Unknown {stage} model id {model_id!r}. "
            f"Known ids: {list(registry)}"
        )
    repo_id = entry["repo_id"]
    repo_type = entry.get("repo_type", "model")
    filename = entry.get("filename")

    if filename is not None:
        local = hfh.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
        )
        logger.info("Model file ready: %s -> %s", model_id, local)
        return Path(local)
    else:
        snapshot = hfh.snapshot_download(repo_id=repo_id, repo_type=repo_type)
        logger.info("Model snapshot ready: %s -> %s", model_id, snapshot)
        return Path(snapshot)
