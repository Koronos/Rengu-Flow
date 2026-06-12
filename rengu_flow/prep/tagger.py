"""Danbooru-style ONNX tagger ensemble for dataset preparation.

Runs one ONNX model at a time in VRAM, buffers per-image probabilities, merges across
models by MAX probability, then sorts descending and formats as a comma-joined tag line.

Heavy imports (onnxruntime, huggingface_hub) are lazy — inside methods only. PIL and
numpy are base dependencies and may be imported at module level.

The ``infer_factory`` seam in ``run_ensemble`` lets tests inject fake inference without
ever touching onnxruntime. The pure merge step (``merge_model_results``) is a separate
function so unit tests can hit it directly.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from rengu_flow.prep.tag_ops import KAOMOJIS, replace_underscores
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaggerModelSpec:
    """Immutable description of one tagger model.

    ``subdir`` is the repository sub-folder that contains the ONNX file, used by
    models like cl_tagger that bundle several checkpoints in one repo.
    ``tags_filename`` may be a ``.csv`` (WD-style) or a ``.json`` (cl_tagger mapping).
    """

    id: str
    repo_id: str
    filename: str
    tags_filename: str
    subdir: str = ""
    input_size: int = 448
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    rating_threshold: float = 0.50
    # Ratings are near-mutually-exclusive, so they resolve by argmax (one rating tag per
    # image, kept only if it clears rating_threshold) instead of per-tag thresholding.
    include_rating: bool = True
    source: str = ""


KNOWN_TAGGERS: dict[str, TaggerModelSpec] = {
    "pixai-v0.9": TaggerModelSpec(
        id="pixai-v0.9",
        repo_id="deepghs/pixai-tagger-v0.9-onnx",
        filename="model.onnx",
        tags_filename="selected_tags.csv",
        general_threshold=0.30,
        character_threshold=0.75,
        rating_threshold=0.50,
        source="deepghs/pixai-tagger-v0.9-onnx on HuggingFace (June 2026)",
    ),
    "cl-tagger-1.01": TaggerModelSpec(
        id="cl-tagger-1.01",
        repo_id="cella110n/cl_tagger",
        filename="model.onnx",
        tags_filename="tag_mapping.json",
        subdir="cl_tagger_1_01",
        general_threshold=0.35,
        character_threshold=0.85,
        rating_threshold=0.50,
        source="cella110n/cl_tagger on HuggingFace (June 2026)",
    ),
    "wd-eva02-large-v3": TaggerModelSpec(
        id="wd-eva02-large-v3",
        repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        filename="model.onnx",
        tags_filename="selected_tags.csv",
        general_threshold=0.35,
        character_threshold=0.85,
        rating_threshold=0.50,
        source="SmilingWolf/wd-eva02-large-tagger-v3 on HuggingFace (June 2026)",
    ),
    "wd-vit-large-v3": TaggerModelSpec(
        id="wd-vit-large-v3",
        repo_id="SmilingWolf/wd-vit-large-tagger-v3",
        filename="model.onnx",
        tags_filename="selected_tags.csv",
        general_threshold=0.35,
        character_threshold=0.85,
        rating_threshold=0.50,
        source="SmilingWolf/wd-vit-large-tagger-v3 on HuggingFace (June 2026)",
    ),
    "wd-swinv2-v3": TaggerModelSpec(
        id="wd-swinv2-v3",
        repo_id="SmilingWolf/wd-swinv2-tagger-v3",
        filename="model.onnx",
        tags_filename="selected_tags.csv",
        general_threshold=0.35,
        character_threshold=0.85,
        rating_threshold=0.50,
        source="SmilingWolf/wd-swinv2-tagger-v3 on HuggingFace (June 2026)",
    ),
}

DEFAULT_TAGGERS: list[str] = ["pixai-v0.9", "cl-tagger-1.01"]


# ---------------------------------------------------------------------------
# Tag list loading
# ---------------------------------------------------------------------------

def _load_tags_csv(path: Path) -> list[tuple[str, int]]:
    """Parse a WD-style selected_tags.csv.

    Returns list of (name, category) pairs. Category codes: 0=general,
    4=character, 9=rating.  Columns may appear in any order as long as
    ``name`` and ``category`` are present.
    """
    rows: list[tuple[str, int]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            try:
                cat = int(row.get("category", 0))
            except (ValueError, TypeError):
                cat = 0
            if name:
                rows.append((name, cat))
    return rows


def _load_tags_json(path: Path) -> list[tuple[str, int]]:
    """Parse a cl_tagger-style tag_mapping.json.

    The file may be structured as:
      - ``{idx: {"name": str, "category": int}}``
      - ``{idx: str}``  (name only, category assumed 0)
      - ``{"tags": [...], "categories": [...]}``  (parallel arrays)

    Any unrecognised layout logs a clear error and raises, not silently returning empty.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        # Parallel array style: {"tags": [...], "categories": [...]}
        if "tags" in raw and isinstance(raw["tags"], list):
            tags = raw["tags"]
            categories = raw.get("categories", [])
            rows: list[tuple[str, int]] = []
            for i, tag in enumerate(tags):
                if not isinstance(tag, str) or not tag.strip():
                    continue
                try:
                    cat = int(categories[i]) if i < len(categories) else 0
                except (ValueError, TypeError):
                    cat = 0
                rows.append((tag.strip(), cat))
            return rows

        # Dict-of-entries style: {idx: str | dict}
        rows = []
        for _idx, entry in raw.items():
            if isinstance(entry, str):
                name = entry.strip()
                cat = 0
            elif isinstance(entry, dict):
                name = str(entry.get("name", "")).strip()
                try:
                    cat = int(entry.get("category", 0))
                except (ValueError, TypeError):
                    cat = 0
            else:
                continue
            if name:
                rows.append((name, cat))
        if rows:
            return rows

    raise ValueError(
        f"Unrecognised tag mapping layout in {path}. "
        "Expected WD-style CSV, or JSON with {{idx: name/dict}} entries, "
        "or {{\"tags\": [...], \"categories\": [...]}}. "
        "Check the file and update the loader."
    )


def load_tag_list(path: Path) -> list[tuple[str, int]]:
    """Load a tag list from a CSV or JSON file, dispatching by extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_tags_csv(path)
    if suffix == ".json":
        return _load_tags_json(path)
    raise ValueError(
        f"Unknown tag list extension {suffix!r} for {path}. "
        "Expected .csv (WD-style) or .json (cl_tagger-style)."
    )


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess_image(image: Image.Image, input_size: int) -> np.ndarray:
    """PIL image -> float32 BGR numpy array ready for ONNX inference.

    Follows the WD-tagger convention:
    1. Flatten alpha channel onto a white background.
    2. Pad to square using white fill (not crop — preserves aspect ratio).
    3. Resize to ``input_size × input_size`` with LANCZOS.
    4. Convert to BGR float32 and add batch dimension.
    """
    # Flatten alpha onto white
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    else:
        image = image.convert("RGB")

    # Pad to square
    w, h = image.size
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(image, ((side - w) // 2, (side - h) // 2))

    # Resize
    resized = padded.resize((input_size, input_size), Image.LANCZOS)

    # RGB -> BGR float32 with batch dim
    arr = np.array(resized, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB to BGR
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# ONNX session wrapper
# ---------------------------------------------------------------------------

class OnnxTagger:
    """Thin wrapper around an onnxruntime InferenceSession.

    Loads lazily. Falls back to CPU if CUDAExecutionProvider is unavailable,
    with a logged warning.
    """

    def __init__(self, spec: TaggerModelSpec, model_path: Path, tags_path: Path) -> None:
        self.spec = spec
        self.model_path = model_path
        self.tags_path = tags_path
        self._session = None
        self._names: np.ndarray | None = None  # display names (underscores resolved)
        self._categories: np.ndarray | None = None
        self._sigmoid_warned = False

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort  # lazy — never at module top

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        else:
            logger.warning(
                "CUDAExecutionProvider unavailable for %s — running on CPU. "
                "Install onnxruntime-gpu for GPU inference.",
                self.spec.id,
            )
        providers.append("CPUExecutionProvider")

        logger.info("Loading ONNX model %s from %s", self.spec.id, self.model_path)
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        tags = load_tag_list(self.tags_path)
        self._names = np.array([replace_underscores(name) for name, _ in tags])
        self._categories = np.array([cat for _, cat in tags], dtype=np.int64)

    def predict_arrays(self, batch: np.ndarray) -> list[dict[str, float]]:
        """Run inference on a preprocessed (N, H, W, 3) float32 BGR batch.

        Returns one dict per image: ``{display_tag: probability}`` for general/character
        tags clearing their thresholds, plus the argmax rating tag (if enabled and above
        rating_threshold). Thresholding is vectorised — vocabularies reach 42k tags.
        """
        self._ensure_loaded()
        assert self._names is not None and self._categories is not None
        spec = self.spec

        input_name = self._session.get_inputs()[0].name
        probs_batch = np.asarray(self._session.run(None, {input_name: batch})[0])

        # Some exports emit logits instead of sigmoid probabilities; normalize once.
        if probs_batch.min() < 0.0 or probs_batch.max() > 1.0:
            if not self._sigmoid_warned:
                logger.info("%s outputs logits — applying sigmoid.", self.spec.id)
                self._sigmoid_warned = True
            probs_batch = 1.0 / (1.0 + np.exp(-probs_batch))

        n_tags = min(len(self._names), probs_batch.shape[1])
        names = self._names[:n_tags]
        cats = self._categories[:n_tags]
        general_mask = cats == 0
        character_mask = cats == 4
        rating_idx = np.flatnonzero(cats == 9)

        results: list[dict[str, float]] = []
        for probs in probs_batch:
            probs = probs[:n_tags]
            keep = (general_mask & (probs >= spec.general_threshold)) | (
                character_mask & (probs >= spec.character_threshold)
            )
            idxs = np.flatnonzero(keep)
            tag_probs = {names[i]: float(probs[i]) for i in idxs}
            if spec.include_rating and rating_idx.size:
                best = rating_idx[int(np.argmax(probs[rating_idx]))]
                if probs[best] >= spec.rating_threshold:
                    tag_probs[names[best]] = float(probs[best])
            results.append(tag_probs)
        return results

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        """Convenience wrapper: preprocess PIL images, then ``predict_arrays``."""
        batch = np.concatenate(
            [_preprocess_image(img, self.spec.input_size) for img in images], axis=0
        )
        return self.predict_arrays(batch)


# ---------------------------------------------------------------------------
# Default infer factory (builds OnnxTagger instances)
# ---------------------------------------------------------------------------

def _default_infer_factory(spec: TaggerModelSpec) -> Callable[[list[Path]], list[dict[str, float]]]:
    """Build a real inference callable for ``spec``.

    Downloads model files via huggingface_hub if not cached. Returns a
    callable that accepts a batch of image paths and returns per-image
    ``{tag: prob}`` dicts already filtered to threshold. Decode + preprocessing
    run in a small persistent thread pool so the GPU isn't stalled on JPEG
    decode for the whole batch.
    """
    from huggingface_hub import hf_hub_download  # lazy

    subdir = spec.subdir

    def _resolve(filename: str) -> Path:
        kwargs: dict = dict(repo_id=spec.repo_id, filename=filename)
        if subdir:
            kwargs["subfolder"] = subdir
        return Path(hf_hub_download(**kwargs))

    model_path = _resolve(spec.filename)
    tags_path = _resolve(spec.tags_filename)

    tagger = OnnxTagger(spec, model_path, tags_path)
    decode_pool = ThreadPoolExecutor(max_workers=4)

    def _decode(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return _preprocess_image(img, spec.input_size)

    def _infer(image_paths: list[Path]) -> list[dict[str, float]]:
        arrays = list(decode_pool.map(_decode, image_paths))
        return tagger.predict_arrays(np.concatenate(arrays, axis=0))

    return _infer


# ---------------------------------------------------------------------------
# Pure merge logic (the test seam)
# ---------------------------------------------------------------------------

def merge_model_results(
    per_model_dicts: list[dict[str, dict[str, float]]],
    *,
    exclude_tags: Iterable[str] = (),
    prepend_tags: Iterable[str] = (),
    max_tags: int = 255,
) -> dict[str, str]:
    """Merge per-model per-image tag dicts into one tag line per image.

    ``per_model_dicts`` is a list (one entry per model) of ``{image_key: {tag: prob}}``.
    Merging strategy: MAX probability wins across models.

    Returns ``{image_key: comma_joined_tag_line}`` sorted by probability descending,
    with ``prepend_tags`` leading (deduplicated), ``exclude_tags`` removed, capped at
    ``max_tags``.
    """
    # Build image key universe
    all_keys: set[str] = set()
    for model_dict in per_model_dicts:
        all_keys.update(model_dict.keys())

    excluded_lower = {t.lower() for t in exclude_tags}
    prepend = list(prepend_tags)  # preserve order

    result: dict[str, str] = {}
    for key in sorted(all_keys):
        # Merge: max prob across models for each tag
        merged: dict[str, float] = {}
        for model_dict in per_model_dicts:
            for tag, prob in model_dict.get(key, {}).items():
                if tag in merged:
                    merged[tag] = max(merged[tag], prob)
                else:
                    merged[tag] = prob

        # Remove excluded (case-insensitive)
        merged = {t: p for t, p in merged.items() if t.lower() not in excluded_lower}

        # Sort by probability descending
        sorted_tags = sorted(merged, key=lambda t: merged[t], reverse=True)

        # Prepend custom tags (dedup: remove from body if already in prepend)
        prepend_lower = {t.lower() for t in prepend}
        body = [t for t in sorted_tags if t.lower() not in prepend_lower]

        final = prepend + body
        final = final[:max_tags]
        result[key] = ", ".join(final)

    return result


# ---------------------------------------------------------------------------
# Ensemble runner
# ---------------------------------------------------------------------------

def run_ensemble(
    image_paths: list[Path],
    specs: list[TaggerModelSpec],
    *,
    overrides: dict | None = None,
    exclude_tags: Iterable[str] = (),
    prepend_tags: Iterable[str] = (),
    max_tags: int = 255,
    batch_size: int = 16,
    on_progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    infer_factory: Callable[[TaggerModelSpec], Callable] | None = None,
) -> dict[str, str]:
    """Run an ensemble of ONNX taggers over ``image_paths``.

    Models run SEQUENTIALLY (one ONNX session in VRAM at a time). Results are
    merged by max probability. The ``infer_factory`` seam allows tests to inject
    fake inference without importing onnxruntime.

    Args:
        image_paths: Absolute paths to images to tag.
        specs: Ordered list of ``TaggerModelSpec`` to run.
        overrides: Optional per-spec threshold overrides, keyed by spec id.
        exclude_tags: Tags to drop from the merged output (case-insensitive).
        prepend_tags: Tags to insert at the front of every tag line (deduped).
        max_tags: Hard cap on tags per image (default 255).
        batch_size: Images per ONNX forward pass (default 16).
        on_progress: ``fn(done, total, phase_msg)`` called after each batch.
        infer_factory: ``fn(spec) -> fn(batch_paths) -> list[{tag: prob}]``.
            Default builds a real ``OnnxTagger`` and downloads from HuggingFace.

    Returns:
        ``{image_path_str: comma_joined_tag_line}`` for every input path.
    """
    if infer_factory is None:
        infer_factory = _default_infer_factory

    total = len(image_paths)
    # per_model_dicts[m][key] = {tag: prob}
    per_model_dicts: list[dict[str, dict[str, float]]] = []

    for spec in specs:
        if overrides and spec.id in overrides:
            from dataclasses import replace

            spec = replace(spec, **overrides[spec.id])
        phase = f"model {spec.id}"
        if on_progress is not None:
            on_progress(0, total, phase)

        infer = infer_factory(spec)
        model_result: dict[str, dict[str, float]] = {}

        done = 0
        stopped = False
        for batch_start in range(0, total, batch_size):
            if should_stop is not None and should_stop():
                logger.info("run_ensemble: stop requested during %s", phase)
                stopped = True
                break
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            tag_dicts = infer(batch_paths)
            for path, tag_dict in zip(batch_paths, tag_dicts):
                model_result[str(path)] = tag_dict
            done += len(batch_paths)
            if on_progress is not None:
                on_progress(done, total, phase)

        per_model_dicts.append(model_result)
        if stopped:
            # Merge what we have: max-prob merging tolerates a model that only covered a
            # prefix of the images (those images just get fewer votes).
            break

    return merge_model_results(
        per_model_dicts,
        exclude_tags=exclude_tags,
        prepend_tags=prepend_tags,
        max_tags=max_tags,
    )
