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
    # Character (4) and copyright/series (3) tags — taggers are weakest at character
    # names, so users can drop the whole category and rely on their own trigger tags.
    include_character: bool = True
    # Input convention — verified against each export's reference inference code:
    #   "wd":           pad-square white + resize, BGR uint8-range float, NHWC (WD v3 line)
    #   "norm05_rgb":   plain resize, RGB /255 normalized (mean=std=0.5), NCHW
    #                   (deepghs exports, e.g. pixai-tagger-v0.9-onnx preprocess.json)
    #   "norm05_bgr_pad": pad-square white + resize, BGR /255 normalized (0.5), NCHW
    #                   (cl_tagger official space app.py)
    preprocess: str = "wd"
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
        preprocess="norm05_rgb",
        source="deepghs/pixai-tagger-v0.9-onnx on HuggingFace (June 2026)",
    ),
    "cl-tagger-1.02": TaggerModelSpec(
        id="cl-tagger-1.02",
        repo_id="cella110n/cl_tagger",
        filename="model_optimized.onnx",
        tags_filename="tag_mapping.json",
        subdir="cl_tagger_1_02",
        general_threshold=0.35,
        character_threshold=0.75,
        rating_threshold=0.50,
        preprocess="norm05_bgr_pad",
        source="cella110n/cl_tagger on HuggingFace (June 2026)",
    ),
    "cl-tagger-1.01": TaggerModelSpec(
        id="cl-tagger-1.01",
        repo_id="cella110n/cl_tagger",
        filename="model_optimized.onnx",
        tags_filename="tag_mapping.json",
        subdir="cl_tagger_1_01",
        general_threshold=0.35,
        character_threshold=0.75,
        rating_threshold=0.50,
        preprocess="norm05_bgr_pad",
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


# cl_tagger names its categories; map them onto WD category codes. Quality tags
# (masterpiece/best quality/...) ride the general threshold; Copyright (series names)
# rides the character threshold; Model/Meta tags are bookkeeping and never emitted.
_NAMED_CATEGORIES = {
    "general": 0,
    "quality": 0,
    "copyright": 3,
    "character": 4,
    "meta": 5,
    "model": 5,
    "rating": 9,
}


def _load_tags_json(path: Path) -> list[tuple[str, int]]:
    """Parse a cl_tagger-style tag_mapping.json.

    Verified layout (cella110n/cl_tagger): ``{"0": {"tag": str, "category": "Rating"}}``
    — index keys, "tag" name key, NAMED categories. Older/other exports may use
    ``{"name": ..., "category": int}`` or parallel ``{"tags": [...], "categories":
    [...]}`` arrays; all three parse. Anything else raises (no silent fallback).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def coerce_category(value) -> int:
        if isinstance(value, str) and not value.isdigit():
            return _NAMED_CATEGORIES.get(value.strip().lower(), 0)
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    if isinstance(raw, dict):
        # Parallel array style: {"tags": [...], "categories": [...]}
        if "tags" in raw and isinstance(raw["tags"], list):
            tags = raw["tags"]
            categories = raw.get("categories", [])
            rows: list[tuple[str, int]] = []
            for i, tag in enumerate(tags):
                if not isinstance(tag, str) or not tag.strip():
                    continue
                cat = coerce_category(categories[i]) if i < len(categories) else 0
                rows.append((tag.strip(), cat))
            return rows

        # Dict-of-entries style: {idx: str | dict}. Model outputs index by position,
        # so iterate in numeric key order, not insertion order.
        def entry_order(item):
            key = item[0]
            return (0, int(key)) if str(key).isdigit() else (1, 0)

        rows = []
        for _idx, entry in sorted(raw.items(), key=entry_order):
            if isinstance(entry, str):
                name = entry.strip()
                cat = 0
            elif isinstance(entry, dict):
                name = str(entry.get("tag") or entry.get("name") or "").strip()
                cat = coerce_category(entry.get("category", 0))
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

def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Flatten any alpha channel onto a white background."""
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert("RGB")


def _pad_square_white(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(image, ((side - w) // 2, (side - h) // 2))
    return padded


def _preprocess_image(
    image: Image.Image, input_size: int, mode: str = "wd"
) -> np.ndarray:
    """PIL image -> float32 batch-of-1 array in the convention ``mode`` expects.

    Modes (each verified against the export's own reference inference code):
    - ``wd``: pad-square white, resize, BGR in the 0..255 range, NHWC. (WD v3 line.)
    - ``norm05_rgb``: plain resize (no padding), RGB, /255 then (x-0.5)/0.5, NCHW.
      (deepghs exports — pixai-tagger-v0.9-onnx ships exactly this preprocess.json.)
    - ``norm05_bgr_pad``: pad-square white, resize, BGR, /255 then (x-0.5)/0.5, NCHW.
      (cl_tagger — official space app.py.)
    """
    image = _flatten_to_rgb(image)

    if mode == "wd":
        resized = _pad_square_white(image).resize((input_size, input_size), Image.BICUBIC)
        arr = np.array(resized, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
        return np.expand_dims(arr, axis=0)  # (1, H, W, 3)

    if mode == "norm05_rgb":
        resized = image.resize((input_size, input_size), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - 0.5) / 0.5
        return np.expand_dims(arr, axis=0)  # (1, 3, H, W)

    if mode == "norm05_bgr_pad":
        resized = _pad_square_white(image).resize((input_size, input_size), Image.BICUBIC)
        arr = np.array(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = arr[::-1, :, :].copy()  # RGB -> BGR (channel axis)
        arr = (arr - 0.5) / 0.5
        return np.expand_dims(arr, axis=0)  # (1, 3, H, W)

    raise ValueError(f"Unknown preprocess mode {mode!r}")


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
        self._names: np.ndarray | None = None  # original tag names (underscores kept)
        self._categories: np.ndarray | None = None
        self._output_name: str | None = None
        self._sigmoid_warned = False

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        # Windows-only, on demand: install + expose the cu12 runtime so the CUDAExecutionProvider can
        # load (onnxruntime-gpu is cu12, our torch is cu13). No-op off Windows. Must run before the
        # session is created, so the GPU tagger only pulls ~1 GB of CUDA libs the first time it runs.
        from rengu_flow.prep.onnx_runtime import ensure_onnx_cuda_runtime

        ensure_onnx_cuda_runtime()

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
        # Keep the ORIGINAL danbooru form (underscores) internally; underscores are
        # collapsed to spaces only at output time, and only when the caller asks for it
        # (see merge_model_results' ``underscores`` flag). This keeps one canonical form
        # for matching against control lists regardless of the chosen output style.
        self._names = np.array([name for name, _ in tags])
        self._categories = np.array([cat for _, cat in tags], dtype=np.int64)
        self._output_name = self._pick_output_name()

    def _pick_output_name(self) -> str:
        """Choose the per-tag score tensor among the model's outputs.

        Some exports emit several heads (pixai-tagger-v0.9-onnx: ``embedding``,
        ``logits``, ``prediction``) — blindly taking outputs[0] would read the
        1024-dim embedding and produce random tags. Prefer the post-sigmoid
        ``prediction``, then ``logits``, then a single output, then any output
        whose last dim matches the vocabulary; anything else is an error.
        """
        outputs = self._session.get_outputs()
        by_name = {o.name: o for o in outputs}
        for preferred in ("prediction", "logits"):
            if preferred in by_name:
                return preferred
        if len(outputs) == 1:
            return outputs[0].name
        for o in outputs:
            if o.shape and o.shape[-1] == len(self._names):
                return o.name
        raise ValueError(
            f"{self.spec.id}: cannot identify the tag-score output among "
            f"{[(o.name, o.shape) for o in outputs]} (vocabulary size {len(self._names)})."
        )

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
        probs_batch = np.asarray(
            self._session.run([self._output_name], {input_name: batch})[0]
        )

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
        # Copyright (series) tags ride the character threshold: same name-like nature.
        if spec.include_character:
            character_mask = (cats == 4) | (cats == 3)
        else:
            character_mask = np.zeros_like(general_mask)
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


# ---------------------------------------------------------------------------
# Default infer factory (builds OnnxTagger instances)
# ---------------------------------------------------------------------------

class _ModelInfer:
    """Real per-model inference callable with a ``close()`` so chunk-outer tagging can
    load one model, run a chunk, then free its ONNX session before loading the next —
    keeping only one model resident at a time (no co-residence OOM with many models)."""

    def __init__(self, spec: TaggerModelSpec, model_path: Path, tags_path: Path):
        self.spec = spec
        self.tagger = OnnxTagger(spec, model_path, tags_path)
        self.pool = ThreadPoolExecutor(max_workers=4)

    def _decode(self, path: Path):
        # Return None (not raise) for an unreadable/corrupt image so one bad file can't
        # abort the whole tag job — the GPU map would otherwise propagate the exception.
        try:
            with Image.open(path) as img:
                return _preprocess_image(img, self.spec.input_size, self.spec.preprocess)
        except Exception as exc:  # noqa: BLE001 — PIL.UnidentifiedImageError, truncated files, etc.
            logger.warning("Tagger: skipping unreadable image %s: %s", path, exc)
            return None

    def __call__(self, image_paths: list[Path]) -> list[dict[str, float]]:
        decoded = list(self.pool.map(self._decode, image_paths))
        return _predict_decoded(decoded, self.tagger.predict_arrays)

    def close(self) -> None:
        self.pool.shutdown(wait=False)
        self.tagger._session = None  # drop the ONNX session; VRAM is reclaimed by the caller


def _default_infer_factory(spec: TaggerModelSpec) -> "_ModelInfer":
    """Build a real, closeable inference callable for ``spec`` (downloads model files via
    huggingface_hub if not cached)."""
    from huggingface_hub import hf_hub_download  # lazy

    def _resolve(filename: str) -> Path:
        kwargs: dict = dict(repo_id=spec.repo_id, filename=filename)
        if spec.subdir:
            kwargs["subfolder"] = spec.subdir
        return Path(hf_hub_download(**kwargs))

    return _ModelInfer(spec, _resolve(spec.filename), _resolve(spec.tags_filename))


def _predict_decoded(decoded, predict_arrays) -> list[dict[str, float]]:
    """Run prediction over the successfully-decoded arrays, keeping the output aligned to
    ``decoded`` — entries that failed to decode (``None``) stay an empty ``{}`` so a single
    unreadable image never shifts or aborts the rest of the batch."""
    valid = [(i, a) for i, a in enumerate(decoded) if a is not None]
    out: list[dict[str, float]] = [{} for _ in decoded]
    if valid:
        probs = predict_arrays(np.concatenate([a for _, a in valid], axis=0))
        for (i, _arr), prob in zip(valid, probs):
            out[i] = prob
    return out


# ---------------------------------------------------------------------------
# Pure merge logic (the test seam)
# ---------------------------------------------------------------------------

def merge_model_results(
    per_model_dicts: list[dict[str, dict[str, float]]],
    *,
    exclude_tags: Iterable[str] = (),
    prepend_tags: Iterable[str] = (),
    max_tags: int = 255,
    underscores: bool = False,
) -> dict[str, str]:
    """Merge per-model per-image tag dicts into one tag line per image.

    ``per_model_dicts`` is a list (one entry per model) of ``{image_key: {tag: prob}}``.
    Merging strategy: MAX probability wins across models.

    Returns ``{image_key: comma_joined_tag_line}`` sorted by probability descending,
    with ``prepend_tags`` leading (deduplicated), ``exclude_tags`` removed, capped at
    ``max_tags``.

    ``underscores`` controls the output form: ``False`` (default) emits the
    natural-language ``long hair`` form, ``True`` keeps the original danbooru
    ``long_hair`` form. Matching for exclude/prepend is underscore-insensitive
    either way, so the same exclude list works regardless of the chosen form.
    """
    # underscore-insensitive, case-insensitive match key (jpeg_artifacts == "jpeg artifacts")
    def _key(t: str) -> str:
        return t.lower().replace("_", " ")

    # Build image key universe
    all_keys: set[str] = set()
    for model_dict in per_model_dicts:
        all_keys.update(model_dict.keys())

    excluded = {_key(t) for t in exclude_tags}
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

        # Remove excluded (case- and underscore-insensitive)
        merged = {t: p for t, p in merged.items() if _key(t) not in excluded}

        # Sort by probability descending
        sorted_tags = sorted(merged, key=lambda t: merged[t], reverse=True)

        # Prepend custom tags (dedup: remove from body if already in prepend)
        prepend_keys = {_key(t) for t in prepend}
        body = [t for t in sorted_tags if _key(t) not in prepend_keys]

        final = prepend + body
        final = final[:max_tags]
        if not underscores:  # taggers/control lists hold the underscore form; spaces only at output
            final = [replace_underscores(t) for t in final]
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
    underscores: bool = False,
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
        underscores: Keep the original ``long_hair`` form (True) or emit ``long hair`` (False, default).
        on_progress: ``fn(done, total, phase_msg)`` called after each batch.
        infer_factory: ``fn(spec) -> fn(batch_paths) -> list[{tag: prob}]``.
            Default builds a real ``OnnxTagger`` and downloads from HuggingFace.

    Returns:
        ``{image_path_str: comma_joined_tag_line}`` for every input path.
    """
    # Single chunk = model-outer: each model loaded once, runs every batch, merged at the end.
    result: dict[str, str] = {}
    run_ensemble_chunked(
        image_paths, specs,
        chunk_size=max(1, len(image_paths)),
        overrides=overrides, exclude_tags=exclude_tags, prepend_tags=prepend_tags,
        max_tags=max_tags, batch_size=batch_size, underscores=underscores,
        on_chunk=lambda _paths, merged: result.update(merged),
        on_progress=on_progress, should_stop=should_stop, infer_factory=infer_factory,
    )
    return result


def run_ensemble_chunked(
    image_paths: list[Path],
    specs: list[TaggerModelSpec],
    *,
    chunk_size: int = 512,
    overrides: dict | None = None,
    exclude_tags: Iterable[str] = (),
    prepend_tags: Iterable[str] = (),
    max_tags: int = 255,
    batch_size: int = 16,
    underscores: bool = False,
    on_chunk: Callable[[list[Path], dict[str, str]], None],
    on_progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    infer_factory: Callable[[TaggerModelSpec], Callable] | None = None,
) -> None:
    """Chunk-outer ensemble for resumable tagging.

    Each chunk of ``chunk_size`` images is run through ALL models — one resident at a time
    (loaded, inferred over the chunk in ``batch_size`` forwards, then closed and its VRAM
    reclaimed) — then merged. The fully-complete ``{path: tag_line}`` for that chunk is handed
    to ``on_chunk`` so the caller can write + save it immediately and resume per chunk. A chunk
    interrupted mid-way is NOT handed to ``on_chunk`` (it would be missing models), so resume
    re-does only that chunk. Keeping one model resident scales to many models without OOM; the
    per-chunk reload is cheap (~0.5 s/model) versus the early-completion + resumability it buys.
    """
    from dataclasses import replace

    if infer_factory is None:
        infer_factory = _default_infer_factory

    total = len(image_paths)
    done = 0
    stopped = False
    for chunk_start in range(0, total, chunk_size):
        if should_stop is not None and should_stop():
            break
        chunk = image_paths[chunk_start : chunk_start + chunk_size]
        per_model: list[dict[str, dict[str, float]]] = []
        for spec in specs:
            if should_stop is not None and should_stop():
                stopped = True
                break
            spec_eff = (
                replace(spec, **overrides[spec.id])
                if (overrides and spec.id in overrides)
                else spec
            )
            infer = infer_factory(spec_eff)
            tag_dicts: list[dict[str, float]] = []
            try:
                for b in range(0, len(chunk), batch_size):
                    if should_stop is not None and should_stop():
                        stopped = True
                        break
                    tag_dicts.extend(infer(chunk[b : b + batch_size]))
            finally:
                close = getattr(infer, "close", None)
                if close is not None:
                    close()
                    from rengu_flow.utils.common import empty_cuda_cache

                    empty_cuda_cache()  # free this model before loading the next (one resident)
            if stopped:
                break
            per_model.append({str(p): d for p, d in zip(chunk, tag_dicts)})
            if on_progress is not None:
                on_progress(done, total, f"model {spec_eff.id}")
        if stopped:
            break  # discard this partial chunk (not persisted); resume re-does only it
        merged = merge_model_results(
            per_model, exclude_tags=exclude_tags, prepend_tags=prepend_tags,
            max_tags=max_tags, underscores=underscores,
        )
        on_chunk(chunk, merged)
        done += len(chunk)
        if on_progress is not None:
            on_progress(done, total, f"tagged {done}/{total}")
