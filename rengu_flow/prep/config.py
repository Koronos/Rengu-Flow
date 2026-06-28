"""Prep-job TOML config: one file describes a dataset folder + per-stage options.

Parsing is tolerant (unknown keys are ignored with a log line, never fatal) so the
schema can evolve freely. A stage reads only its own section; sections for other
stages may coexist in the same file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

STAGES = ("tag", "caption", "clean", "quality")


@dataclass
class TagStageConfig:
    models: list[str] = field(default_factory=lambda: ["pixai-v0.9", "cl-tagger-1.02"])
    overrides: dict[str, dict] = field(default_factory=dict)  # spec id -> field overrides
    exclude_tags: list[str] = field(default_factory=list)
    prepend_tags: list[str] = field(default_factory=list)
    max_tags: int = 255
    batch_size: int = 16
    overwrite: bool = False  # False: skip images whose line 1 already has tags
    # Global confidence floors applied to every selected model (None/0 = per-model
    # defaults); per-model [tag.overrides.<id>] entries still win.
    general_threshold: float | None = None
    character_threshold: float | None = None
    rating_threshold: float | None = None  # argmax rating tag kept only if it clears this
    include_character_tags: bool = True  # character + series names (taggers' weak spot)
    include_rating: bool = True  # one argmax rating tag (general/sensitive/...)


@dataclass
class CaptionStageConfig:
    model: str = "joycaption-beta-one"
    quantization: str = "bf16"
    prompt: str = ""  # custom prompt; overrides the composed base+modifiers
    prompt_base: str = "descriptive-long"
    prompt_modifiers: list[str] = field(default_factory=lambda: ["demographics"])
    character_name: str = ""  # trigger name (inherent traits absorbed into it)
    character_canon: str = ""  # canonical look; deviations from it get described
    outfit: str = "describe"  # describe | omit | mixed (only with character_name)
    target_line: int = 2  # 1-based caption line; 3+ adds caption variants
    max_new_tokens: int = 512
    temperature: float | None = None  # None = the model's recommended sampling
    top_p: float | None = None
    exact_generation: bool = False  # ToriiGate: per-image (unpadded), exact but ~2.5x slower
    batch_size: int = 4
    use_tags_as_grounding: bool = True
    overwrite: bool = False
    max_image_side: int = 1536  # downscale long side before the VLM (0 = off)
    min_image_side: int = 0  # skip images smaller than this (0 = off)


@dataclass
class CleanStageConfig:
    confidence: float = 0.35
    mask_dilation_px: int = 8
    in_place: bool = False
    output_dir: str = ""
    copy_undetected: bool = True


@dataclass
class QualityStageConfig:
    # "blur" (Laplacian, dep-free) | "aesthetic" (deepghs booru appeal) | "iqa" (pyiqa technical NR-IQA)
    metric: str = "blur"
    blur_threshold: float = 80.0  # blur: Laplacian-variance floor (long-side-512 copy); tune per set
    min_side: int = 0  # blur: flag images whose shorter side is below this (0 = off)
    min_detail: float = 0.0  # blur: flag low effective resolution (pixelated/upscaled); 0 = off
    aesthetic_min_label: str = "normal"  # aesthetic: flag images ranked below this booru label
    aesthetic_model: str = ""  # aesthetic: imgutils model_name override ("" = its default)
    iqa_model: str = "clipiqa"  # iqa: pyiqa model (clipiqa/arniqa: any domain; musiq/maniqa: photos)
    iqa_threshold: float = 10.0  # iqa: percentile cull 0..100 — flag the lowest N% by quality in the set
    action: str = "report"  # "report" (non-destructive) | "move" flagged into <path>/low_quality
    output_dir: str = ""  # destination for moved files (default <path>/low_quality)


@dataclass
class PrepConfig:
    path: str = ""
    caption_format: str = "sidecar"  # "sidecar" | "json"
    caption_ext: str = ".txt"
    tag: TagStageConfig = field(default_factory=TagStageConfig)
    caption: CaptionStageConfig = field(default_factory=CaptionStageConfig)
    clean: CleanStageConfig = field(default_factory=CleanStageConfig)
    quality: QualityStageConfig = field(default_factory=QualityStageConfig)

    def validate_for_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown prep stage {stage!r}; expected one of {STAGES}")
        if not self.path:
            raise ValueError("Prep config needs a dataset 'path'")
        if not Path(self.path).is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.path}")
        if self.caption_format not in ("sidecar", "json"):
            raise ValueError(f"Unknown caption_format {self.caption_format!r}")


def _fill_dataclass(instance, data: dict, *, context: str):
    """Copy known keys from ``data`` onto ``instance``; log and drop unknown keys."""
    known = set(instance.__dataclass_fields__)
    for key, value in data.items():
        if key in known:
            setattr(instance, key, value)
        else:
            logger.info("Ignoring unknown prep config key %s.%s", context, key)
    return instance


def parse_prep_config(data: dict) -> PrepConfig:
    config = PrepConfig()
    for key in ("path", "caption_format", "caption_ext"):
        if key in data:
            setattr(config, key, data[key])
    if isinstance(data.get("tag"), dict):
        _fill_dataclass(config.tag, data["tag"], context="tag")
    if isinstance(data.get("caption"), dict):
        _fill_dataclass(config.caption, data["caption"], context="caption")
    if isinstance(data.get("clean"), dict):
        _fill_dataclass(config.clean, data["clean"], context="clean")
    if isinstance(data.get("quality"), dict):
        _fill_dataclass(config.quality, data["quality"], context="quality")
    return config


def load_prep_config(path: str | Path) -> PrepConfig:
    import toml

    with open(path, encoding="utf-8") as f:
        return parse_prep_config(toml.load(f))
