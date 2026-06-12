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

STAGES = ("tag", "caption", "clean")


@dataclass
class TagStageConfig:
    models: list[str] = field(default_factory=lambda: ["pixai-v0.9", "cl-tagger-1.02"])
    overrides: dict[str, dict] = field(default_factory=dict)  # spec id -> field overrides
    exclude_tags: list[str] = field(default_factory=list)
    prepend_tags: list[str] = field(default_factory=list)
    max_tags: int = 255
    batch_size: int = 16
    overwrite: bool = False  # False: skip images whose line 1 already has tags


@dataclass
class CaptionStageConfig:
    model: str = "joycaption-beta-one"
    quantization: str = "bf16"
    prompt: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
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
class PrepConfig:
    path: str = ""
    caption_format: str = "sidecar"  # "sidecar" | "json"
    caption_ext: str = ".txt"
    tag: TagStageConfig = field(default_factory=TagStageConfig)
    caption: CaptionStageConfig = field(default_factory=CaptionStageConfig)
    clean: CleanStageConfig = field(default_factory=CleanStageConfig)

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
    return config


def load_prep_config(path: str | Path) -> PrepConfig:
    import toml

    with open(path, encoding="utf-8") as f:
        return parse_prep_config(toml.load(f))


def render_prep_config(config: PrepConfig) -> str:
    """Serialize a PrepConfig back to TOML (used by the UI when staging a job)."""
    import toml
    from dataclasses import asdict

    return toml.dumps(asdict(config))
