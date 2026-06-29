"""Tag dropout for dataset captions (rules with per-tag drop probability)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TagDropoutRule:
    tags: frozenset[str]
    drop_probability: float


@dataclass
class TagDropoutConfig:
    enabled: bool = False
    default_probability: float = 0.0
    mode: str = "per_tag"  # per_tag | full
    case_sensitive: bool = False
    rules: list[TagDropoutRule] = field(default_factory=list)


def load_tags_file(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_tags(caption: str, delimiter: str) -> list[str]:
    if not caption.strip():
        return []
    parts = caption.split(delimiter)
    return [p.strip() for p in parts if p.strip()]


def join_tags(tags: list[str], delimiter: str) -> str:
    if not tags:
        return ""
    return delimiter.join(tags)


def _tag_key(tag: str, case_sensitive: bool) -> str:
    # Underscore and space forms are equivalent (long_hair == "long hair"), so a control
    # list kept in the original danbooru (underscore) form drops both forms regardless of
    # how the tagger emitted them. Applied on both sides, so the match is symmetric (this
    # also collapses kaomoji underscores like ^_^ -> "^ ^", but consistently, so they still
    # match each other). Case folding still honours case_sensitive.
    key = tag.replace("_", " ")
    return key if case_sensitive else key.lower()


def resolve_tag_probability(
    tag: str,
    rules: list[TagDropoutRule],
    default_probability: float,
    *,
    case_sensitive: bool,
) -> float:
    key = _tag_key(tag, case_sensitive)
    for rule in rules:
        for rule_tag in rule.tags:
            if _tag_key(rule_tag, case_sensitive) == key:
                return rule.drop_probability
    return default_probability


def _drop_tag(rng: random.Random, probability: float) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return rng.random() < probability


def apply_tag_dropout(
    caption: str,
    config: TagDropoutConfig,
    rng: random.Random,
    *,
    delimiter: str = ", ",
) -> str:
    if not config.enabled:
        return caption
    tags = split_tags(caption, delimiter)
    if not tags:
        return caption

    if config.mode == "full":
        if not _drop_tag(rng, config.default_probability):
            return caption
        kept: list[str] = []
        for t in tags:
            p = resolve_tag_probability(
                t, config.rules, config.default_probability, case_sensitive=config.case_sensitive
            )
            if p <= 0:
                kept.append(t)
        return join_tags(kept, delimiter)

    kept = []
    for t in tags:
        p = resolve_tag_probability(
            t, config.rules, config.default_probability, case_sensitive=config.case_sensitive
        )
        if not _drop_tag(rng, p):
            kept.append(t)
    return join_tags(kept, delimiter)


def build_tag_dropout_config(
    directory_config: dict[str, Any],
    dataset_config: dict[str, Any],
    *,
    tags_file_base: Path | None = None,
) -> TagDropoutConfig:
    def _get(key: str, default=None):
        if key in directory_config:
            return directory_config[key]
        return dataset_config.get(key, default)

    enabled = bool(_get("tag_dropout_enabled", False))
    default_probability = float(_get("tag_dropout_probability", 0.0) or 0.0)
    mode = _get("tag_dropout_mode", "per_tag") or "per_tag"
    if mode not in ("per_tag", "full"):
        mode = "per_tag"
    case_sensitive = bool(_get("tag_match_case_sensitive", False))

    raw_rules = list(_get("tag_dropout_rules") or [])
    rules: list[TagDropoutRule] = []
    for entry in raw_rules:
        if not isinstance(entry, dict):
            continue
        p = float(entry.get("drop_probability", 0.0))
        tag_list: list[str] = []
        if entry.get("tags"):
            tag_list.extend(str(t) for t in entry["tags"])
        tags_file = entry.get("tags_file")
        if tags_file and tags_file_base is not None:
            path = Path(tags_file)
            if not path.is_absolute():
                path = tags_file_base / path
            if path.is_file():
                tag_list.extend(load_tags_file(path))
        if tag_list:
            rules.append(TagDropoutRule(tags=frozenset(tag_list), drop_probability=p))

    return TagDropoutConfig(
        enabled=enabled,
        default_probability=default_probability,
        mode=mode,
        case_sensitive=case_sensitive,
        rules=rules,
    )
