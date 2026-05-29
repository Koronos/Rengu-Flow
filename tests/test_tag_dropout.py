"""Unit tests for tag dropout."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from rengu_flow.data.tag_dropout import (
    TagDropoutConfig,
    TagDropoutRule,
    apply_tag_dropout,
    build_tag_dropout_config,
    resolve_tag_probability,
    split_tags,
)


def test_split_tags_comma_delimiter():
    assert split_tags("a, b, c", ", ") == ["a", "b", "c"]


def test_resolve_tag_probability_first_rule_wins():
    rules = [TagDropoutRule(tags=frozenset(["char"]), drop_probability=0.1)]
    assert resolve_tag_probability("char", rules, 0.5, case_sensitive=False) == 0.1
    assert resolve_tag_probability("other", rules, 0.5, case_sensitive=False) == 0.5


def test_per_tag_dropout_deterministic():
    config = TagDropoutConfig(
        enabled=True,
        default_probability=1.0,
        mode="per_tag",
        rules=[TagDropoutRule(tags=frozenset(["keep"]), drop_probability=0.0)],
    )
    rng1 = random.Random(99)
    rng2 = random.Random(99)
    out1 = apply_tag_dropout("keep, drop1, drop2", config, rng1, delimiter=", ")
    out2 = apply_tag_dropout("keep, drop1, drop2", config, rng2, delimiter=", ")
    assert out1 == out2
    assert "keep" in out1


def test_full_mode_keeps_zero_probability_tags():
    config = TagDropoutConfig(
        enabled=True,
        default_probability=1.0,
        mode="full",
        rules=[TagDropoutRule(tags=frozenset(["anchor"]), drop_probability=0.0)],
    )
    rng = random.Random(1)
    # Force full drop via default_probability=1
    out = apply_tag_dropout("anchor, b, c", config, rng, delimiter=", ")
    assert out == "anchor"


def test_build_tag_dropout_config_from_toml_dict(tmp_path: Path):
    tags_file = tmp_path / "drop.txt"
    tags_file.write_text("bad_tag\n", encoding="utf-8")
    dataset = {
        "tag_dropout_enabled": True,
        "tag_dropout_probability": 0.3,
        "tag_dropout_rules": [
            {"tags": ["hero"], "drop_probability": 0.05},
            {"tags_file": "drop.txt", "drop_probability": 1.0},
        ],
    }
    cfg = build_tag_dropout_config({}, dataset, tags_file_base=tmp_path)
    assert cfg.enabled
    assert cfg.default_probability == 0.3
    assert resolve_tag_probability("hero", cfg.rules, cfg.default_probability, case_sensitive=False) == 0.05
    assert resolve_tag_probability("bad_tag", cfg.rules, cfg.default_probability, case_sensitive=False) == 1.0


def test_build_rejects_invalid_mode_defaults_to_per_tag():
    cfg = build_tag_dropout_config(
        {"tag_dropout_mode": "invalid"},
        {"tag_dropout_enabled": True, "tag_dropout_mode": "invalid"},
    )
    assert cfg.mode == "per_tag"
