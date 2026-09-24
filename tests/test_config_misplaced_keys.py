"""Top-level training keys written inside [model] / [adapter] are rejected at validate time.

In TOML every key after a ``[section]`` header belongs to that table, so a
``blocks_to_swap = 20`` below ``[model]`` becomes ``model.blocks_to_swap`` — which nothing
reads, so the setting was silently ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rengu_flow.config.validation import (
    ConfigValidationError,
    collect_validation_errors,
    top_level_training_keys,
    validate_config,
)
from rengu_flow.registry.model_capabilities import (
    ADAPTER_FIELD_TEMPLATES,
    model_capability_registry,
)

pytestmark = pytest.mark.no_ui_db

SRC_ROOT = Path(__file__).resolve().parents[1] / "rengu_flow"


def _config(**model_extra):
    return {
        "dataset": "examples/minimal_krea2_dataset.toml",
        "model": {
            "type": "krea2",
            "dtype": "bfloat16",
            "transformer_path": "t.safetensors",
            "vae_path": "v.safetensors",
            "text_encoder_path": "te.safetensors",
            **model_extra,
        },
        "adapter": {"type": "lora", "rank": 16},
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "blocks_to_swap", 20),
        ("model", "compile", True),
        ("model", "compile_dynamic", True),
        ("model", "compile_scope", "block"),
        ("model", "reentrant_activation_checkpointing", True),
        ("model", "activation_checkpointing", True),
        ("model", "gradient_accumulation_steps", 4),
        ("adapter", "blocks_to_swap", 20),
        ("adapter", "gradient_clipping", 1.0),
    ],
)
def test_top_level_key_inside_section_is_rejected(section, key, value):
    cfg = _config()
    cfg[section][key] = value
    with pytest.raises(ConfigValidationError, match=f"{section}.{key}: '{key}' belongs at top level"):
        validate_config(cfg)


@pytest.mark.parametrize("key,value", [("blocks_to_swap", 20), ("compile_scope", "block")])
def test_same_key_at_top_level_passes(key, value):
    cfg = _config()
    cfg[key] = value
    assert not [i for i in collect_validation_errors(cfg) if "belongs at top level" in i]


def test_top_level_keys_never_collide_with_section_keys():
    """A key that is legitimately read under [model] / [adapter] must not be flagged."""
    section_keys = {
        spec["path"].split(".", 1)[1]
        for cap in model_capability_registry.values()
        for spec in cap.model_fields
    } | {
        spec["path"].split(".", 1)[1]
        for specs in ADAPTER_FIELD_TEMPLATES.values()
        for spec in specs
    }
    assert not section_keys & top_level_training_keys()


# Top-level reads: config.get("x") / config["x"] on the full training config.
_TOP_LEVEL_READ = re.compile(r'\b(?:self\.)?(?:config|training_config)\s*(?:\.get\(\s*|\[\s*)["\']([a-z0-9_]+)["\']')
_NOT_TRAINING_KNOBS = {
    "model", "adapter", "optimizer", "dataset", "preview", "tracking", "train", "tread",
    "bench", "_dataset_config_loaded", "cache_format", "pretrained_model_name_or_path",
    "max_images", "subsample_ratio",  # directory-level dataset keys read via a `config` local
}


def test_registry_covers_every_top_level_read():
    """Every top-level key the code reads is known to the misplaced-key guard, so a new knob
    (added to the code but not defaulted) cannot silently escape it. Add missing ones to
    validation._OPTIONAL_TOP_LEVEL_KEYS."""
    read: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        if "vendor" in path.parts:
            continue
        read |= set(_TOP_LEVEL_READ.findall(path.read_text(encoding="utf-8")))
    missing = sorted(read - top_level_training_keys() - _NOT_TRAINING_KNOBS)
    assert not missing, missing
