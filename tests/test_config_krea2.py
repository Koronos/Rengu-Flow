"""Validation for krea2 model configs."""

import pytest

from rengu_flow.config import set_config_defaults
from rengu_flow.config.validation import ConfigValidationError, validate_config


def _krea2_config(**model_extra):
    cfg = {
        "dataset": "examples/minimal_krea2_dataset.toml",
        "model": {
            "type": "krea2",
            "dtype": "bfloat16",
            "checkpoint_path": "path/to/Krea-2-Raw",
            **model_extra,
        },
        "optimizer": {"type": "adamw", "lr": 1e-6},
    }
    return cfg


def test_validate_krea2_minimal_passes():
    """Full finetune (no [adapter] section)."""
    validate_config(_krea2_config())


@pytest.mark.parametrize(
    "adapter",
    [
        {"type": "lora", "rank": 16},
        {"type": "lokr", "rank": 6, "factor": -1},
        {"type": "lycoris_locon", "rank": 8},
    ],
    ids=["lora", "lokr", "lycoris_locon"],
)
def test_validate_krea2_with_adapter_passes(adapter):
    cfg = _krea2_config()
    cfg["adapter"] = adapter
    validate_config(cfg)


def test_validate_krea2_missing_checkpoint_path_raises():
    cfg = _krea2_config()
    del cfg["model"]["checkpoint_path"]
    with pytest.raises(ConfigValidationError, match="checkpoint_path"):
        validate_config(cfg)


def test_validate_krea2_component_overrides_do_not_replace_checkpoint_path():
    """transformer_path/text_encoder_path/vae_path are overrides, not substitutes:
    checkpoint_path is still the required key."""
    cfg = _krea2_config(
        transformer_path="path/to/transformer",
        text_encoder_path="path/to/text_encoder",
        vae_path="path/to/vae",
    )
    del cfg["model"]["checkpoint_path"]
    with pytest.raises(ConfigValidationError, match="checkpoint_path"):
        validate_config(cfg)


def test_krea2_defaults_after_set_config_defaults():
    cfg = _krea2_config()
    cfg["preview"] = {}
    set_config_defaults(cfg)
    assert cfg["model"]["cache_text_embeddings"] is True
    assert cfg["preview"]["num_inference_steps"] == 28
    assert cfg["preview"]["guidance_scale"] == 4.5
    assert cfg["model"]["transformer_4bit"] is False


def test_krea2_transformer_4bit_and_fp8_matmul_are_mutually_exclusive():
    cfg = _krea2_config(transformer_4bit=True, transformer_fp8_matmul=True)
    with pytest.raises(ConfigValidationError, match="mutually"):
        set_config_defaults(cfg)
