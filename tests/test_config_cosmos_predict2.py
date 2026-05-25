"""Validation for cosmos_predict2 / anima model configs."""

import pytest

from renga_flow.config.validation import ConfigValidationError, validate_config


def _cosmos_config(**model_extra):
    cfg = {
        "dataset": "examples/minimal_cosmos_predict2_dataset.toml",
        "model": {
            "type": "cosmos_predict2",
            "dtype": "bfloat16",
            "transformer_path": "t.safetensors",
            "vae_path": "v.safetensors",
            "llm_path": "l.safetensors",
            **model_extra,
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }
    return cfg


def test_validate_cosmos_predict2_minimal_passes():
    validate_config(_cosmos_config())


def test_validate_anima_requires_llm_path():
    cfg = _cosmos_config()
    cfg["model"]["type"] = "anima"
    del cfg["model"]["llm_path"]
    with pytest.raises(ConfigValidationError, match="llm_path"):
        validate_config(cfg)


def test_validate_cosmos_requires_llm_or_t5():
    cfg = _cosmos_config()
    del cfg["model"]["llm_path"]
    with pytest.raises(ConfigValidationError, match="llm_path|t5_path"):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["transformer_path", "vae_path"])
def test_validate_cosmos_missing_paths(key):
    cfg = _cosmos_config()
    del cfg["model"][key]
    with pytest.raises(ConfigValidationError, match=key):
        validate_config(cfg)


def test_validate_cosmos_with_t5_path_passes():
    cfg = _cosmos_config()
    del cfg["model"]["llm_path"]
    cfg["model"]["t5_path"] = "t5.safetensors"
    validate_config(cfg)
