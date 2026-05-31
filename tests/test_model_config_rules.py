"""Model capability rules stay aligned with UI visibility."""

import pytest

from rengu_flow.config.validation import ConfigValidationError, validate_config
from rengu_flow.registry.model_capabilities import get_capability
from rengu_flow.registry.model_config_rules import (
    one_of_groups,
    required_model_keys,
    validate_config_model_rules,
    validate_training_keys_for_model,
)


def test_sdxl_required_keys_from_capability() -> None:
    cap = get_capability("sdxl")
    assert cap is not None
    assert "checkpoint_path" in required_model_keys(cap)
    assert "transformer_path" not in required_model_keys(cap)


def test_cosmos_required_keys_from_capability() -> None:
    cap = get_capability("cosmos_predict2")
    assert cap is not None
    keys = required_model_keys(cap)
    assert "transformer_path" in keys
    assert "vae_path" in keys
    assert "llm_path" not in keys
    assert ["llm_path", "t5_path"] in one_of_groups(cap)


def test_blocks_to_swap_allowed_for_sdxl() -> None:
    cfg = {
        "dataset": "d.toml",
        "model": {"type": "sdxl", "dtype": "bfloat16", "checkpoint_path": "x.safetensors"},
        "optimizer": {"type": "adamw"},
        "blocks_to_swap": 4,
        "adapter": {"type": "lora", "rank": 8},
    }
    validate_training_keys_for_model(cfg)


def test_blocks_to_swap_allowed_for_cosmos() -> None:
    cfg = {
        "dataset": "d.toml",
        "model": {
            "type": "cosmos_predict2",
            "dtype": "bfloat16",
            "transformer_path": "t",
            "vae_path": "v",
            "llm_path": "l",
        },
        "optimizer": {"type": "adamw"},
        "blocks_to_swap": 2,
    }
    validate_config_model_rules(cfg)


def test_block_swap_prefetch_allowed_for_sdxl() -> None:
    cfg = {
        "dataset": "d.toml",
        "model": {"type": "sdxl", "dtype": "bfloat16", "checkpoint_path": "x.safetensors"},
        "optimizer": {"type": "adamw", "gradient_release": True},
        "blocks_to_swap": 6,
        "block_swap_prefetch": True,
    }
    validate_training_keys_for_model(cfg)


def test_validate_config_uses_rules_for_sdxl_checkpoint() -> None:
    cfg = {
        "dataset": "d.toml",
        "model": {"type": "sdxl", "dtype": "bfloat16"},
        "optimizer": {"type": "adamw"},
    }
    with pytest.raises(ConfigValidationError, match="checkpoint_path"):
        validate_config(cfg)
