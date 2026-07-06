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
            "transformer_path": "path/to/krea2_raw_bf16.safetensors",
            "vae_path": "path/to/qwen_image_vae.safetensors",
            "text_encoder_path": "path/to/qwen3vl_4b_bf16.safetensors",
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


@pytest.mark.parametrize("missing_key", ["transformer_path", "vae_path", "text_encoder_path"])
def test_validate_krea2_missing_component_path_raises(missing_key):
    """Each per-component path is its own one_of([<component>_path, checkpoint_path]) group:
    dropping one without setting checkpoint_path leaves that component unresolved."""
    cfg = _krea2_config()
    del cfg["model"][missing_key]
    with pytest.raises(ConfigValidationError, match=missing_key):
        validate_config(cfg)


def test_validate_krea2_checkpoint_path_only_passes():
    """A full diffusers-layout checkpoint_path alone satisfies all three one_of groups."""
    cfg = _krea2_config()
    del cfg["model"]["transformer_path"]
    del cfg["model"]["vae_path"]
    del cfg["model"]["text_encoder_path"]
    cfg["model"]["checkpoint_path"] = "path/to/Krea-2-Raw"
    validate_config(cfg)


def test_validate_krea2_checkpoint_path_with_transformer_override_passes():
    """checkpoint_path plus a per-component override still validates (the override wins for
    that component; checkpoint_path fills the rest)."""
    cfg = _krea2_config()
    cfg["model"]["checkpoint_path"] = "path/to/Krea-2-Raw"
    validate_config(cfg)


def test_krea2_defaults_after_set_config_defaults():
    cfg = _krea2_config()
    cfg["preview"] = {}
    set_config_defaults(cfg)
    assert cfg["model"]["cache_text_embeddings"] is True
    assert cfg["preview"]["num_inference_steps"] == 28
    assert cfg["preview"]["guidance_scale"] == 4.5
    assert cfg["model"]["transformer_4bit"] is False


def test_krea2_4bit_with_block_swap_defaults_to_reentrant_ac():
    """bnb autograd pins packed weights under non-reentrant AC, defeating swap eviction
    (measured 12.75 vs 6.3 GiB peak); the combo defaults to reentrant checkpointing."""
    cfg = _krea2_config(transformer_4bit=True)
    cfg["blocks_to_swap"] = 16
    cfg["activation_checkpointing"] = True
    set_config_defaults(cfg)
    assert cfg["reentrant_activation_checkpointing"] is True

    cfg = _krea2_config(transformer_4bit=True)  # no swap: default untouched (False)
    set_config_defaults(cfg)
    assert cfg["reentrant_activation_checkpointing"] is False


def test_krea2_transformer_4bit_and_fp8_matmul_are_mutually_exclusive():
    cfg = _krea2_config(transformer_4bit=True, transformer_fp8_matmul=True)
    with pytest.raises(ConfigValidationError, match="mutually"):
        set_config_defaults(cfg)
