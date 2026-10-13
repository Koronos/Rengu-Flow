"""Validation and defaults for qwen_image21 model configs."""

import pytest

from rengu_flow.config.validation import ConfigValidationError, validate_config

pytestmark = pytest.mark.no_ui_db


def _config(**model_extra):
    return {
        "dataset": "examples/minimal_qwen_image21_dataset.toml",
        "model": {
            "type": "qwen_image21",
            "dtype": "bfloat16",
            "diffusers_path": "path/to/Qwen-Image-2.1",
            **model_extra,
        },
        "optimizer": {"type": "adamw", "lr": 1e-6},
    }


@pytest.mark.parametrize(
    "adapter",
    [
        {"type": "lora", "rank": 16},
        {"type": "lokr", "rank": 6, "factor": -1},
        {"type": "lycoris_locon", "rank": 8},
        {"type": "lycoris_loha", "rank": 8},
    ],
    ids=["lora", "lokr", "lycoris_locon", "lycoris_loha"],
)
def test_validate_with_adapter_passes(adapter):
    cfg = _config()
    cfg["adapter"] = adapter
    validate_config(cfg)


def test_block_swap_is_supported():
    cfg = _config(transformer_fp8_matmul=True)
    cfg["blocks_to_swap"] = 24
    cfg["adapter"] = {"type": "lora", "rank": 16}
    validate_config(cfg)


def test_tread_is_rejected_at_validation():
    cfg = _config()
    cfg["tread"] = {"drop_ratio": 0.5}
    with pytest.raises(ConfigValidationError, match="tread"):
        validate_config(cfg)


@pytest.mark.parametrize("shift,ok", [(0, False), (-1.0, False), (3.0, True)])
def test_shift_must_be_positive(shift, ok):
    """0 would silently fall back to the dynamic shift (shift_timesteps treats it as unset)."""
    cfg = _config(shift=shift)
    if ok:
        validate_config(cfg)
    else:
        with pytest.raises(ConfigValidationError, match="model.shift must be > 0"):
            validate_config(cfg)
