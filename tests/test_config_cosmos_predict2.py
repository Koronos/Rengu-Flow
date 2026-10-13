"""Validation for cosmos_predict2 model configs."""

import pytest

from rengu_flow.config.validation import ConfigValidationError, validate_config


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


def test_validate_rejects_anima_model_type():
    cfg = _cosmos_config()
    cfg["model"]["type"] = "anima"
    with pytest.raises(ConfigValidationError, match="Unknown model type"):
        validate_config(cfg)


def test_cosmos_dataset_validation_requires_frame_bucket_one(monkeypatch):
    from rengu_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline

    monkeypatch.setattr(
        CosmosPredict2Pipeline,
        "__init__",
        lambda self, config: setattr(self, "config", config),
    )
    pipe = CosmosPredict2Pipeline({"model": {}})
    with pytest.raises(ConfigValidationError, match="frame_buckets"):
        pipe.model_specific_dataset_config_validation({"frame_buckets": [4, 8]})
    pipe.model_specific_dataset_config_validation({"frame_buckets": [1, 4]})


@pytest.mark.parametrize("shift,ok", [(0, False), (-1.0, False), (3.0, True)])
def test_shift_must_be_positive(shift, ok):
    """0 would silently mean no fixed shift (flux_shift or unshifted), not a shift of 0."""
    cfg = _cosmos_config(shift=shift)
    if ok:
        validate_config(cfg)
    else:
        with pytest.raises(ConfigValidationError, match="model.shift must be > 0"):
            validate_config(cfg)
