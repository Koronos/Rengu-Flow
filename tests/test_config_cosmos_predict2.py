"""Validation for cosmos_predict2 model configs."""

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


def test_validate_rejects_anima_model_type():
    cfg = _cosmos_config()
    cfg["model"]["type"] = "anima"
    with pytest.raises(ConfigValidationError, match="Unknown model type"):
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


def test_cosmos_dataset_validation_requires_frame_bucket_one(monkeypatch):
    from renga_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline

    monkeypatch.setattr(
        CosmosPredict2Pipeline,
        "__init__",
        lambda self, config: setattr(self, "config", config),
    )
    pipe = CosmosPredict2Pipeline({"model": {}})
    with pytest.raises(ConfigValidationError, match="frame_buckets"):
        pipe.model_specific_dataset_config_validation({"frame_buckets": [4, 8]})
    pipe.model_specific_dataset_config_validation({"frame_buckets": [1, 4]})
