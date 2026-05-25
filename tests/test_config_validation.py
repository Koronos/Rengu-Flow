"""Tests for config validation: validate_config."""

import pytest

from renga_flow.config.validation import ConfigValidationError, validate_config


def test_validate_config_minimal_passes(minimal_config):
    validate_config(minimal_config)


@pytest.mark.parametrize("section", ["model", "optimizer", "dataset"])
def test_validate_config_missing_section_raises(minimal_config, section):
    del minimal_config[section]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(minimal_config)
    assert section in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()


@pytest.mark.parametrize("key", ["type", "dtype"])
def test_validate_config_model_missing_key_raises(minimal_config, key):
    del minimal_config["model"][key]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(minimal_config)
    assert key in str(exc_info.value).lower()


def test_validate_config_optimizer_missing_type(minimal_config):
    del minimal_config["optimizer"]["type"]
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(minimal_config)
    assert "type" in str(exc_info.value).lower()


def test_validate_config_adapter_missing_type(minimal_config):
    minimal_config["adapter"] = {"rank": 8}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(minimal_config)
    assert "type" in str(exc_info.value).lower()


def test_validate_config_adapter_invalid_type(minimal_config):
    minimal_config["adapter"] = {"type": "other", "rank": 8}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(minimal_config)
    assert "lora" in str(exc_info.value).lower() or "lokr" in str(exc_info.value).lower()


def test_validate_config_adapter_missing_rank_and_dim(minimal_config):
    minimal_config["adapter"] = {"type": "lora"}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(minimal_config)
    assert "rank" in str(exc_info.value).lower() or "dim" in str(exc_info.value).lower()


def test_validate_config_gradient_release_requires_pipeline_stages_one(minimal_config):
    minimal_config["optimizer"]["gradient_release"] = True
    minimal_config["pipeline_stages"] = 2
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(minimal_config)
    assert "gradient_release" in str(exc_info.value).lower()


@pytest.mark.parametrize("adapter", [{"type": "lora", "rank": 8}, {"type": "lokr", "dim": 8}])
def test_validate_config_adapter_valid_passes(minimal_config, adapter):
    minimal_config["adapter"] = adapter
    validate_config(minimal_config)
