"""Tests for model registry: get_model, model_registry."""

import pytest

try:
    from rengu_flow.registry.models import get_model, model_registry
except ImportError as e:
    pytest.skip(f"Cannot import registry (diffusers/huggingface_hub): {e}", allow_module_level=True)


def test_model_registry_contains_sdxl():
    assert "sdxl" in model_registry


def test_get_model_missing_model_raises():
    config = {"optimizer": {"type": "adamw"}, "dataset": "x.toml"}
    with pytest.raises(KeyError) as exc_info:
        get_model(config)
    assert "model" in str(exc_info.value).lower()


def test_get_model_missing_type_raises(minimal_config):
    del minimal_config["model"]["type"]
    with pytest.raises(KeyError) as exc_info:
        get_model(minimal_config)
    assert "type" in str(exc_info.value).lower()


def test_get_model_unknown_type_raises(minimal_config):
    minimal_config["model"]["type"] = "unknown"
    with pytest.raises(ValueError) as exc_info:
        get_model(minimal_config)
    assert "unknown" in str(exc_info.value).lower() or "sdxl" in str(exc_info.value).lower()


def test_get_model_sdxl_returns_instance_without_loading(minimal_config_copy):
    """After R1 (lazy load), get_model with type sdxl returns an instance without loading checkpoint."""
    from rengu_flow.config.defaults import set_config_defaults

    set_config_defaults(minimal_config_copy)
    model = get_model(minimal_config_copy)
    assert model is not None
    assert hasattr(model, "config")
    assert model.config["model"]["type"] == "sdxl"
    assert not hasattr(model, "_pipeline") or model._pipeline is None
