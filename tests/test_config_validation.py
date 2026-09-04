"""Tests for config validation: validate_config."""

import pytest

from rengu_flow.config.validation import (
    ConfigValidationError,
    collect_validation_warnings,
    validate_config,
)


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


@pytest.mark.parametrize("optim_type", ["msam", "nekaon", "schedulefree", "lookahead"])
def test_gradient_release_lookahead_optimizer_emits_warning(minimal_config, optim_type):
    minimal_config["optimizer"]["gradient_release"] = True
    minimal_config["optimizer"]["type"] = optim_type
    warnings = collect_validation_warnings(minimal_config)
    assert len(warnings) == 1
    assert "gradient_release" in warnings[0]
    assert optim_type in warnings[0]
    assert "true iterate" in warnings[0]


def test_gradient_release_plain_optimizer_no_warning(minimal_config):
    minimal_config["optimizer"]["gradient_release"] = True
    assert collect_validation_warnings(minimal_config) == []


def test_gradient_release_without_flag_no_warning(minimal_config):
    minimal_config["optimizer"]["type"] = "nekaon"
    assert collect_validation_warnings(minimal_config) == []


def test_run_prepared_logs_gradient_release_lookahead_warning(tmp_path):
    """CLI validate-only must emit config advisories through the project logger."""
    try:
        from rengu_flow.main import parse_args, run_prepared
    except ImportError as e:
        import pytest

        pytest.skip(f"Cannot import rengu_flow.main: {e}")

    config_file = tmp_path / "train.toml"
    config_file.write_text(
        "\n".join(
            [
                'dataset = "examples/minimal_dataset.toml"',
                'output_dir = "output"',
                "[model]",
                'type = "sdxl"',
                'dtype = "bfloat16"',
                'checkpoint_path = "/tmp/x.safetensors"',
                "[optimizer]",
                'type = "nekaon"',
                "lr = 1e-4",
                "gradient_release = true",
            ]
        ),
        encoding="utf-8",
    )
    import io
    import logging

    from rengu_flow.utils.logging import logger

    # The project logger binds sys.stdout at import time, so capsys cannot see it:
    # attach a temporary handler instead.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        run_prepared(parse_args(["--config", str(config_file), "--validate-only"]))
    finally:
        logger.removeHandler(handler)
    assert "gradient_release" in buf.getvalue()
