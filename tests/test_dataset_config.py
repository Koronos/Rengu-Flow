"""Tests for dataset config loading and validation."""

from pathlib import Path

import pytest

from renga_flow.config.loader import load_dataset_config, load_eval_dataset_config
from renga_flow.data.dataset_config import (
    DatasetConfigError,
    validate_dataset_config_for_real_data,
)


@pytest.mark.parametrize("config", [
    {"model": {}, "optimizer": {}},
    {"model": {}, "optimizer": {}, "dataset": None},
], ids=["no_dataset_key", "dataset_none"])
def test_load_dataset_config_returns_none_when_no_dataset(config):
    """When config has no dataset key or dataset is None, returns None."""
    assert load_dataset_config(config) is None


def test_load_dataset_config_missing_file(minimal_config):
    minimal_config["dataset"] = "/nonexistent/dataset.toml"
    with pytest.raises(FileNotFoundError):
        load_dataset_config(minimal_config)


def test_load_dataset_config_valid_file(tmp_path, minimal_config):
    dataset_path = tmp_path / "dataset.toml"
    dataset_path.write_text("resolutions = [[1024, 1024]]\ndirectories = []")
    minimal_config["dataset"] = str(dataset_path)
    loaded = load_dataset_config(minimal_config)
    assert loaded is not None
    assert "resolutions" in loaded
    assert loaded["resolutions"] == [[1024, 1024]]


def test_load_eval_dataset_config_from_path(tmp_path):
    eval_path = tmp_path / "eval_ds.toml"
    eval_path.write_text("resolutions = [[512, 512]]")
    name, config = load_eval_dataset_config(str(eval_path))
    assert name == "eval_eval_ds"
    assert config["resolutions"] == [[512, 512]]


def test_load_eval_dataset_config_from_dict(tmp_path):
    config_path = tmp_path / "eval.toml"
    config_path.write_text("resolutions = [[768, 768]]")
    name, config = load_eval_dataset_config({"name": "my_eval", "config": str(config_path)})
    assert name == "my_eval"
    assert config["resolutions"] == [[768, 768]]


def test_load_eval_dataset_config_missing_path():
    with pytest.raises(FileNotFoundError):
        load_eval_dataset_config("/nonexistent/eval.toml")


def test_validate_dataset_config_missing_directory():
    """validate_dataset_config_for_real_data raises when 'directory' is missing."""
    with pytest.raises(DatasetConfigError, match="directory"):
        validate_dataset_config_for_real_data({"resolutions": [1024]})


def test_validate_dataset_config_empty_directory():
    """validate_dataset_config_for_real_data raises when directory list is empty."""
    with pytest.raises(DatasetConfigError, match="non-empty"):
        validate_dataset_config_for_real_data({"directory": []})


def test_validate_dataset_config_missing_path():
    """validate_dataset_config_for_real_data raises when an entry has no 'path'."""
    with pytest.raises(DatasetConfigError, match="path"):
        validate_dataset_config_for_real_data({
            "directory": [{"num_repeats": 1}],
        })


def test_validate_tag_dropout_probability_range():
    with pytest.raises(DatasetConfigError, match="tag_dropout_probability"):
        validate_dataset_config_for_real_data({
            "directory": [{"path": "/tmp/x", "num_repeats": 1}],
            "tag_dropout_probability": 1.5,
        })


def test_validate_uncond_fraction_range():
    with pytest.raises(DatasetConfigError, match="uncond_fraction"):
        validate_dataset_config_for_real_data({
            "directory": [{"path": "/tmp/x", "num_repeats": 1}],
            "uncond_fraction": -0.1,
        })


def test_validate_dataset_config_missing_num_repeats():
    """validate_dataset_config_for_real_data raises when an entry has no 'num_repeats'."""
    with pytest.raises(DatasetConfigError, match="num_repeats"):
        validate_dataset_config_for_real_data({
            "directory": [{"path": "/some/path"}],
        })


def test_validate_dataset_config_num_repeats_zero():
    """validate_dataset_config_for_real_data raises when num_repeats <= 0."""
    with pytest.raises(DatasetConfigError, match="> 0"):
        validate_dataset_config_for_real_data({
            "directory": [{"path": "/some/path", "num_repeats": 0}],
        })


def test_validate_dataset_config_passes():
    """validate_dataset_config_for_real_data passes with valid config."""
    validate_dataset_config_for_real_data({
        "directory": [{"path": "/some/path", "num_repeats": 2}],
    })
