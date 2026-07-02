"""Integration tests: example configs load/validate; main --validate-only."""

import subprocess
import sys
from pathlib import Path

import pytest

from rengu_flow.config import load_config, load_dataset_config, set_config_defaults, validate_config


EXAMPLE_CONFIGS = [
    "minimal_config.toml",
    "minimal_config_lora_sdxl.toml",
    "minimal_config_lokr_vendored.toml",
    "full_model_sdxl.toml",
    "full_model_sdxl_unet_only.toml",
    "minimal_config_cosmos_predict2_lora.toml",
    "minimal_config_cosmos_predict2_lokr.toml",
    "minimal_config_cosmos_predict2_finetune.toml",
    "config_with_preview.toml",
    "config_with_eval_and_monitoring.toml",
    "config_oom_skip.toml",
]


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("config_name", EXAMPLE_CONFIGS)
def test_example_config_loads_validates_and_dataset_if_present(examples_dir, repo_root, config_name):
    """Load config, set defaults, validate; if dataset key exists and file exists, load dataset config."""
    config_path = examples_dir / config_name
    if not config_path.exists():
        pytest.skip(f"Example {config_name} not found")
    config = load_config(config_path)
    set_config_defaults(config)
    validate_config(config)
    if config.get("dataset"):
        dataset_path = repo_root / config["dataset"]
        if dataset_path.exists():
            config["dataset"] = str(dataset_path)
            loaded = load_dataset_config(config)
            assert loaded is not None


def test_main_validate_only_cosmos_predict2_exits_zero(repo_root, tmp_path):
    config_content = """
dataset = "examples/minimal_cosmos_predict2_dataset.toml"

[model]
type = "cosmos_predict2"
dtype = "bfloat16"
transformer_path = "path/to/transformer.safetensors"
vae_path = "path/to/vae.safetensors"
llm_path = "path/to/llm.safetensors"

[optimizer]
type = "adamw"
lr = 1.0e-4
"""
    config_path = tmp_path / "validate_cosmos.toml"
    config_path.write_text(config_content.strip())
    result = subprocess.run(
        [sys.executable, "-m", "rengu_flow.main", "--config", str(config_path), "--validate-only"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def test_main_validate_only_exits_zero(repo_root, tmp_path):
    """With --validate-only, main loads config, validates, and exits with code 0 (no dataset load)."""
    config_content = """
dataset = "nonexistent_dataset.toml"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "path/to/sdxl.safetensors"

[optimizer]
type = "adamw"
lr = 1.0e-4
"""
    config_path = tmp_path / "validate_only.toml"
    config_path.write_text(config_content.strip())
    result = subprocess.run(
        [sys.executable, "-m", "rengu_flow.main", "--config", str(config_path), "--validate-only"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
