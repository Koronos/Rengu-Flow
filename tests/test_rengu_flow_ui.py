"""Tests for UI run-staging validation helpers."""

from rengu_flow_ui import run_staging


def test_validate_minimal_toml() -> None:
    bad = "not toml"
    r = run_staging.validate_toml_text(bad)
    assert r["ok"] is False
    minimal = """
dataset = "examples/minimal_dataset.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"
[optimizer]
type = "adamw"
"""
    r2 = run_staging.validate_toml_text(minimal)
    assert r2["ok"] is True
