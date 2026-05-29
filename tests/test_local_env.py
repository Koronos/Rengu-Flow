"""Tests for repo-root .env loading and model path overrides."""

import os

import pytest

from rengu_flow.config.local_env import (
    apply_model_paths_from_env,
    load_repo_dotenv,
    model_path_errors,
    parse_dotenv_line,
)


def test_parse_dotenv_line():
    assert parse_dotenv_line("# comment") is None
    assert parse_dotenv_line('export FOO="bar baz"') == ("FOO", "bar baz")
    assert parse_dotenv_line("KEY=value") == ("KEY", "value")


def test_load_repo_dotenv_and_apply_sdxl(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'RENGU_SDXL_CHECKPOINT_PATH="/tmp/my model.safetensors"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("RENGU_SDXL_CHECKPOINT_PATH", raising=False)
    assert load_repo_dotenv(env_file) is True
    config = {"model": {"type": "sdxl", "dtype": "bfloat16"}}
    applied = apply_model_paths_from_env(config)
    assert applied == ["RENGU_SDXL_CHECKPOINT_PATH"]
    assert config["model"]["checkpoint_path"] == "/tmp/my model.safetensors"


def test_apply_cosmos_paths(monkeypatch):
    monkeypatch.setenv("RENGU_COSMOS_TRANSFORMER_PATH", "/t.safetensors")
    monkeypatch.setenv("RENGU_COSMOS_VAE_PATH", "/v.safetensors")
    monkeypatch.setenv("RENGU_COSMOS_LLM_PATH", "/l.safetensors")
    config = {"model": {"type": "cosmos_predict2", "dtype": "bfloat16"}}
    apply_model_paths_from_env(config)
    assert config["model"]["transformer_path"] == "/t.safetensors"
    assert config["model"]["vae_path"] == "/v.safetensors"
    assert config["model"]["llm_path"] == "/l.safetensors"


def test_model_path_errors_after_apply(tmp_path, monkeypatch):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("RENGU_SDXL_CHECKPOINT_PATH", str(ckpt))
    config = {"model": {"type": "sdxl"}}
    apply_model_paths_from_env(config)
    assert model_path_errors(config) == []
    config["model"]["checkpoint_path"] = str(tmp_path / "missing.safetensors")
    assert any("not found" in e for e in model_path_errors(config))
