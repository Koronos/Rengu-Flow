"""Tests for repo-root .env loading and model path overrides."""



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


def test_run_prepared_applies_externally_set_env_model_paths(tmp_path, monkeypatch):
    """The trainer honors RENGU_*_PATH already present in its environment (exported by
    the smoke scripts from .env); it never reads .env itself, so normal runs are
    unaffected. Regressed in 31686ee: smoke fixtures without [model] paths failed
    validation even with the vars exported."""
    import pytest

    try:
        from rengu_flow.main import parse_args, run_prepared
    except ImportError as e:
        pytest.skip(f"Cannot import rengu_flow.main: {e}")

    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("RENGU_SDXL_CHECKPOINT_PATH", str(ckpt))
    config_file = tmp_path / "train.toml"
    config_file.write_text(
        "\n".join(
            [
                'dataset = "examples/minimal_dataset.toml"',
                'output_dir = "output"',
                "[model]",
                'type = "sdxl"',
                'dtype = "bfloat16"',
                "[adapter]",
                'type = "lycoris_loha"',
                "rank = 8",
                "[optimizer]",
                'type = "adamw"',
                "lr = 1e-4",
            ]
        ),
        encoding="utf-8",
    )
    # validate-only exercises load_config -> env apply -> defaults -> validate_config;
    # without the env application this raises SystemExit("Config validation failed ...").
    run_prepared(parse_args(["--config", str(config_file), "--validate-only"]))
