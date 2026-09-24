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


def test_apply_krea2_paths(monkeypatch):
    for key, value in {
        "RENGU_KREA2_TRANSFORMER_PATH": "/t.safetensors",
        "RENGU_KREA2_VAE_PATH": "/v.safetensors",
        "RENGU_KREA2_TEXT_ENCODER_PATH": "/te.safetensors",
        "RENGU_KREA2_CHECKPOINT_PATH": "/Krea-2-Raw",
    }.items():
        monkeypatch.setenv(key, value)
    config = {"model": {"type": "krea2", "dtype": "bfloat16"}}
    apply_model_paths_from_env(config)
    assert config["model"]["transformer_path"] == "/t.safetensors"
    assert config["model"]["vae_path"] == "/v.safetensors"
    assert config["model"]["text_encoder_path"] == "/te.safetensors"
    assert config["model"]["checkpoint_path"] == "/Krea-2-Raw"


def test_model_path_errors_krea2_one_of(tmp_path):
    """krea2 components are one_of(<component>_path, checkpoint_path) and may be folders."""
    ckpt_dir = tmp_path / "Krea-2-Raw"
    ckpt_dir.mkdir()
    te_dir = tmp_path / "text_encoder"
    te_dir.mkdir()
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"x")

    missing = model_path_errors({"model": {"type": "krea2"}})
    assert len(missing) == 3
    assert all("checkpoint_path" in e for e in missing)

    assert model_path_errors({"model": {"type": "krea2", "checkpoint_path": str(ckpt_dir)}}) == []

    components = {"transformer_path": str(dit), "vae_path": str(dit), "text_encoder_path": str(te_dir)}
    assert model_path_errors({"model": {"type": "krea2", **components}}) == []

    # A partial set is fine when checkpoint_path fills the rest.
    partial = {"type": "krea2", "transformer_path": str(dit), "checkpoint_path": str(ckpt_dir)}
    assert model_path_errors({"model": partial}) == []

    bad = {"type": "krea2", **components, "vae_path": str(tmp_path / "nope.safetensors")}
    errors = model_path_errors({"model": bad})
    assert errors == [f"[model].vae_path not found: {tmp_path / 'nope.safetensors'}"]


def test_check_config_model_paths_cli(tmp_path, monkeypatch, capsys):
    """`python -m rengu_flow.config.local_env CONFIG` (smoke pre-check) exits 77 (skip) on
    missing paths."""
    import pytest

    from rengu_flow.config.local_env import SMOKE_SKIP_EXIT, check_config_model_paths

    for key in ("RENGU_KREA2_TRANSFORMER_PATH", "RENGU_KREA2_VAE_PATH",
                "RENGU_KREA2_TEXT_ENCODER_PATH", "RENGU_KREA2_CHECKPOINT_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("rengu_flow.config.local_env.load_repo_dotenv", lambda *a, **k: False)
    config_file = tmp_path / "train.toml"
    config_file.write_text('[model]\ntype = "krea2"\ndtype = "bfloat16"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        check_config_model_paths(config_file)
    assert exc.value.code == SMOKE_SKIP_EXIT == 77
    assert "checkpoint_path" in capsys.readouterr().err

    ckpt_dir = tmp_path / "Krea-2-Raw"
    ckpt_dir.mkdir()
    monkeypatch.setenv("RENGU_KREA2_CHECKPOINT_PATH", str(ckpt_dir))
    check_config_model_paths(config_file)  # no SystemExit
