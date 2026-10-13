"""Validation and defaults for nanosaur2 model configs."""

import pytest

from rengu_flow.config.validation import ConfigValidationError, validate_config

pytestmark = pytest.mark.no_ui_db


def _config(**model_extra):
    return {
        "dataset": "examples/minimal_krea2_dataset.toml",
        "model": {
            "type": "nanosaur2",
            "dtype": "bfloat16",
            "transformer_path": "path/to/nanosaur2_diffusion_model.safetensors",
            "vae_path": "path/to/nanosaur2_vae.safetensors",
            "text_encoder_path": "path/to/nanosaur2_text_encoder.safetensors",
            **model_extra,
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }


@pytest.mark.parametrize(
    "adapter",
    [None, {"type": "lora", "rank": 16}, {"type": "lokr", "rank": 8, "factor": -1}, {"type": "lycoris_locon", "rank": 8}],
    ids=["full_finetune", "lora", "lokr", "lycoris_locon"],
)
def test_validate_passes(adapter):
    cfg = _config()
    if adapter:
        cfg["adapter"] = adapter
    validate_config(cfg)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"sparse_skip_prob": -0.1}, "sparse_skip_prob"),
        ({"sparse_skip_prob": 1.5}, "sparse_skip_prob"),
        ({"sparse_skip_prob_uncond": -1}, "sparse_skip_prob_uncond"),
        ({"sparse_skip_prob_uncond": 2}, "sparse_skip_prob_uncond"),
        ({"timestep_sample_method": "beta"}, "timestep_sample_method"),
        ({"shift": 0}, "shift"),
    ],
)
def test_invalid_values_raise(extra, match):
    with pytest.raises(ConfigValidationError, match=match):
        validate_config(_config(**extra))


def test_block_swap_is_not_offered():
    cfg = _config()
    cfg["blocks_to_swap"] = 4
    with pytest.raises(ConfigValidationError, match="block_swap"):
        validate_config(cfg)


def test_text_cache_is_keyed_by_the_encoder_file(tmp_path):
    from rengu_flow.registry.model_capabilities import get_capability

    te = tmp_path / "nanosaur2_text_encoder.safetensors"
    te.write_bytes(b"x")
    identity = get_capability("nanosaur2").text_cache_identity({"text_encoder_path": str(te)})
    assert identity["text_encoder_file"].startswith("text_encoder_path:1:")


def test_pipeline_rejects_out_of_range_path_drop_rates(tmp_path):
    import torch

    from rengu_flow.model.nanosaur2.pipeline import Nanosaur2Pipeline

    with pytest.raises(ConfigValidationError, match="sparse_skip_prob_uncond"):
        Nanosaur2Pipeline({"model": {"type": "nanosaur2", "dtype": torch.float32, "sparse_skip_prob_uncond": 1.5}})


# ---- smoke model paths (local_env) ----------------------------------------------------------

ENV = {
    "transformer_path": "RENGU_NANOSAUR2_TRANSFORMER_PATH",
    "vae_path": "RENGU_NANOSAUR2_VAE_PATH",
    "text_encoder_path": "RENGU_NANOSAUR2_TEXT_ENCODER_PATH",
}


def test_smoke_fixture_skips_with_77_without_weights(monkeypatch):
    from pathlib import Path

    from rengu_flow.config import local_env

    for var in ENV.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(local_env, "load_repo_dotenv", lambda *a, **k: False)
    fixture = Path(__file__).parent / "fixtures" / "smoke" / "train_nanosaur2.toml"
    for name in ("train_nanosaur2.toml", "train_nanosaur2_full.toml"):
        with pytest.raises(SystemExit) as exc:
            local_env.check_config_model_paths(fixture.with_name(name))
        assert exc.value.code == local_env.SMOKE_SKIP_EXIT
