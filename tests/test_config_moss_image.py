"""Validation and defaults for moss_image model configs."""

import pytest

from rengu_flow.config.validation import ConfigValidationError, validate_config

pytestmark = pytest.mark.no_ui_db


def _config(**model_extra):
    return {
        "dataset": "examples/minimal_krea2_dataset.toml",
        "model": {
            "type": "moss_image",
            "dtype": "bfloat16",
            "transformer_path": "path/to/moss_image_base_bf16.safetensors",
            "vae_path": "path/to/flux2-vae_small.safetensors",
            "text_encoder_path": "path/to/qwen3.5_9b_bf16.safetensors",
            **model_extra,
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }


@pytest.mark.parametrize(
    "adapter",
    [None, {"type": "lora", "rank": 32}, {"type": "lokr", "rank": 10000, "factor": 8}, {"type": "lycoris_locon", "rank": 8}],
    ids=["full_finetune", "lora", "lokr", "lycoris_locon"],
)
def test_validate_passes(adapter):
    cfg = _config()
    if adapter:
        cfg["adapter"] = adapter
    validate_config(cfg)


def test_the_bundle_folder_alone_is_enough():
    cfg = _config(diffusers_path="path/to/MOSS-Image-Base")
    for key in ("transformer_path", "vae_path", "text_encoder_path"):
        del cfg["model"][key]
    validate_config(cfg)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"timestep_sample_method": "beta"}, "timestep_sample_method"),
        ({"timestep_std": 0}, "timestep_std"),
        ({"timestep_cutoff": 1.0}, "timestep_cutoff"),
        ({"timestep_cutoff": 0}, "timestep_cutoff"),
        ({"timestep_uniform_mix": 1.5}, "timestep_uniform_mix"),
        ({"shift": 0}, "shift"),
        ({"text_encoder_offload": "sometimes"}, "text_encoder_offload"),
    ],
)
def test_invalid_values_raise(extra, match):
    with pytest.raises(ConfigValidationError, match=match):
        validate_config(_config(**extra))


@pytest.mark.parametrize("method", ["moss", "logit_normal", "uniform"])
def test_every_sample_method_validates(method):
    validate_config(_config(timestep_sample_method=method))


def test_defaults_are_the_reports_final_stage():
    from rengu_flow.config.defaults import set_config_defaults

    cfg = _config()
    cfg["preview"] = {}
    set_config_defaults(cfg)
    model = cfg["model"]
    assert model["timestep_sample_method"] == "moss"
    assert (model["timestep_mu"], model["timestep_std"], model["timestep_cutoff"], model["timestep_uniform_mix"]) == (
        -1.09,
        0.95,
        0.95,
        0.01,
    )
    assert "shift" not in model  # the report trains its final stage without a shift
    assert cfg["preview"]["num_inference_steps"] == 50 and cfg["preview"]["guidance_scale"] == 3.0
    assert cfg["preview"]["negative_prompt"] == ""


def test_text_cache_cannot_be_disabled():
    from rengu_flow.model.moss_image.pipeline import MOSSImagePipeline

    pipeline = object.__new__(MOSSImagePipeline)
    with pytest.raises(ConfigValidationError, match="cache_text_embeddings"):
        pipeline._init_pipeline({"model": {"type": "moss_image", "cache_text_embeddings": False}})


@pytest.mark.parametrize("key", ["dtype", "transformer_dtype"])
def test_fp16_fails_validation(key):
    with pytest.raises(ConfigValidationError, match=f"model.{key} = float16 is not supported for moss_image"):
        validate_config(_config(**{key: "float16"}))


@pytest.mark.parametrize("key", ["dtype", "transformer_dtype"])
def test_fp16_is_refused(key):
    import torch

    from rengu_flow.model.moss_image.pipeline import MOSSImagePipeline

    model = {"type": "moss_image", "dtype": torch.bfloat16, key: torch.float16}
    with pytest.raises(ConfigValidationError, match="float16"):
        MOSSImagePipeline({"model": model})


def test_text_cache_is_keyed_by_the_encoder_file(tmp_path):
    from rengu_flow.registry.model_capabilities import get_capability

    te = tmp_path / "qwen3.5_9b_bf16.safetensors"
    te.write_bytes(b"x")
    identity = get_capability("moss_image").text_cache_identity({"text_encoder_path": str(te)})
    assert identity["text_encoder_file"].startswith("text_encoder_path:1:")


def test_text_cache_is_keyed_by_the_tokenizer_override():
    from rengu_flow.registry.model_capabilities import get_capability

    cap = get_capability("moss_image")
    assert cap.text_cache_identity({}) == {}
    assert cap.text_cache_identity({"tokenizer_path": "my/tokenizer"}) == {"tokenizer_path": "my/tokenizer"}


def test_smoke_fixtures_skip_with_77_without_weights(monkeypatch):
    from pathlib import Path

    from rengu_flow.config import local_env

    for key in ("TRANSFORMER", "VAE", "TEXT_ENCODER", "DIFFUSERS"):
        monkeypatch.delenv(f"RENGU_MOSS_IMAGE_{key}_PATH", raising=False)
    monkeypatch.setattr(local_env, "load_repo_dotenv", lambda *a, **k: False)
    smoke = Path(__file__).parent / "fixtures" / "smoke"
    for name in ("train_moss_image.toml", "train_moss_image_lokr.toml", "train_moss_image_full.toml"):
        with pytest.raises(SystemExit) as exc:
            local_env.check_config_model_paths(smoke / name)
        assert exc.value.code == local_env.SMOKE_SKIP_EXIT
