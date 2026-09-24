"""Validation for krea2 model configs."""

import pytest

from rengu_flow.config import set_config_defaults
from rengu_flow.config.validation import ConfigValidationError, validate_config


def _krea2_config(**model_extra):
    cfg = {
        "dataset": "examples/minimal_krea2_dataset.toml",
        "model": {
            "type": "krea2",
            "dtype": "bfloat16",
            "transformer_path": "path/to/krea2_raw_bf16.safetensors",
            "vae_path": "path/to/qwen_image_vae.safetensors",
            "text_encoder_path": "path/to/qwen3vl_4b_bf16.safetensors",
            **model_extra,
        },
        "optimizer": {"type": "adamw", "lr": 1e-6},
    }
    return cfg


def test_validate_krea2_minimal_passes():
    """Full finetune (no [adapter] section)."""
    validate_config(_krea2_config())


@pytest.mark.parametrize(
    "adapter",
    [
        {"type": "lora", "rank": 16},
        {"type": "lokr", "rank": 6, "factor": -1},
        {"type": "lycoris_locon", "rank": 8},
    ],
    ids=["lora", "lokr", "lycoris_locon"],
)
def test_validate_krea2_with_adapter_passes(adapter):
    cfg = _krea2_config()
    cfg["adapter"] = adapter
    validate_config(cfg)


@pytest.mark.parametrize("missing_key", ["transformer_path", "vae_path", "text_encoder_path"])
def test_validate_krea2_missing_component_path_raises(missing_key):
    """Each per-component path is its own one_of([<component>_path, checkpoint_path]) group:
    dropping one without setting checkpoint_path leaves that component unresolved."""
    cfg = _krea2_config()
    del cfg["model"][missing_key]
    with pytest.raises(ConfigValidationError, match=missing_key):
        validate_config(cfg)


def test_validate_krea2_checkpoint_path_only_passes():
    """A full diffusers-layout checkpoint_path alone satisfies all three one_of groups."""
    cfg = _krea2_config()
    del cfg["model"]["transformer_path"]
    del cfg["model"]["vae_path"]
    del cfg["model"]["text_encoder_path"]
    cfg["model"]["checkpoint_path"] = "path/to/Krea-2-Raw"
    validate_config(cfg)


def test_validate_krea2_checkpoint_path_with_transformer_override_passes():
    """checkpoint_path plus a per-component override still validates (the override wins for
    that component; checkpoint_path fills the rest)."""
    cfg = _krea2_config()
    cfg["model"]["checkpoint_path"] = "path/to/Krea-2-Raw"
    validate_config(cfg)


def test_krea2_defaults_after_set_config_defaults():
    cfg = _krea2_config()
    cfg["preview"] = {}
    set_config_defaults(cfg)
    assert cfg["model"]["cache_text_embeddings"] is True
    assert cfg["preview"]["num_inference_steps"] == 28
    assert cfg["preview"]["guidance_scale"] == 4.5
    assert cfg["model"]["transformer_4bit"] is False


def test_krea2_4bit_with_block_swap_defaults_to_reentrant_ac():
    """bnb autograd pins packed weights under non-reentrant AC, defeating swap eviction
    (measured 12.75 vs 6.3 GiB peak); the combo defaults to reentrant checkpointing."""
    cfg = _krea2_config(transformer_4bit=True)
    cfg["blocks_to_swap"] = 16
    cfg["activation_checkpointing"] = True
    set_config_defaults(cfg)
    assert cfg["reentrant_activation_checkpointing"] is True

    cfg = _krea2_config(transformer_4bit=True)  # no swap: default untouched (False)
    set_config_defaults(cfg)
    assert cfg["reentrant_activation_checkpointing"] is False


def test_krea2_transformer_4bit_and_fp8_matmul_are_mutually_exclusive():
    cfg = _krea2_config(transformer_4bit=True, transformer_fp8_matmul=True)
    with pytest.raises(ConfigValidationError, match="mutually"):
        set_config_defaults(cfg)


def _krea2_adapter_config(**model_extra):
    cfg = _krea2_config(**model_extra)
    cfg["adapter"] = {"type": "lokr", "rank": 6}
    return cfg


@pytest.mark.parametrize(
    "model_type,has_fp8_dtype,has_guidance",
    [("krea2", False, False), ("cosmos_predict2", True, False), ("sdxl", False, True)],
)
def test_model_scoped_defaults(model_type, has_fp8_dtype, has_guidance):
    """fp8_matmul_dtype (cosmos-only reader), model.guidance (sdxl legacy key) and
    preview_offload_dit_for_decode (cosmos preview only) are defaulted only where used."""
    cfg = {"model": {"type": model_type, "dtype": "bfloat16"}, "preview": {}}
    set_config_defaults(cfg)
    assert ("fp8_matmul_dtype" in cfg["model"]) is has_fp8_dtype
    assert ("guidance" in cfg["model"]) is has_guidance
    assert ("preview_offload_dit_for_decode" in cfg["preview"]) is (model_type == "cosmos_predict2")


@pytest.mark.parametrize(
    "top,expected",
    [
        ({"compile": True, "compile_scope": "block", "activation_checkpointing": True}, True),
        ({"compile": True, "compile_scope": "model", "activation_checkpointing": True}, False),
        ({"compile": False, "compile_scope": "block", "activation_checkpointing": True}, False),
        ({"compile": True, "compile_scope": "block", "activation_checkpointing": False}, False),
    ],
    ids=["fp8_block_ac", "model_scope", "no_compile", "no_ac"],
)
def test_fp8_block_compile_defaults_to_reentrant_ac(top, expected):
    cfg = _krea2_adapter_config(transformer_fp8_matmul=True)
    cfg.update(top)
    set_config_defaults(cfg)
    assert cfg["reentrant_activation_checkpointing"] is expected


@pytest.mark.parametrize("explicit,warns", [(False, True), (True, False)])
def test_explicit_non_reentrant_in_fp8_block_combo_warns(explicit, warns):
    from rengu_flow.config.validation import collect_validation_warnings

    cfg = _krea2_adapter_config(transformer_fp8_matmul=True)
    cfg.update(compile=True, compile_scope="block", activation_checkpointing=True)
    cfg["reentrant_activation_checkpointing"] = explicit
    set_config_defaults(cfg)
    assert cfg["reentrant_activation_checkpointing"] is explicit  # explicit value kept
    got = [w for w in collect_validation_warnings(cfg) if "reentrant" in w]
    assert bool(got) is warns


@pytest.mark.parametrize("quant_key", ["transformer_4bit", "transformer_fp8_matmul"])
def test_full_finetune_rejects_quantized_base(quant_key):
    cfg = _krea2_config(**{quant_key: True})  # no [adapter]
    with pytest.raises(ConfigValidationError, match=f"model.{quant_key} quantizes the frozen base"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"tread": {"start_block": 2}}, "needs drop_ratio"),
        ({"tread": {"drop_ratio": 1.0}}, "drop_ratio must be in"),
        ({"tread": {"drop_ratio": 0.0}}, "drop_ratio must be in"),
        ({"tread": {"drop_ratio": 0.5, "disable_after_frac": 0.0}}, "disable_after_frac must be in"),
        ({"tread": {"drop_ratio": 0.5, "disable_after_frac": 1.5}}, "disable_after_frac must be in"),
        ({"tread": {"drop_ratio": 0.5, "start_block": 0}}, "tread route"),
        ({"tread": {"drop_ratio": 0.5, "end_block": 27}}, "tread route"),
        ({"tread": {"drop_ratio": 0.5, "start_block": 10, "end_block": 5}}, "tread route"),
        ({"model_timestep_sample_method": "cosine"}, "timestep_sample_method must be one of"),
        ({"model_shift": 0}, "model.shift must be > 0"),
        ({"model_shift": -1.0}, "model.shift must be > 0"),
        ({"model_max_sequence_length": 0}, "max_sequence_length must be >= 1"),
        ({"model_max_sequence_length": 1.5}, "max_sequence_length must be an integer"),
    ],
)
def test_config_time_errors(extra, match):
    cfg = _krea2_adapter_config()
    for key, value in extra.items():
        if key.startswith("model_"):
            cfg["model"][key[len("model_"):]] = value
        else:
            cfg[key] = value
    with pytest.raises(ConfigValidationError, match=match):
        validate_config(cfg)


@pytest.mark.parametrize(
    "extra",
    [
        {"tread": {"drop_ratio": 0.5, "disable_after_frac": 0.85}},
        {"tread": {"drop_ratio": 0.5, "start_block": 2, "end_block": 25}},
        {"model_shift": 3.0, "model_timestep_sample_method": "uniform", "model_max_sequence_length": 256},
    ],
)
def test_valid_values_pass(extra):
    cfg = _krea2_adapter_config()
    for key, value in extra.items():
        if key.startswith("model_"):
            cfg["model"][key[len("model_"):]] = value
        else:
            cfg[key] = value
    validate_config(cfg)


@pytest.mark.parametrize(
    "model_extra,expected",
    [
        ({}, {}),
        ({"max_sequence_length": 512}, {}),
        ({"max_sequence_length": 256, "tokenizer_path": "tok"}, {"max_sequence_length": "256"}),
    ],
    ids=["defaults", "explicit_default", "changed"],
)
def test_text_cache_identity(model_extra, expected):
    """Text-encoder settings key the text-embedding cache; defaults are left out."""
    from rengu_flow.registry.model_capabilities import get_capability

    model = _krea2_config(**model_extra)["model"]
    assert get_capability("krea2").text_cache_identity(model) == expected


def test_text_cache_identity_empty_for_models_without_keys():
    """Models that declare no text_cache_keys keep their existing cache key unchanged."""
    from rengu_flow.data.dataset import text_cache_fingerprint_args

    cfg = {"model": {"type": "cosmos_predict2", "dtype": "bfloat16", "llm_path": "x"}}
    assert text_cache_fingerprint_args(cfg) == []
    assert text_cache_fingerprint_args({}) == []
