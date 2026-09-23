"""Validation and defaults for qwen_image21 model configs."""

import pytest

from rengu_flow.config import set_config_defaults
from rengu_flow.config.validation import ConfigValidationError, validate_config

pytestmark = pytest.mark.no_ui_db


def _config(**model_extra):
    return {
        "dataset": "examples/minimal_qwen_image21_dataset.toml",
        "model": {
            "type": "qwen_image21",
            "dtype": "bfloat16",
            "diffusers_path": "path/to/Qwen-Image-2.1",
            **model_extra,
        },
        "optimizer": {"type": "adamw", "lr": 1e-6},
    }


def test_validate_minimal_full_finetune_passes():
    validate_config(_config())


@pytest.mark.parametrize(
    "adapter",
    [
        {"type": "lora", "rank": 16},
        {"type": "lokr", "rank": 6, "factor": -1},
        {"type": "lycoris_locon", "rank": 8},
        {"type": "lycoris_loha", "rank": 8},
    ],
    ids=["lora", "lokr", "lycoris_locon", "lycoris_loha"],
)
def test_validate_with_adapter_passes(adapter):
    cfg = _config()
    cfg["adapter"] = adapter
    validate_config(cfg)


def test_per_component_paths_without_diffusers_path_pass():
    cfg = _config(
        transformer_path="path/to/qwen_image_2.1_bf16.safetensors",
        vae_path="path/to/vae",
        text_encoder_path="path/to/qwen3vl_8b_bf16.safetensors",
    )
    del cfg["model"]["diffusers_path"]
    validate_config(cfg)


@pytest.mark.parametrize("missing_key", ["transformer_path", "vae_path", "text_encoder_path"])
def test_missing_component_without_diffusers_path_raises(missing_key):
    cfg = _config(transformer_path="t", vae_path="v", text_encoder_path="te")
    del cfg["model"]["diffusers_path"]
    del cfg["model"][missing_key]
    with pytest.raises(ConfigValidationError, match=missing_key):
        validate_config(cfg)


def test_block_swap_is_supported():
    cfg = _config(transformer_fp8_matmul=True)
    cfg["blocks_to_swap"] = 24
    cfg["adapter"] = {"type": "lora", "rank": 16}
    validate_config(cfg)


def test_defaults_after_set_config_defaults():
    cfg = _config()
    cfg["preview"] = {}
    set_config_defaults(cfg)
    assert cfg["model"]["cache_text_embeddings"] is True
    assert cfg["model"]["transformer_fp8_matmul"] is False
    assert cfg["model"]["transformer_4bit"] is False
    assert cfg["model"]["fp8_grad_mode"] == "bf16"
    # Reference sampling: no CFG (true_cfg_scale 1.0); previews trim 40 steps to 28.
    assert cfg["preview"]["num_inference_steps"] == 28
    assert cfg["preview"]["guidance_scale"] == 1.0
    assert cfg["preview"]["negative_prompt"] == ""
    assert cfg["preview"]["preview_offload_text_encoder"] is True


def test_krea2_preview_defaults_unchanged():
    cfg = _config()
    cfg["model"]["type"] = "krea2"
    cfg["preview"] = {}
    set_config_defaults(cfg)
    assert cfg["preview"]["num_inference_steps"] == 28
    assert cfg["preview"]["guidance_scale"] == 4.5


def test_quantization_knobs_are_mutually_exclusive():
    cfg = _config(transformer_4bit=True, transformer_fp8_matmul=True)
    with pytest.raises(ConfigValidationError, match="mutually"):
        set_config_defaults(cfg)


def test_capability_declares_adapters_features_and_groups():
    from rengu_flow.model.qwen_image21.pipeline import ADAPTER_LAYER_GROUPS
    from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES
    from rengu_flow.registry.model_capabilities import get_capability

    cap = get_capability("qwen_image21")
    assert cap.full_finetune and cap.preview
    assert cap.features == {"preview": True, "block_swap": True, "edit": True}
    assert set(cap.adapters) == {"lora", "lokr", *LYCORIS_ADAPTER_TYPES}
    assert sorted(cap.adapter_layer_groups) == sorted(ADAPTER_LAYER_GROUPS)


def test_registry_resolves_the_pipeline_lazily():
    from rengu_flow.registry.models import _ensure_model_imported, canonical_model_types, model_registry

    _ensure_model_imported("qwen_image21")
    assert "qwen_image21" in model_registry and "qwen_image21" in canonical_model_types


def test_preview_memory_knobs_visible_in_ui():
    from rengu_flow_ui.preview_form import WHEN_DIT_PREVIEW

    assert "qwen_image21" in WHEN_DIT_PREVIEW["in"]


def test_tread_is_rejected_at_validation():
    cfg = _config()
    cfg["tread"] = {"drop_ratio": 0.5}
    with pytest.raises(ConfigValidationError, match="tread"):
        validate_config(cfg)


def test_install_profile_includes_transformers_stack():
    from rengu_flow.install.manager import profiles_for_config_dict

    assert "cosmos" in profiles_for_config_dict({"model": {"type": "qwen_image21"}})
