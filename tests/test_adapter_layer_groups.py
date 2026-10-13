"""Per-layer adapter training: named layer groups + include/exclude globs."""

from __future__ import annotations

import pytest
import torch

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model.krea2.dit import Krea2Transformer2DModel
from rengu_flow.model.krea2.pipeline import (
    ADAPTER_LAYER_GROUPS as KREA2_GROUPS,
    ADAPTER_TARGET_MODULES as KREA2_TARGETS,
)
from rengu_flow.networks import adapter_dit
from rengu_flow.networks.adapter_targets import apply_layer_groups, filter_target_names


def _tiny_krea2() -> Krea2Transformer2DModel:
    torch.manual_seed(0)
    return Krea2Transformer2DModel(
        in_channels=16,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        timestep_embed_dim=16,
        text_hidden_dim=24,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=48,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(4, 2, 2),
    )


def _lokr_config(**extra):
    return {
        "type": "lokr",
        "rank": 4,
        "alpha": 4,
        "factor": -1,
        "decompose_both": False,
        "full_matrix": False,
        "dtype": torch.float32,
        **extra,
    }


# --- pure helpers ---------------------------------------------------------------


def test_filter_target_names_globs():
    names = ["text_fusion.blocks.0.attn.to_q", "transformer_blocks.0.ff.gate", "txt_in.linear_1"]
    assert filter_target_names(names, ["text_fusion.*", "txt_in.*"], None) == [
        "text_fusion.blocks.0.attn.to_q",
        "txt_in.linear_1",
    ]
    assert filter_target_names(names, None, ["*.ff.*"]) == [
        "text_fusion.blocks.0.attn.to_q",
        "txt_in.linear_1",
    ]
    assert filter_target_names(names, None, None) == names


def test_apply_layer_groups_expands_and_merges():
    cfg = {"layer_groups": ["text_fusion"], "target_include": ["img_in"]}
    apply_layer_groups(cfg, KREA2_GROUPS)
    assert "img_in" in cfg["target_include"]
    assert "text_fusion.*" in cfg["target_include"]
    assert "txt_in.*" in cfg["target_include"]


def test_apply_layer_groups_unknown_name_raises():
    with pytest.raises(ConfigValidationError, match="text_fusio"):
        apply_layer_groups({"layer_groups": ["text_fusio"]}, KREA2_GROUPS)
    with pytest.raises(ConfigValidationError, match="none for this model"):
        apply_layer_groups({"layer_groups": ["anything"]}, None)


# --- krea2 (each declared group matching real linears, and the spec's groups being the
# pipeline's: tests/contract/test_pipeline_contract.py, every model) ---------------------


def test_krea2_lokr_text_fusion_only():
    model = _tiny_krea2()
    adapter_dit.configure(
        model,
        _lokr_config(layer_groups=["text_fusion"]),
        targets=KREA2_TARGETS,
        layer_groups=KREA2_GROUPS,
    )
    adapted = {
        name
        for name, module in model.named_modules()
        if hasattr(module, "_lokr_scale")
    }
    assert adapted, "no modules adapted"
    assert all(n.startswith(("text_fusion", "txt_in")) for n in adapted), adapted
    # The image-side stack stays frozen with no adapter params.
    assert not any(n.startswith("transformer_blocks") for n in adapted)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.dtype == torch.float32 for p in trainable)


def test_krea2_lokr_multiple_groups():
    model = _tiny_krea2()
    adapter_dit.configure(
        model,
        _lokr_config(layer_groups=["text_fusion", "attention"]),
        targets=KREA2_TARGETS,
        layer_groups=KREA2_GROUPS,
    )
    adapted = {n for n, m in model.named_modules() if hasattr(m, "_lokr_scale")}
    assert any(n.startswith("text_fusion") for n in adapted)
    assert any(".attn." in n and n.startswith("transformer_blocks") for n in adapted)
    assert not any(".ff." in n and n.startswith("transformer_blocks") for n in adapted)


def test_krea2_lora_peft_respects_groups():
    model = _tiny_krea2()
    cfg = {
        "type": "lora",
        "rank": 4,
        "alpha": 4,
        "dropout": 0.0,
        "dtype": torch.float32,
        "layer_groups": ["text_fusion"],
    }
    adapter_dit.configure(model, cfg, targets=KREA2_TARGETS, layer_groups=KREA2_GROUPS)
    lora_modules = {
        name.rsplit(".lora_A", 1)[0]
        for name, _ in model.named_modules()
        if name.endswith("lora_A")
    }
    assert lora_modules, "PEFT attached nothing"
    assert all(("text_fusion" in n or "txt_in" in n) for n in lora_modules), lora_modules


def test_krea2_globs_without_groups_still_filter():
    model = _tiny_krea2()
    adapter_dit.configure(
        model,
        _lokr_config(target_include=["transformer_blocks.*.ff.*"]),
        targets=KREA2_TARGETS,
        layer_groups=KREA2_GROUPS,
    )
    adapted = {n for n, m in model.named_modules() if hasattr(m, "_lokr_scale")}
    assert adapted and all(".ff." in n for n in adapted)


def test_patterns_matching_nothing_raise():
    model = _tiny_krea2()
    with pytest.raises(ConfigValidationError, match="matched no modules"):
        adapter_dit.configure(
            model,
            _lokr_config(target_include=["does_not_exist.*"]),
            targets=KREA2_TARGETS,
            layer_groups=KREA2_GROUPS,
        )


# --- config validation (the form's options per model: tests/contract/test_spec_contract.py) ---


def test_validate_config_rejects_unknown_group():
    from rengu_flow.config.validation import validate_config

    cfg = {
        "dataset": "examples/minimal_krea2_dataset.toml",
        "model": {
            "type": "krea2",
            "dtype": "bfloat16",
            "transformer_path": "x.safetensors",
            "vae_path": "v.safetensors",
            "text_encoder_path": "t.safetensors",
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "adapter": {"type": "lokr", "rank": 4, "layer_groups": ["not_a_group"]},
    }
    with pytest.raises(ConfigValidationError, match="not_a_group"):
        validate_config(cfg)


# --- qwen_image21 -----------------------------------------------------------------


def _tiny_qwen_image21():
    from rengu_flow.model.qwen_image21.dit import QwenImage21Transformer2DModel

    torch.manual_seed(0)
    return QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        context_in_dim=12,
        mlp_ratio=3,
        axes_dims_rope=(2, 2, 4),
    )


def test_qwen_image21_groups_match_real_modules_and_partition_the_linears():
    """Every group selects something, and together the groups cover each target linear
    exactly once (no overlap, nothing left out)."""
    from rengu_flow.model.qwen_image21.pipeline import (
        ADAPTER_LAYER_GROUPS as QWEN21_GROUPS,
        ADAPTER_TARGET_MODULES as QWEN21_TARGETS,
    )
    from rengu_flow.registry.model_capabilities import get_capability

    names = adapter_dit._collect_target_linears(_tiny_qwen_image21(), QWEN21_TARGETS)
    seen: list[str] = []
    for group, patterns in QWEN21_GROUPS.items():
        matched = filter_target_names(names, list(patterns), None)
        assert matched, f"layer group {group!r} matched no module (patterns {patterns})"
        seen.extend(matched)
    assert sorted(seen) == sorted(names)
    assert sorted(get_capability("qwen_image21").adapter_layer_groups) == sorted(QWEN21_GROUPS)


def test_qwen_image21_lokr_attention_only():
    from rengu_flow.model.qwen_image21.pipeline import (
        ADAPTER_LAYER_GROUPS as QWEN21_GROUPS,
        ADAPTER_TARGET_MODULES as QWEN21_TARGETS,
    )

    model = _tiny_qwen_image21()
    adapter_dit.configure(
        model, _lokr_config(layer_groups=["attention"]), targets=QWEN21_TARGETS, layer_groups=QWEN21_GROUPS
    )
    adapted = {n for n, m in model.named_modules() if hasattr(m, "_lokr_scale")}
    assert adapted and all(".attn." in n for n in adapted), adapted
