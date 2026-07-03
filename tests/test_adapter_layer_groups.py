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


# --- krea2: every declared group matches real module paths ----------------------


def test_krea2_groups_match_real_modules():
    """Regression vs module renames: each named group must select at least one linear."""
    model = _tiny_krea2()
    names = adapter_dit._collect_target_linears(model, KREA2_TARGETS)
    for group, patterns in KREA2_GROUPS.items():
        matched = filter_target_names(names, list(patterns), None)
        assert matched, f"layer group {group!r} matched no module (patterns {patterns})"


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


# --- cosmos ---------------------------------------------------------------------


def test_cosmos_groups_match_real_modules():
    from rengu_flow.model.cosmos_predict2.dit import MiniTrainDIT
    from rengu_flow.model.cosmos_predict2.pipeline import (
        ADAPTER_LAYER_GROUPS as COSMOS_GROUPS,
    )

    model = MiniTrainDIT(
        max_img_h=32,
        max_img_w=32,
        max_frames=1,
        in_channels=4,
        out_channels=4,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=32,
        num_blocks=2,
        num_heads=2,
        crossattn_emb_channels=16,
        pos_emb_cls="rope3d",
    )
    names = adapter_dit._collect_target_linears(model, adapter_dit.ADAPTER_TARGET_MODULES)
    assert names, "no target linears collected from cosmos DiT"
    for group, patterns in COSMOS_GROUPS.items():
        matched = filter_target_names(names, list(patterns), None)
        assert matched, f"layer group {group!r} matched no module (patterns {patterns})"


# --- capability / config / UI sync ----------------------------------------------


def test_capability_names_match_pipeline_groups():
    from rengu_flow.model.cosmos_predict2.pipeline import (
        ADAPTER_LAYER_GROUPS as COSMOS_GROUPS,
    )
    from rengu_flow.registry.model_capabilities import get_capability

    assert sorted(get_capability("krea2").adapter_layer_groups) == sorted(KREA2_GROUPS)
    assert sorted(get_capability("cosmos_predict2").adapter_layer_groups) == sorted(
        COSMOS_GROUPS
    )


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


def test_schema_offers_layer_groups_for_krea2():
    from rengu_flow_ui.config_schema import get_sections

    fields = [
        f
        for s in get_sections()
        for f in s["fields"]
        if f["path"] == "adapter.layer_groups"
    ]
    krea2_variant = next(
        f
        for f in fields
        if any(
            cond.get("field") == "model.type" and cond.get("in") == ["krea2"]
            for cond in (f.get("when") or {}).get("all", [])
        )
    )
    assert "text_fusion" in krea2_variant["options"]
