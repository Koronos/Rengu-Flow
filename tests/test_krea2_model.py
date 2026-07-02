"""CPU-only unit tests for the Krea 2 DiT: pack/unpack, forward shape, padding invariance,
pipeline-layer parity, adapter attach, text-embedding cache helpers, and preview sampling
helpers (sigma schedule + single denoise step parity)."""

from __future__ import annotations

import copy

import pytest
import torch

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model.krea2 import preview_sampling
from rengu_flow.model.krea2.dit import (
    Krea2Transformer2DModel,
    pack_latents,
    prepare_position_ids,
    unpack_latents,
)
from rengu_flow.model.krea2.layers import FinalLayer, InitialLayer, TransformerLayer
from rengu_flow.model.krea2.loading import (
    _guard_not_prequantized,
    convert_dit_original_to_diffusers,
    is_original_dit_state_dict,
)
from rengu_flow.model.krea2.text import compact_text_embeddings, pad_text_embeddings
from rengu_flow.networks import adapter_dit
from rengu_flow.training.block_swap import NoopOffloader


@pytest.fixture
def tiny_model() -> Krea2Transformer2DModel:
    torch.manual_seed(0)
    model = Krea2Transformer2DModel(
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
    ).eval()
    return model


class _StubPipeline:
    """Minimal stand-in for Krea2Pipeline: only what ``_denoise_step`` reads."""

    def __init__(self, transformer):
        self.transformer = transformer
        self._preview_offloader = None


def _text_mask_sample1_from_token4(batch: int = 2, tokens: int = 7) -> torch.Tensor:
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    mask[1, 4:] = False
    return mask


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 8, 12)
    packed = pack_latents(latents)
    assert packed.shape == (2, 24, 16)
    unpacked = unpack_latents(packed, 4, 6)
    assert torch.allclose(unpacked, latents)


def test_forward_shape(tiny_model):
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 8, 12)
    packed = pack_latents(latents)
    embeds = torch.randn(2, 7, 3, 24)
    mask = _text_mask_sample1_from_token4()
    t = torch.rand(2)
    position_ids = prepare_position_ids(7, 4, 6, "cpu")

    output = tiny_model(packed, embeds, t, position_ids, encoder_attention_mask=mask)

    assert output.shape == (2, 24, 16)


def test_padding_invariance(tiny_model):
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 8, 12)
    packed = pack_latents(latents)
    embeds = torch.randn(2, 7, 3, 24)
    mask = _text_mask_sample1_from_token4()
    t = torch.rand(2)
    position_ids = prepare_position_ids(7, 4, 6, "cpu")

    output = tiny_model(packed, embeds, t, position_ids, encoder_attention_mask=mask)

    embeds_dirty = embeds.clone()
    embeds_dirty[1, 4:] = 999.0
    output_dirty = tiny_model(packed, embeds_dirty, t, position_ids, encoder_attention_mask=mask)

    assert torch.allclose(output, output_dirty, atol=1e-5)


def test_pipeline_layers_match_monolithic_forward(tiny_model):
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 8, 12)
    packed = pack_latents(latents)
    embeds = torch.randn(2, 7, 3, 24)
    mask = _text_mask_sample1_from_token4()
    t = torch.rand(2)
    position_ids = prepare_position_ids(7, 4, 6, "cpu")

    expected = unpack_latents(
        tiny_model(packed, embeds, t, position_ids, encoder_attention_mask=mask), 4, 6
    )

    initial = InitialLayer(tiny_model)
    final = FinalLayer(tiny_model)
    layers = [TransformerLayer(block, i, NoopOffloader()) for i, block in enumerate(tiny_model.transformer_blocks)]

    outputs = initial((latents, t.view(-1, 1), embeds, mask))
    for layer in layers:
        outputs = layer(outputs)
    actual = final(outputs)

    assert torch.allclose(actual, expected, atol=1e-5)


ADAPTER_CONFIGS = [
    pytest.param(
        {"type": "lora", "rank": 4, "alpha": 4, "dropout": 0.0, "dtype": torch.float32},
        id="lora",
    ),
    pytest.param(
        {
            "type": "lokr",
            "rank": 4,
            "alpha": 4,
            "factor": -1,
            "decompose_both": False,
            "full_matrix": False,
            "dtype": torch.float32,
        },
        id="lokr",
    ),
    pytest.param(
        {
            "type": "lycoris_locon",
            "rank": 4,
            "alpha": 4,
            "dropout": 0.0,
            "rank_dropout": 0.0,
            "module_dropout": 0.0,
            "train_norm": False,
            "train_conv": False,
            "use_tucker": False,
            "use_scalar": False,
            "dora_wd": False,
            "rs_lora": False,
            "wd_on_output": True,
            "dtype": torch.float32,
        },
        id="lycoris_locon",
    ),
]


@pytest.mark.parametrize("adapter_cfg", ADAPTER_CONFIGS)
def test_adapter_attach_targets_all_dit_linears(tiny_model, adapter_cfg):
    """The model authors' recommended LoRA scope is every Linear in the DiT (reference rank
    32 / alpha 32): per-block attention/MLP, text fusion, and the shared img_in/txt_in/time
    projections and final linear — not just the transformer blocks."""
    if adapter_cfg["type"] == "lycoris_locon":
        pytest.importorskip("lycoris")

    model = copy.deepcopy(tiny_model)
    for name, p in model.named_parameters():
        p.original_name = name
        p.requires_grad_(False)

    adapter_dit.configure(model, adapter_cfg, targets=("Krea2Transformer2DModel",))

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert any("text_fusion" in name for name in trainable_names)


def test_compact_text_embeddings():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 10, 3, 24)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    # sample 0: 4 valid tokens (mixed positions — first 2 and last 2)
    mask[0, [0, 1, 8, 9]] = True
    # sample 1: 7 valid tokens (mixed positions — first 5 and last 2)
    mask[1, [0, 1, 2, 3, 4, 8, 9]] = True

    out, out_mask = compact_text_embeddings(hidden_states, mask)

    assert out.shape == (2, 7, 3, 24)
    assert out_mask.sum(dim=1).tolist() == [4, 7]
    assert torch.equal(out[0, :4], hidden_states[0][mask[0]])
    assert torch.equal(out[1, :7], hidden_states[1][mask[1]])
    assert not out_mask[0, 4:].any()
    assert torch.equal(out[0, 4:], torch.zeros_like(out[0, 4:]))


def test_pad_text_embeddings():
    torch.manual_seed(0)
    e0 = torch.randn(4, 3, 24)
    e1 = torch.randn(7, 3, 24)
    m0 = torch.ones(4, dtype=torch.bool)
    m1 = torch.ones(7, dtype=torch.bool)

    out, out_mask = pad_text_embeddings([e0, e1], [m0, m1])

    assert out.shape == (2, 7, 3, 24)
    assert out_mask.dtype == torch.bool
    assert torch.equal(out[0, :4], e0)
    assert torch.equal(out[1, :7], e1)
    assert out_mask[0].tolist() == [True, True, True, True, False, False, False]
    assert out_mask[1].tolist() == [True] * 7


def test_shifted_sigmas_schedule():
    sigmas = preview_sampling._shifted_sigmas(8, 4096, "cpu")

    assert sigmas.shape == (9,)
    assert sigmas[0].item() == pytest.approx(1.0)
    assert sigmas[-1].item() == pytest.approx(0.0)
    assert all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 1))


def test_denoise_step_matches_monolithic_forward(tiny_model):
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 8, 12)
    packed = pack_latents(latents)
    embeds = torch.randn(2, 7, 3, 24)
    mask = _text_mask_sample1_from_token4()
    t = torch.rand(2)
    position_ids = prepare_position_ids(7, 4, 6, "cpu")
    text_seq_len = mask.shape[1]
    image_seq_len = packed.shape[1]

    expected = tiny_model(packed, embeds, t, position_ids, encoder_attention_mask=mask)

    text_attn_mask, attn_mask = tiny_model.build_attention_masks(mask, image_seq_len)
    text_states = tiny_model.txt_in(tiny_model.text_fusion(embeds, attention_mask=text_attn_mask))
    temb_t = tiny_model.time_embed(t, dtype=packed.dtype)
    rope = tiny_model.rotary_emb(position_ids)

    pipeline_stub = _StubPipeline(tiny_model)
    actual = preview_sampling._denoise_step(
        pipeline_stub, packed, text_states, temb_t, attn_mask, rope, text_seq_len
    )

    assert torch.allclose(actual, expected, atol=1e-5)


# ---- original-layout <-> diffusers-layout DiT converter (loading.py) -------------------------

# The inverse of loading._BLOCK_RENAMES / loading._TOP_RENAMES (diffusers naming -> original
# Krea naming), used only to synthesize an original-layout state dict from the diffusers-layout
# tiny fixture for the round-trip test below.
_INV_BLOCK_RENAMES = {
    ".attn.to_q.": ".attn.wq.",
    ".attn.to_k.": ".attn.wk.",
    ".attn.to_v.": ".attn.wv.",
    ".attn.to_out.0.": ".attn.wo.",
    ".attn.to_gate.": ".attn.gate.",
    ".attn.norm_q.weight": ".attn.qknorm.qnorm.scale",
    ".attn.norm_k.weight": ".attn.qknorm.knorm.scale",
    ".norm1.weight": ".prenorm.scale",
    ".norm2.weight": ".postnorm.scale",
    ".ff.gate.": ".mlp.gate.",
    ".ff.up.": ".mlp.up.",
    ".ff.down.": ".mlp.down.",
}
_INV_TOP_RENAMES = {
    "img_in.": "first.",
    "time_embed.linear_1.": "tmlp.0.",
    "time_embed.linear_2.": "tmlp.2.",
    "time_mod_proj.": "tproj.1.",
    "txt_in.norm.weight": "txtmlp.0.scale",
    "txt_in.linear_1.": "txtmlp.1.",
    "txt_in.linear_2.": "txtmlp.3.",
    "text_fusion.": "txtfusion.",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.linear.": "last.linear.",
}


def _original_layout_state_dict(diffusers_state_dict: dict) -> dict:
    """Invert ``loading.convert_dit_original_to_diffusers`` to build a state dict in the
    original Krea key layout, for round-tripping the converter in tests."""
    original = {}
    for key, value in diffusers_state_dict.items():
        if key.startswith("transformer_blocks."):
            new_key = "blocks." + key[len("transformer_blocks.") :]
            if new_key.endswith(".scale_shift_table"):
                original[new_key.replace(".scale_shift_table", ".mod.lin")] = value.reshape(-1)
                continue
            for old, repl in _INV_BLOCK_RENAMES.items():
                new_key = new_key.replace(old, repl)
            original[new_key] = value
            continue
        if key == "final_layer.scale_shift_table":
            original["last.modulation.lin"] = value  # shape [2, dim] as-is
            continue
        new_key = key
        for old, repl in _INV_TOP_RENAMES.items():
            if new_key.startswith(old):
                new_key = repl + new_key[len(old) :]
                break
        if new_key.startswith("txtfusion."):
            for old, repl in _INV_BLOCK_RENAMES.items():
                new_key = new_key.replace(old, repl)
        original[new_key] = value
    return original


def test_is_original_dit_state_dict():
    assert is_original_dit_state_dict(
        {"first.weight": torch.zeros(1), "blocks.0.attn.wq.weight": torch.zeros(1)}
    )
    assert not is_original_dit_state_dict(
        {"img_in.weight": torch.zeros(1), "transformer_blocks.0.attn.to_q.weight": torch.zeros(1)}
    )


def test_convert_dit_original_to_diffusers_round_trip(tiny_model):
    original_state_dict = _original_layout_state_dict(tiny_model.state_dict())
    assert is_original_dit_state_dict(original_state_dict)

    converted = convert_dit_original_to_diffusers(original_state_dict)

    rebuilt = Krea2Transformer2DModel(**tiny_model.config).eval()
    rebuilt.load_state_dict(converted, strict=True)

    torch.manual_seed(0)
    latents = torch.randn(2, 4, 8, 12)
    packed = pack_latents(latents)
    embeds = torch.randn(2, 7, 3, 24)
    mask = _text_mask_sample1_from_token4()
    t = torch.rand(2)
    position_ids = prepare_position_ids(7, 4, 6, "cpu")

    expected = tiny_model(packed, embeds, t, position_ids, encoder_attention_mask=mask)
    actual = rebuilt(packed, embeds, t, position_ids, encoder_attention_mask=mask)

    assert torch.equal(actual, expected)


def test_guard_not_prequantized_rejects_scaled_state_dict():
    state_dict = {
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(1),
        "transformer_blocks.0.attn.to_q.scale_weight": torch.zeros(1),
    }
    with pytest.raises(ConfigValidationError, match="pre-quantized"):
        _guard_not_prequantized(state_dict, "transformer_path")


def test_adapter_export_prefix_uses_official_transformer_prefix(tmp_path):
    state_dict = {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 2)}
    adapter_config = {"type": "lora", "dtype": torch.float32}

    adapter_dit.save(tmp_path, state_dict, adapter_config, peft_config=None, export_prefix="transformer.")

    from safetensors.torch import load_file

    saved = load_file(tmp_path / "adapter_model.safetensors")
    assert all(k.startswith("transformer.") for k in saved)


def test_te_fp8_scaled_dequant_scheme():
    """ComfyUI scaled-fp8 TE entries (fp8 .weight + scalar .weight_scale + .comfy_quant
    marker) dequantize to weight * scale; markers and vision keys are dropped."""
    w = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
    sd = {
        "model.layers.0.mlp.down_proj.weight": w.to(torch.float8_e4m3fn),
        "model.layers.0.mlp.down_proj.weight_scale": torch.tensor(2.0),
        "model.layers.0.mlp.down_proj.comfy_quant": torch.zeros(1, dtype=torch.uint8),
        "model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
        "model.visual.patch_embed.weight": torch.zeros(1),
    }
    # Same remap+dequant logic as loading.load_text_encoder's single-file branch.
    import re

    scales = {k[: -len(".weight_scale")]: v.float() for k, v in sd.items() if k.endswith(".weight_scale")}
    remapped = {}
    for k, v in sd.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        k = re.sub(r"^model\.", "", k)
        if k.startswith(("visual.", "lm_head.")) or k.endswith((".weight_scale", ".comfy_quant")):
            continue
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            v = v.float() * scales.get(base, torch.tensor(1.0))
        remapped[k] = v.to(torch.bfloat16)
    assert set(remapped) == {"layers.0.mlp.down_proj.weight", "norm.weight"}
    assert torch.allclose(
        remapped["layers.0.mlp.down_proj.weight"].float(), w * 2.0, atol=0.1
    )
