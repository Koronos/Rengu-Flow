"""CPU-only unit tests for the Krea 2 DiT: pack/unpack, forward shape, padding invariance,
pipeline-layer parity, adapter attach, text-embedding cache helpers, and preview sampling
helpers (sigma schedule + single denoise step parity)."""

from __future__ import annotations

import copy

import pytest
import torch

from rengu_flow.model.krea2 import preview_sampling
from rengu_flow.model.krea2.dit import (
    Krea2Transformer2DModel,
    pack_latents,
    prepare_position_ids,
    unpack_latents,
)
from rengu_flow.model.krea2.layers import FinalLayer, InitialLayer, TransformerLayer
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
def test_adapter_attach_targets_transformer_blocks_only(tiny_model, adapter_cfg):
    if adapter_cfg["type"] == "lycoris_locon":
        pytest.importorskip("lycoris")

    model = copy.deepcopy(tiny_model)
    for name, p in model.named_parameters():
        p.original_name = name
        p.requires_grad_(False)

    adapter_dit.configure(model, adapter_cfg, targets=("Krea2TransformerBlock",))

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert not any("text_fusion" in name for name in trainable_names)


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
