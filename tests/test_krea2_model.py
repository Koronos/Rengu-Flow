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


def test_pipeline_grid_metadata_uses_shape_without_storage(tiny_model):
    """Grid dimensions must not require CUDA scalar reads in the final pipeline layer."""
    latents = torch.randn(2, 4, 8, 12)
    embeds = torch.randn(2, 7, 3, 24)
    mask = _text_mask_sample1_from_token4()
    t = torch.rand(2)

    outputs = InitialLayer(tiny_model)((latents, t.view(-1, 1), embeds, mask))
    grid = outputs[7]
    assert grid.shape == (4, 6, 0)
    assert grid.numel() == 0
    actual = FinalLayer(tiny_model)(outputs)
    assert actual.shape == latents.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pipeline_grid_metadata_cuda_has_no_value_transfer(tiny_model):
    """CUDA smoke for the zero-storage grid representation used across pipeline stages."""
    model = tiny_model.to("cuda")
    latents = torch.randn(2, 4, 8, 12, device="cuda")
    embeds = torch.randn(2, 7, 3, 24, device="cuda")
    mask = _text_mask_sample1_from_token4().to("cuda")
    t = torch.rand(2, device="cuda")

    outputs = InitialLayer(model)((latents, t.view(-1, 1), embeds, mask))
    grid = outputs[7]
    assert grid.is_cuda and grid.shape == (4, 6, 0) and grid.numel() == 0
    actual = FinalLayer(model)(outputs)
    torch.cuda.synchronize()
    assert actual.shape == latents.shape and torch.isfinite(actual).all()


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


def test_pipeline_layers_all_valid_mask(tiny_model):
    """All-valid masks take the fused-SDPA fast path (attn mask None -> 0-size sentinel
    through the layer tuple) and must still match the monolithic forward."""
    from rengu_flow.model.krea2.dit import pack_latents, prepare_position_ids, unpack_latents
    from rengu_flow.model.krea2.layers import FinalLayer, InitialLayer, TransformerLayer
    from rengu_flow.training.block_swap import NoopOffloader

    torch.manual_seed(0)
    lat = torch.randn(2, 4, 8, 12)
    embeds = torch.randn(2, 7, 3, 24)
    mask = torch.ones(2, 7, dtype=torch.bool)
    t = torch.rand(2)

    layers = (
        [InitialLayer(tiny_model)]
        + [TransformerLayer(b, i, NoopOffloader()) for i, b in enumerate(tiny_model.transformer_blocks)]
        + [FinalLayer(tiny_model)]
    )
    x = (lat, t.view(-1, 1), embeds, mask)
    for layer in layers:
        x = layer(x)
    with torch.no_grad():
        ref = tiny_model(pack_latents(lat), embeds, t, prepare_position_ids(7, 4, 6, "cpu"), encoder_attention_mask=mask)
    assert torch.allclose(x.detach(), unpack_latents(ref, 4, 6), atol=1e-5)


# ---- training perf/memory: RoPE grad, GAS text trim, ragged cache, VAE mode, text AC,
# ---- adapter scope, preview memoization, lazy/streamed loading ------------------------------


def _stub_pipeline(transformer=None, **config):
    """A Krea2Pipeline without its component loads (``__init__`` reads the VAE from disk)."""
    from rengu_flow.model.krea2.pipeline import Krea2Pipeline

    pipe = object.__new__(Krea2Pipeline)
    pipe.config = {"model": {"dtype": torch.float32}, **config}
    pipe.model_config = pipe.config["model"]
    pipe._init_block_swap_state()
    pipe.transformer = transformer
    pipe.cache_text_embeddings = True
    pipe._preview_embed_cache = {}
    return pipe


def _step_inputs():
    torch.manual_seed(0)
    return (torch.randn(2, 4, 8, 12), torch.rand(2).view(-1, 1), torch.randn(2, 7, 3, 24), _text_mask_sample1_from_token4())


def _run_layers(layers, inputs, reentrant_ac: bool = False):
    from torch.utils.checkpoint import checkpoint

    x = inputs
    for layer in layers:
        if reentrant_ac and isinstance(layer, TransformerLayer):
            x = checkpoint(lambda *xs, _l=layer: _l(xs), *x, use_reentrant=True)
        else:
            x = layer(x)
    return x


def _trainable(model):
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


@pytest.mark.parametrize("reentrant_ac", [False, True], ids=["plain", "reentrant_ac"])
def test_initial_layer_rope_tables_do_not_require_grad(tiny_model, reentrant_ac):
    """Only hidden/temb/temb_mod carry gradient: the RoPE tables are constants. Parameter
    grads (text fusion, img_in, blocks) match the monolithic forward, with and without
    reentrant AC around the blocks."""
    model = _trainable(tiny_model)
    inputs = _step_inputs()

    initial = InitialLayer(model)
    outputs = initial(inputs)
    assert [t.requires_grad for t in outputs[:5]] == [True, True, True, False, False]

    # Reentrant AC hands passthrough tensors back requiring grad: the blocks must still see
    # constant RoPE tables.
    rope_grad = []
    hooks = [
        b.register_forward_pre_hook(lambda _m, args: rope_grad.append(args[2][0].requires_grad))
        for b in model.transformer_blocks
    ]
    blocks = [TransformerLayer(b, i, NoopOffloader()) for i, b in enumerate(model.transformer_blocks)]
    _run_layers([initial, *blocks, FinalLayer(model)], inputs, reentrant_ac).square().mean().backward()
    for hook in hooks:
        hook.remove()
    assert rope_grad and not any(rope_grad)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    ref = model(
        pack_latents(inputs[0]), inputs[2], inputs[1].view(-1), prepare_position_ids(7, 4, 6, "cpu"),
        encoder_attention_mask=inputs[3],
    )
    unpack_latents(ref, 4, 6).square().mean().backward()
    ref_grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads.keys() == ref_grads.keys()
    assert any(n.startswith("text_fusion.") for n in grads)
    for name, grad in ref_grads.items():
        assert torch.allclose(grads[name], grad, atol=1e-5), name


def test_initial_layer_pipe_parallel_marks_every_float_output(tiny_model):
    """DeepSpeed pipe (> 1 stage) backprops every floating inter-stage tensor."""
    outputs = InitialLayer(tiny_model, pipe_parallel=True)(_step_inputs())
    assert all(t.requires_grad for t in outputs if torch.is_floating_point(t))


def test_text_branch_checkpointed_under_activation_checkpointing(tiny_model, monkeypatch):
    """With AC on, text_fusion + txt_in run under non-reentrant checkpoint (grad mode only);
    outputs and grads match the un-checkpointed layer."""
    model = _trainable(tiny_model)
    inputs = _step_inputs()
    calls = []
    real_checkpoint = torch.utils.checkpoint.checkpoint

    def spy(fn, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return real_checkpoint(fn, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)

    results = []
    for checkpoint_text in (False, True):
        out = InitialLayer(model, checkpoint_text=checkpoint_text)(inputs)[0]
        out.square().mean().backward()
        results.append((out.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
        model.zero_grad(set_to_none=True)
    with torch.no_grad():
        InitialLayer(model, checkpoint_text=True)(inputs)

    assert calls == [False]
    (out_a, grads_a), (out_b, grads_b) = results
    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        assert torch.allclose(grads_a[name], grads_b[name], atol=1e-6), name


@pytest.mark.parametrize(
    ("ac", "expected"), [(False, False), (True, True), ("auto", False)], ids=["off", "on", "auto"]
)
def test_to_layers_checkpoints_text_only_for_manual_ac(tiny_model, ac, expected):
    initial = _stub_pipeline(tiny_model, activation_checkpointing=ac).to_layers()[0]
    assert isinstance(initial, InitialLayer)
    assert initial.checkpoint_text is expected
    assert initial.pipe_parallel is False
    assert _stub_pipeline(tiny_model, pipeline_stages=2).to_layers()[0].pipe_parallel is True


def _krea2_step_features(lengths: list[int], tokens: int = 7):
    """``prepare_inputs``-shaped features of one step: text right-padded to ``tokens``."""
    mask = torch.zeros(len(lengths), tokens, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    embeds = torch.randn(len(lengths), tokens, 3, 24) * mask[..., None, None]
    return (torch.randn(len(lengths), 4, 8, 12), torch.rand(len(lengths)), embeds, mask)


@pytest.mark.parametrize("pipe_parallel", [False, True], ids=["single_stage", "pipe_parallel"])
def test_loader_trims_text_padding_per_micro_batch(tiny_model, pipe_parallel):
    """GAS=2 with mixed caption lengths: each micro-batch drops the text lanes only the other
    one used, so it takes the unmasked attention path — except under pipeline parallelism,
    where the micro-batches of a step must keep one shape."""
    from rengu_flow.data import PipelineDataLoader, SyntheticSDXLDataset

    pipe = _stub_pipeline(tiny_model)
    features = _krea2_step_features([3, 3, 7, 5])
    label = (torch.randn(4, 16, 8, 12), None)

    class Model:
        trim_micro_batch = staticmethod(pipe.trim_micro_batch)

        def prepare_inputs(self, batch, timestep_quantile=None):
            return features, label

    class Engine:
        is_pipe_parallel = pipe_parallel

        def is_first_stage(self):  # a middle stage: the loader skips the target broadcast
            return False

        is_last_stage = is_first_stage

    loader = PipelineDataLoader(SyntheticSDXLDataset(num_batches=1, micro_batch_size=1), Engine(), 2, Model())
    (mb0, _), (mb1, _) = loader._prepare_batch({})

    if pipe_parallel:
        assert mb0[2].shape[1] == mb1[2].shape[1] == 7
        return
    assert mb0[2].shape == (2, 3, 3, 24) and bool(mb0[3].all())
    assert mb1[2].shape == (2, 7, 3, 24)
    assert torch.equal(mb0[2], features[2][:2, :3])
    # The trimmed all-valid micro-batch ships the 0-size "no mask" sentinel (fused SDPA) and
    # predicts exactly what the padded one did.
    trimmed = InitialLayer(tiny_model)((mb0[0], mb0[1].view(-1, 1), mb0[2], mb0[3]))
    assert trimmed[5].numel() == 0
    padded = (features[0][:2], features[1][:2].view(-1, 1), features[2][:2], features[3][:2])
    rest = [TransformerLayer(b, i, NoopOffloader()) for i, b in enumerate(tiny_model.transformer_blocks)]
    rest.append(FinalLayer(tiny_model))
    with torch.no_grad():
        assert torch.allclose(_run_layers(rest, trimmed), _run_layers([InitialLayer(tiny_model), *rest], padded), atol=1e-5)


def test_text_encoder_fn_returns_per_row_valid_tokens(monkeypatch):
    """caching_batch_size > 1: each caption is cached at its own length (no False tails)."""
    from rengu_flow.model.krea2 import pipeline as krea2_pipeline

    embeds = torch.randn(2, 6, 3, 24)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, [0, 1, 5]] = True  # middle padding: [prompt | PAD | suffix]
    mask[1, :] = True
    monkeypatch.setattr(krea2_pipeline, "encode_prompts", lambda *a, **k: (embeds, mask))
    pipe = _stub_pipeline()
    pipe.tokenizer, pipe.select_layers, pipe.max_sequence_length = None, (1,), 6

    out = pipe.get_call_text_encoder_fn(torch.nn.Linear(1, 1))(["a", "b"], False)

    assert [tuple(e.shape) for e in out["prompt_embeds"]] == [(3, 3, 24), (6, 3, 24)]
    assert [m.tolist() for m in out["text_mask"]] == [[True] * 3, [True] * 6]
    assert torch.equal(out["prompt_embeds"][0], embeds[0][mask[0]])
    padded, padded_mask = pad_text_embeddings(out["prompt_embeds"], out["text_mask"])
    assert padded.shape == (2, 6, 3, 24) and padded_mask.sum(1).tolist() == [3, 6]


def test_vae_fn_caches_distribution_mode():
    """The cached latent is the posterior mode (mean), not one frozen random draw."""
    from types import SimpleNamespace

    class Dist:
        def mode(self):
            return torch.full((1, 2, 1, 2, 2), 3.0)

        def sample(self):
            raise AssertionError("latent_dist.sample() must not be used for caching")

    class Vae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(latents_mean=[1.0, 1.0], latents_std=[2.0, 2.0])

        def encode(self, x):
            return SimpleNamespace(latent_dist=Dist())

    pipe = _stub_pipeline()
    pipe.vae = Vae()
    out = pipe.get_call_vae_fn(pipe.vae)(torch.zeros(1, 3, 16, 16))
    assert torch.equal(out["latents"], torch.ones(1, 2, 2, 2))


def test_adapter_layer_groups_cover_default_scope(tiny_model):
    """The union of every layer group equals the default all-linears scope (text-fusion
    projector included)."""
    from rengu_flow.model.krea2.pipeline import ADAPTER_LAYER_GROUPS
    from rengu_flow.networks.adapter_targets import filter_target_names

    all_linears = adapter_dit._collect_target_linears(tiny_model, ("Krea2Transformer2DModel",))
    assert "text_fusion.projector" in all_linears
    union = [p for patterns in ADAPTER_LAYER_GROUPS.values() for p in patterns]
    assert sorted(filter_target_names(all_linears, union, None)) == sorted(all_linears)


def test_preview_prompt_embeds_memoized_without_autocast(monkeypatch):
    """Preview prompts are encoded once (no autocast, like the training cache); afterwards the
    text encoder goes back to meta and the VAE stays in host RAM (no disk reload next time)."""
    from rengu_flow.model.dit_common.streaming import LazyStreamedEncoder
    from rengu_flow.model.krea2 import pipeline as krea2_pipeline

    loads, encodes, vae_loads = [], [], []

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

    def fake_encode(text_encoder, tokenizer, prompts, **kwargs):
        encodes.append((list(prompts), torch.is_autocast_enabled("cpu")))
        text_encoder.load()
        mask = torch.zeros(len(prompts), 5, dtype=torch.bool)
        for i, p in enumerate(prompts):
            mask[i, : len(p) + 1] = True
        return torch.randn(len(prompts), 5, 3, 24), mask

    def fake_load_vae(*args, **kwargs):
        vae_loads.append(1)
        return torch.nn.Linear(1, 1)

    def fake_load_encoder():
        loads.append(1)
        return Encoder()

    monkeypatch.setattr(krea2_pipeline, "encode_prompts", fake_encode)
    monkeypatch.setattr(krea2_pipeline.loading, "load_vae", fake_load_vae)
    pipe = _stub_pipeline()
    pipe.tokenizer, pipe.select_layers, pipe.max_sequence_length = None, (1,), 5
    pipe._component_path = lambda component: component
    pipe.text_encoder = LazyStreamedEncoder(fake_load_encoder, layers_of=lambda m: m.layers)
    pipe.vae = torch.nn.Linear(1, 1, device="meta")

    for _ in range(2):
        pipe.ensure_vae_for_preview()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            rows = pipe.preview_prompt_embeds(["ab", ""], {}, "cpu")
        pipe.restore_after_preview()
        assert [r.shape[0] for r in rows] == [3, 1]
        assert next(pipe.text_encoder.parameters()).device.type == "meta"
        assert next(pipe.vae.parameters()).device.type == "cpu"

    assert encodes == [(["ab", ""], False)]
    assert len(loads) == 1 and len(vae_loads) == 1


def test_pipeline_init_does_not_load_text_encoder(monkeypatch, tmp_path):
    """A warm text cache never needs the encoder: construction must not read it."""
    from rengu_flow.model.krea2 import pipeline as krea2_pipeline

    def eager(*args, **kwargs):
        raise AssertionError("text encoder loaded eagerly")

    monkeypatch.setattr(krea2_pipeline.loading, "load_vae", lambda *a, **k: torch.nn.Linear(1, 1))
    monkeypatch.setattr(krea2_pipeline.loading, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(krea2_pipeline.loading, "load_text_encoder", eager)
    pipe = krea2_pipeline.Krea2Pipeline({"model": {"dtype": torch.float32, "checkpoint_path": str(tmp_path)}})
    assert next(pipe.text_encoder.parameters()).device.type == "meta"


def test_load_text_encoder_folder_reads_text_decoder_only(monkeypatch, tmp_path):
    """A transformers folder goes through the shared text-only loader (no vision tower)."""
    import shutil

    from rengu_flow.model.dit_common import qwen3vl
    from rengu_flow.model.krea2 import loading

    shutil.copy(loading.QWEN3VL_ASSETS / "config.json", tmp_path / "config.json")
    (tmp_path / "model.safetensors").write_bytes(b"")
    seen = {}

    def fake_loader(files, text_config, dtype):
        seen.update(files=files, config=type(text_config).__name__, dtype=dtype)
        return "text-model"

    monkeypatch.setattr(qwen3vl, "load_qwen3vl_text_model", fake_loader)
    assert loading.load_text_encoder(tmp_path, torch.bfloat16) == "text-model"
    assert seen == {"files": [tmp_path / "model.safetensors"], "config": "Qwen3VLTextConfig", "dtype": torch.bfloat16}


def test_load_transformer_single_file_casts_while_reading(monkeypatch, tmp_path, tiny_model):
    """Single-file DiT: read tensor by tensor and cast to the load dtype as read."""
    from safetensors.torch import save_file

    from rengu_flow.model.krea2 import dit, loading

    path = tmp_path / "krea2_raw.safetensors"
    original = _original_layout_state_dict(tiny_model.state_dict())
    save_file({k: v.contiguous() for k, v in original.items()}, str(path))
    config = dict(tiny_model.config)
    monkeypatch.setattr(dit, "Krea2Transformer2DModel", lambda: Krea2Transformer2DModel(**config))

    loaded = loading.load_transformer(path, torch.bfloat16).state_dict()

    for name, value in tiny_model.state_dict().items():
        assert loaded[name].dtype == torch.bfloat16, name
        assert torch.equal(loaded[name], value.to(torch.bfloat16)), name
