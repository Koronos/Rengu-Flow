"""Cosmos Predict2 pipeline layers: which inter-layer tensors require grad (CPU, tiny DiT)."""

from __future__ import annotations

import pytest
import torch

from rengu_flow.model.cosmos_predict2.dit import MiniTrainDIT
from rengu_flow.model.cosmos_predict2.layers import (
    FinalLayer,
    InitialLayer,
    LLMAdapterLayer,
    NoopOffloader,
    TransformerLayer,
)
from rengu_flow.model.cosmos_predict2.llm_adapter import LLMAdapter

CTX = 16


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
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
        crossattn_emb_channels=CTX,
        pos_emb_cls="rope3d",
        use_adaln_lora=True,
        adaln_lora_dim=8,
    )
    # A small stand-in for the 1024-d Anima LLM adapter (same module, tiny dims).
    model.llm_adapter = LLMAdapter(
        source_dim=CTX, target_dim=CTX, model_dim=CTX, num_layers=1, num_heads=2, self_attn=True
    )
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def _step_inputs():
    """Cached-text inputs: ``(x, t, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)``."""
    g = torch.Generator().manual_seed(1)
    t5_mask = torch.ones(2, 5, dtype=torch.long)
    t5_mask[1, 3:] = 0
    return (
        torch.randn(2, 4, 1, 8, 8, generator=g),
        torch.rand(2, generator=g),
        torch.randn(2, 6, CTX, generator=g),
        torch.ones(2, 6, dtype=torch.long),
        torch.randint(0, 100, (2, 5), generator=g),
        t5_mask,
    )


def _run_layers(layers, inputs, reentrant_ac: bool):
    from torch.utils.checkpoint import checkpoint

    x = inputs
    for layer in layers:
        if reentrant_ac and isinstance(layer, TransformerLayer):
            x = checkpoint(lambda *xs, _l=layer: _l(xs), *x, use_reentrant=True)
        else:
            x = layer(x)
    return x


@pytest.mark.parametrize("reentrant_ac", [False, True], ids=["plain", "reentrant_ac"])
def test_initial_layer_marks_only_gradient_carrying_outputs(tiny_model, reentrant_ac):
    """Only x / t_embedding / adaln_lora are marked: the RoPE table and timesteps are constants
    and the cached text embedding is a leaf (it gets grad from the LLM adapter when that trains).
    Blocks see a constant RoPE table even under reentrant AC, and every parameter grad (LLM
    adapter included) matches the monolithic forward."""
    model = tiny_model
    inputs = _step_inputs()

    initial = InitialLayer(model, None, is_generic_llm=False)
    outputs = initial(inputs)
    assert [bool(getattr(t, "requires_grad", False)) for t in outputs] == [
        True, True, False, False, False, False, False, True, False
    ]

    rope_grad = []
    hooks = [
        b.register_forward_pre_hook(
            lambda _m, _args, kwargs: rope_grad.append(kwargs["rope_emb_L_1_1_D"].requires_grad),
            with_kwargs=True,
        )
        for b in model.blocks
    ]
    layers = [
        initial,
        LLMAdapterLayer(model.llm_adapter),
        *[TransformerLayer(b, i, NoopOffloader()) for i, b in enumerate(model.blocks)],
        FinalLayer(model),
    ]
    _run_layers(layers, inputs, reentrant_ac).square().mean().backward()
    for hook in hooks:
        hook.remove()
    assert rope_grad and not any(rope_grad)  # reentrant AC also records its recompute
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    x, t, embeds, attn_mask, t5_ids, t5_mask = inputs
    crossattn = model.llm_adapter(
        source_hidden_states=embeds,
        target_input_ids=t5_ids,
        target_attention_mask=t5_mask,
        source_attention_mask=attn_mask,
    )
    crossattn[~t5_mask.bool()] = 0
    padding_mask = torch.zeros(2, 1, 8, 8)
    model(x, t, crossattn, padding_mask=padding_mask).square().mean().backward()
    ref_grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

    assert grads.keys() == ref_grads.keys()
    assert any(n.startswith("llm_adapter.") for n in grads)
    for name, grad in ref_grads.items():
        assert torch.allclose(grads[name], grad, atol=1e-5), name


def test_initial_layer_pipe_parallel_marks_every_float_output(tiny_model):
    """DeepSpeed pipe (> 1 stage) backprops every floating inter-stage tensor."""
    outputs = InitialLayer(tiny_model, None, is_generic_llm=False, pipe_parallel=True)(_step_inputs())
    assert all(t.requires_grad for t in outputs if torch.is_floating_point(t))


def test_to_layers_passes_pipe_parallel(tiny_model):
    from rengu_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline

    for stages, expected in ((1, False), (2, True)):
        pipe = object.__new__(CosmosPredict2Pipeline)
        pipe.config = {"pipeline_stages": stages}
        pipe.transformer, pipe.text_encoder, pipe.cache_text_embeddings = tiny_model, None, True
        pipe.is_generic_llm, pipe.use_llm_adapter, pipe.offloader = False, True, NoopOffloader()
        assert pipe.to_layers()[0].pipe_parallel is expected
