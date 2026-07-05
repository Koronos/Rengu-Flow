"""CPU-only unit tests for TREAD-style token routing (rengu_flow.training.token_routing
+ the krea2 Route layers): selection invariants, identity bypass, gradient flow through
dropped tokens, eval no-op, and the pipeline-layer tuple protocol."""

from __future__ import annotations

import pytest
import torch

from rengu_flow.model.krea2.dit import Krea2TransformerBlock, prepare_position_ids
from rengu_flow.model.krea2.dit import Krea2RotaryPosEmbed
from rengu_flow.model.krea2.layers import RouteEndLayer, RouteStartLayer, TransformerLayer
from rengu_flow.training.block_swap import NoopOffloader
from rengu_flow.training.token_routing import (
    resolve_route,
    route_end,
    route_start,
    sample_keep_index,
)

TEXT, IMG, DIM = 3, 8, 32  # tiny krea2 block: heads 4 x head_dim 8, kv heads 2


def test_sample_keep_index_invariants():
    torch.manual_seed(0)
    idx = sample_keep_index(TEXT, IMG, drop_ratio=0.5, device="cpu")
    assert idx.tolist()[:TEXT] == [0, 1, 2]  # text always kept, in order
    assert len(idx) == TEXT + IMG - 4
    assert len(set(idx.tolist())) == len(idx)  # no dupes
    assert idx.tolist() == sorted(idx.tolist())  # order-preserving
    # ratio 0 -> full sequence
    assert sample_keep_index(TEXT, IMG, 0.0, "cpu").tolist() == list(range(TEXT + IMG))


def test_route_roundtrip_and_grad_flow():
    torch.manual_seed(1)
    full = torch.randn(2, TEXT + IMG, DIM, requires_grad=True)
    keep = sample_keep_index(TEXT, IMG, 0.5, "cpu")
    routed = route_start(full, keep)
    assert routed.shape == (2, len(keep), DIM)
    out = route_end(routed * 2.0, full, keep)
    dropped = [i for i in range(TEXT + IMG) if i not in set(keep.tolist())]
    assert torch.equal(out[:, dropped], full.detach()[:, dropped])  # identity bypass
    assert torch.equal(out[:, keep], (routed * 2.0).detach())
    out.sum().backward()  # grads reach dropped tokens through the bypass
    assert full.grad is not None and (full.grad[:, dropped] != 0).any()


def test_resolve_route_validation():
    assert resolve_route(28, 2, -3) == (2, 25)
    with pytest.raises(ValueError):
        resolve_route(28, 0, 25)  # block 0 must stay unrouted
    with pytest.raises(ValueError):
        resolve_route(28, 5, 27)  # last block must stay unrouted
    with pytest.raises(ValueError):
        resolve_route(28, 10, 5)


def _layer_inputs(batch=2, with_mask=False):
    torch.manual_seed(2)
    hidden = torch.randn(batch, TEXT + IMG, DIM, requires_grad=True)
    temb = torch.randn(batch, 1, DIM)
    temb_mod = torch.randn(batch, 1, 6 * DIM)
    rope = Krea2RotaryPosEmbed(theta=1000.0, axes_dim=[4, 2, 2])
    pos = prepare_position_ids(TEXT, 2, 4, "cpu")
    cos, sin = rope(pos)
    if with_mask:
        mask = torch.ones(batch, 1, 1, TEXT + IMG, dtype=torch.bool)
        mask[:, :, :, 1] = False  # one padded text token
    else:
        mask = torch.empty(0)
    text_mask = torch.ones(batch, TEXT, dtype=torch.bool)
    grid = torch.tensor([2, 4])
    return (hidden, temb, temb_mod, cos.float(), sin.float(), mask, text_mask, grid)


@pytest.mark.parametrize("with_mask", [False, True])
def test_route_layers_through_block(with_mask):
    inputs = _layer_inputs(with_mask=with_mask)
    block = Krea2TransformerBlock(DIM, 64, 4, 2, 1e-5)
    start, mid, end = RouteStartLayer(0.5), TransformerLayer(block, 0, NoopOffloader()), RouteEndLayer()
    for m in (start, mid, end):
        m.train()

    routed_tuple = start(inputs)
    assert len(routed_tuple) == 13
    keep_idx = routed_tuple[-1]
    assert routed_tuple[0].shape[1] == len(keep_idx)
    assert routed_tuple[3].shape[0] == len(keep_idx)  # rope cos sliced
    if with_mask:
        assert routed_tuple[5].shape[-1] == len(keep_idx)  # key mask sliced

    out_tuple = end(mid(routed_tuple))
    assert len(out_tuple) == 8
    out = out_tuple[0]
    assert out.shape == inputs[0].shape
    dropped = [i for i in range(TEXT + IMG) if i not in set(keep_idx.tolist())]
    assert torch.equal(out[:, dropped], inputs[0].detach()[:, dropped])
    out.sum().backward()
    assert inputs[0].grad is not None and inputs[0].grad.abs().sum() > 0


def test_route_layers_noop_in_eval():
    inputs = _layer_inputs()
    start, end = RouteStartLayer(0.5), RouteEndLayer()
    start.eval(), end.eval()
    with torch.no_grad():
        out = start(inputs)
        assert len(out) == 8 and out[0] is inputs[0]
        assert end(out) is out
