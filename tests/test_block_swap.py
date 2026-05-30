"""Unit tests for shared block swap offloader (no GPU required for layout logic)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rengu_flow.training.block_swap import BlockSwapOffloader, NoopOffloader


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


def test_noop_offloader_is_disabled() -> None:
    noop = NoopOffloader()
    assert noop.enabled is False
    noop.wait_for_block(0)
    noop.submit_move_blocks_forward(0)
    noop.teardown()


def test_block_swap_disabled_when_zero() -> None:
    blocks = nn.ModuleList([_Block(), _Block()])
    off = BlockSwapOffloader(blocks, blocks_to_swap=0, device="cpu")
    assert off.enabled is False
    assert next(blocks[0].parameters()).device.type == "cpu"


def test_block_swap_moves_all_blocks_to_cpu_when_enabled() -> None:
    blocks = nn.ModuleList([_Block(), _Block(), _Block()])
    for b in blocks:
        b.to("cpu")
    off = BlockSwapOffloader(blocks, blocks_to_swap=2, device="cpu")
    assert off.enabled is True
    for b in blocks:
        assert next(b.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device moves")
def test_block_swap_roundtrip_cuda() -> None:
    blocks = nn.ModuleList([_Block(), _Block()])
    blocks.to("cuda")
    off = BlockSwapOffloader(blocks, blocks_to_swap=2, device="cuda")
    off.wait_for_block(0)
    assert next(blocks[0].parameters()).device.type == "cuda"
    off.submit_move_blocks_forward(0)
    assert next(blocks[0].parameters()).device.type == "cpu"
    off.teardown()
    for b in blocks:
        assert next(b.parameters()).device.type == "cuda"
