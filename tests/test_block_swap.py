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


def test_swap_trainable_false_filters_trainable_params() -> None:
    # A block with a frozen "base" weight and a trainable "adapter" weight (LoRA-like). With
    # swap_trainable=False the offloader must stream only the frozen base and leave the trainable
    # adapter where it is — otherwise a preview parks the adapter on CPU, the training offloader's
    # resume() (which re-homes only frozen params) never brings it back, and the next training step
    # hits a cpu-weight/cuda-input device mismatch. Observe the filter with the GPU-free "meta"
    # device as the "streamed-away" target: frozen moves, adapter stays.
    class _AdapterBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(4, 4)
            self.adapter = nn.Linear(4, 4)
            self.base.requires_grad_(False)
            self.adapter.requires_grad_(True)

    block = _AdapterBlock()
    off = BlockSwapOffloader(nn.ModuleList([block]), blocks_to_swap=1, device="cpu", swap_trainable=False)
    swappable = {id(p) for p in off._swappable_params(block)}
    assert id(block.base.weight) in swappable and id(block.base.bias) in swappable  # frozen streams
    assert id(block.adapter.weight) not in swappable  # trainable adapter stays resident, not streamed

    # swap_trainable=True (default) streams everything, adapter included.
    off_all = BlockSwapOffloader(nn.ModuleList([block]), 1, device="cpu")
    assert id(block.adapter.weight) in {id(p) for p in off_all._swappable_params(block)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device moves")
def test_swap_trainable_false_keeps_adapter_on_gpu_cuda() -> None:
    # The real-device guarantee: frozen base streams to CPU, trainable adapter stays on CUDA.
    class _AdapterBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(4, 4)
            self.adapter = nn.Linear(4, 4)
            self.base.requires_grad_(False)
            self.adapter.requires_grad_(True)

    blocks = nn.ModuleList([_AdapterBlock(), _AdapterBlock()])
    blocks.to("cuda")
    off = BlockSwapOffloader(blocks, blocks_to_swap=2, device="cuda", swap_trainable=False)
    for b in blocks:
        assert next(b.base.parameters()).device.type == "cpu"      # frozen streamed off
        assert next(b.adapter.parameters()).device.type == "cuda"  # adapter stayed resident
    off.wait_for_block(0)
    assert next(blocks[0].base.parameters()).device.type == "cuda"
    off.submit_move_blocks_forward(0)
    assert next(blocks[0].base.parameters()).device.type == "cpu"
    assert next(blocks[0].adapter.parameters()).device.type == "cuda"  # never left the GPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device moves")
def test_block_swap_keeps_tiny_buffers_resident_cuda() -> None:
    """Tiny buffers (fp8 weight scales) must stay GPU-resident even when the block streams to CPU —
    the training offloader's restore re-homes only params, so a parked scale strands on CPU and
    causes a cuda/cpu mismatch on the next forward."""

    class _Fp8ish(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, 4))       # streams CPU<->GPU
            self.register_buffer("scale", torch.ones(4))        # tiny -> stays resident
            self.register_buffer("big", torch.zeros(1_500_000)) # large -> streams

    blocks = nn.ModuleList([_Fp8ish()])
    blocks.to("cuda")
    b = blocks[0]
    BlockSwapOffloader(blocks, blocks_to_swap=1, device="cuda")  # __init__ applies the CPU layout
    assert b.weight.device.type == "cpu"   # weight parked
    assert b.scale.device.type == "cuda"   # tiny buffer NOT parked (the fix)
    assert b.big.device.type == "cpu"      # large buffer streams like the weight


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
