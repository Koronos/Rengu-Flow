"""Unit tests for HookBlockSwapOffloader (LRU residency / backward-aware pull), CPU-only.

Uses device='cpu' so the move logic and LRU bookkeeping are exercised without a GPU; residency is
asserted via the offloader's internal LRU set rather than real device moves.
"""

from __future__ import annotations

import torch
from torch import nn

from rengu_flow.training.block_swap import HookBlockSwapOffloader


def _blocks(n: int) -> nn.ModuleList:
    # Each "block" has a couple of leaf modules with parameters (like a resnet's conv1/conv2).
    return nn.ModuleList(
        nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)) for _ in range(n)
    )


def test_disabled_when_zero() -> None:
    off = HookBlockSwapOffloader(_blocks(4), blocks_to_swap=0, device="cpu")
    assert off.enabled is False
    assert off._handles == []


def test_resident_cap_and_lru_eviction() -> None:
    blocks = _blocks(5)
    off = HookBlockSwapOffloader(blocks, blocks_to_swap=3, device="cpu")  # cap = 5 - 3 = 2
    assert off.enabled is True and off.resident_cap == 2
    off._ensure_resident_sync(0)
    off._ensure_resident_sync(1)
    assert set(off._resident) == {0, 1}
    off._ensure_resident_sync(2)  # evicts LRU (0)
    assert set(off._resident) == {1, 2}
    off._ensure_resident_sync(1)  # touch 1 -> most-recently-used
    off._ensure_resident_sync(3)  # evicts LRU (2), not 1
    assert set(off._resident) == {1, 3}


def test_hooks_registered_on_leaf_modules() -> None:
    blocks = _blocks(3)
    off = HookBlockSwapOffloader(blocks, blocks_to_swap=2, device="cpu")
    # 3 blocks * 2 leaf Linears * 2 hooks (fwd-pre + bwd-pre) = 12 handles
    assert len(off._handles) == 12
    # every leaf Linear maps back to its block index
    assert set(off._block_of_module.values()) == {0, 1, 2}


def test_forward_pre_hook_marks_resident() -> None:
    blocks = _blocks(4)
    off = HookBlockSwapOffloader(blocks, blocks_to_swap=3, device="cpu")  # cap = 1
    x = torch.randn(2, 4)
    blocks[2](x)  # running block 2's forward should pull it resident via the hook
    assert 2 in off._resident
    assert len(off._resident) <= off.resident_cap


def test_swap_trainable_false_keeps_trainable_resident_and_disables_prefetch() -> None:
    blocks = _blocks(4)
    # freeze the second Linear of each block; keep the first trainable
    for b in blocks:
        b[1].weight.requires_grad_(False)
        b[1].bias.requires_grad_(False)
    off = HookBlockSwapOffloader(
        blocks, blocks_to_swap=2, device="cpu", prefetch=True, swap_trainable=False
    )
    assert off._swap_trainable is False
    assert off._prefetch is False  # prefetch requires swap_trainable
    # offloading a block must not crash and must leave trainable params trainable
    off._offload_block_to_cpu(blocks[0])
    assert blocks[0][0].weight.requires_grad is True
    assert blocks[0][1].weight.requires_grad is False


def test_swap_trainable_true_moves_whole_block() -> None:
    blocks = _blocks(3)
    off = HookBlockSwapOffloader(blocks, blocks_to_swap=2, device="cpu", swap_trainable=True)
    assert off._swap_trainable is True
    off._offload_block_to_cpu(blocks[0])  # whole-block path; no error on CPU
    assert next(blocks[0].parameters()).device.type == "cpu"


def test_keep_submodel_on_cpu_after_cache_default_false() -> None:
    from rengu_flow.model.base import BasePipeline

    class _M(BasePipeline):
        pass

    assert _M().keep_submodel_on_cpu_after_cache(object()) is False


def test_teardown_removes_hooks() -> None:
    blocks = _blocks(3)
    off = HookBlockSwapOffloader(blocks, blocks_to_swap=2, device="cpu")
    off.teardown()
    assert off._handles == []
    assert off.enabled is False
    # after teardown, running a forward must not touch the (cleared) LRU set
    blocks[0](torch.randn(2, 4))
    assert off._resident == {} or len(off._resident) == 0


class _Krea2StyleBlock(nn.Module):
    """Block that owns a raw Parameter directly and uses it BEFORE any leaf module runs
    (krea2's scale_shift_table). A leaf-only hook set never pulls it resident in time."""

    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.zeros(4))
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x + self.table)


def test_block_root_with_direct_params_is_hooked() -> None:
    blocks = nn.ModuleList(_Krea2StyleBlock() for _ in range(3))
    off = HookBlockSwapOffloader(blocks, blocks_to_swap=2, device="cpu")
    # The block ROOT owns params directly, so it must be hooked alongside its leaves —
    # its forward-pre fires before `x + self.table`, the first use of the direct param.
    assert all(id(b) in off._block_of_module for b in blocks)
    x = torch.randn(2, 4)
    blocks[1](x)
    assert 1 in off._resident
