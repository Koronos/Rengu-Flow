"""Block swap on the single-GPU torch engine (engine='accelerate'), LoRA-style.

Proves the hook offloader — written for the DeepSpeed pipeline engine — also drives placement when
the plain TorchEngine runs the forward/backward: frozen base weights stream CPU<->GPU on demand while
the small trainable adapters stay GPU-resident, and the whole model is never hauled onto the GPU at
once. CUDA-gated because the whole point is device placement.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rengu_flow.engine import SequentialPipe, TorchEngine
from rengu_flow.training.block_swap import HookBlockSwapOffloader


class _AdapterBlock(nn.Module):
    """A frozen base weight (swappable) plus a tiny trainable adapter (kept resident) — like LoRA."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.lin.weight.requires_grad_(False)            # frozen base -> swaps to CPU
        self.adapter = nn.Parameter(torch.zeros(dim))    # trainable -> stays on GPU

    def forward(self, t):
        (x,) = t
        return (self.lin(x) + self.adapter,)


def _loss_fn(out, label):
    (y,) = out
    (target,) = label
    return ((y - target) ** 2).mean()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block swap placement needs CUDA")
def test_torch_engine_lora_block_swap_keeps_residency_capped():
    dim, num_blocks, blocks_to_swap = 8, 4, 3
    blocks = nn.ModuleList([_AdapterBlock(dim) for _ in range(num_blocks)])

    # Mirror model.enable_block_swap(...) for adapter training: swap_trainable=False.
    offloader = HookBlockSwapOffloader(blocks, blocks_to_swap, device="cuda", swap_trainable=False)
    resident_cap = num_blocks - blocks_to_swap

    module = SequentialPipe(list(blocks), _loss_fn)
    adapters = [b.adapter for b in blocks]
    engine = TorchEngine(
        module,
        lambda params: torch.optim.SGD(params, lr=0.5),
        adapters,
        {"gradient_accumulation_steps": 1, "gradient_clipping": 1.0},
        block_swap=True,  # must NOT haul the module onto the GPU
    )

    # With block_swap=True the engine left placement alone; the model is still all on CPU.
    assert all(b.lin.weight.device.type == "cpu" for b in blocks), "engine pre-moved frozen weights"

    # prepare_block_swap_training() equivalent: push blocks to CPU, adapters stay resident on GPU.
    offloader.apply_training_layout()
    assert all(b.adapter.device.type == "cuda" for b in blocks), "trainable adapter must stay resident"

    before = torch.stack([b.adapter.detach().cpu().clone() for b in blocks])
    x = torch.randn(2, dim, device="cuda")
    target = torch.randn(2, dim, device="cuda")
    loss = engine.train_batch(iter([((x,), (target,))]))

    assert torch.isfinite(loss).all()
    assert engine.get_global_grad_norm() is not None and engine.get_global_grad_norm() > 0
    after = torch.stack([b.adapter.detach().cpu().clone() for b in blocks])
    assert not torch.allclose(before, after), "adapters did not update — backward/step did not run"

    # The swap actually limited residency: frozen weights resident must not exceed the cap, i.e. at
    # least `blocks_to_swap` of them got evicted to CPU. (Defeating the swap would leave all on GPU.)
    resident_frozen = sum(b.lin.weight.device.type == "cuda" for b in blocks)
    assert resident_frozen <= resident_cap, f"{resident_frozen} frozen blocks resident > cap {resident_cap}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block swap placement needs CUDA")
def test_torch_engine_lora_block_swap_prefetch():
    """Adapter-mode prefetch: pinned frozen weights stream on a side stream while adapters stay
    resident. Validates the gate now allows it (swap_trainable=False) and the frozen-weight
    'immutable master, no copy-back' eviction keeps the base weights bit-exact."""
    dim, num_blocks, blocks_to_swap = 8, 4, 2  # cap=2 → prefetch eligible (needs >=2 resident)
    blocks = nn.ModuleList([_AdapterBlock(dim) for _ in range(num_blocks)])
    frozen_ref = [b.lin.weight.detach().clone() for b in blocks]

    offloader = HookBlockSwapOffloader(
        blocks, blocks_to_swap, device="cuda", prefetch=True, swap_trainable=False
    )
    assert offloader._prefetch is True, "prefetch must engage for adapter runs with cap>=2"

    module = SequentialPipe(list(blocks), _loss_fn)
    adapters = [b.adapter for b in blocks]
    engine = TorchEngine(
        module,
        lambda params: torch.optim.SGD(params, lr=0.5),
        adapters,
        {"gradient_accumulation_steps": 1, "gradient_clipping": 1.0},
        block_swap=True,
    )
    offloader.apply_training_layout()

    before = torch.stack([b.adapter.detach().cpu().clone() for b in blocks])
    x = torch.randn(2, dim, device="cuda")
    target = torch.randn(2, dim, device="cuda")
    loss = engine.train_batch(iter([((x,), (target,))]))

    assert torch.isfinite(loss).all()
    after = torch.stack([b.adapter.detach().cpu().clone() for b in blocks])
    assert not torch.allclose(before, after), "adapters did not update under prefetch"
    # Frozen weights are never copied back (immutable master); they must remain bit-exact.
    for b, ref in zip(blocks, frozen_ref):
        assert torch.equal(b.lin.weight.detach().cpu(), ref.cpu()), "frozen weight mutated under prefetch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block swap placement needs CUDA")
def test_torch_engine_lora_block_swap_pinned_without_overlap():
    """Maximal swap (cap=1): overlap can't engage (needs 2 co-resident blocks), but pinning still
    applies — DMA pulls from page-locked masters at no extra VRAM. Validates _pin WITHOUT _prefetch."""
    dim, num_blocks, blocks_to_swap = 8, 3, 3  # cap = max(1, 3 - 3) = 1
    blocks = nn.ModuleList([_AdapterBlock(dim) for _ in range(num_blocks)])
    frozen_ref = [b.lin.weight.detach().clone() for b in blocks]

    off = HookBlockSwapOffloader(
        blocks, blocks_to_swap, device="cuda", prefetch=True, swap_trainable=False
    )
    assert off._pin is True, "pinning must stay on even at cap=1"
    assert off._prefetch is False, "overlap must be off at cap=1 (no room for a second block)"

    module = SequentialPipe(list(blocks), _loss_fn)
    engine = TorchEngine(
        module,
        lambda p: torch.optim.SGD(p, lr=0.5),
        [b.adapter for b in blocks],
        {"gradient_accumulation_steps": 1, "gradient_clipping": 1.0},
        block_swap=True,
    )
    off.apply_training_layout()

    before = torch.stack([b.adapter.detach().cpu().clone() for b in blocks])
    x = torch.randn(2, dim, device="cuda")
    target = torch.randn(2, dim, device="cuda")
    loss = engine.train_batch(iter([((x,), (target,))]))

    assert torch.isfinite(loss).all()
    after = torch.stack([b.adapter.detach().cpu().clone() for b in blocks])
    assert not torch.allclose(before, after), "adapters did not update under pinned sync"
    for b, ref in zip(blocks, frozen_ref):
        assert torch.equal(b.lin.weight.detach().cpu(), ref.cpu()), "frozen weight mutated under pinned sync"
