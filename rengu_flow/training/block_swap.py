"""CPU block swap during training and inference (shared by SDXL UNet and Cosmos DiT)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class BlockSwapHandler(Protocol):
    def wait_for_block(self, block_idx: int) -> None: ...

    def submit_move_blocks_forward(self, block_idx: int) -> None: ...


class NoopOffloader:
    """Placeholder when block swap is disabled."""

    @property
    def enabled(self) -> bool:
        return False

    def wait_for_block(self, block_idx: int) -> None:
        pass

    def submit_move_blocks_forward(self, block_idx: int) -> None:
        pass

    def teardown(self) -> None:
        pass

    def apply_training_layout(self) -> None:
        pass


class BlockSwapOffloader:
    """Keep swapable blocks on CPU between forward steps; move one block to GPU per step."""

    def __init__(
        self,
        blocks: list[nn.Module] | nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device | str = "cuda",
    ):
        self.blocks: list[nn.Module] = list(blocks)
        self.num_blocks = len(self.blocks)
        self.blocks_to_swap = min(max(int(blocks_to_swap), 0), self.num_blocks)
        self.device = torch.device(device)
        self._enabled = self.blocks_to_swap > 0
        if self._enabled:
            self.apply_training_layout()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def apply_training_layout(self) -> None:
        if not self._enabled:
            return
        for block in self.blocks:
            block.to("cpu")

    def wait_for_block(self, block_idx: int) -> None:
        if not self._enabled:
            return
        self.blocks[block_idx].to(self.device, non_blocking=True)

    def submit_move_blocks_forward(self, block_idx: int) -> None:
        if not self._enabled:
            return
        self.blocks[block_idx].to("cpu", non_blocking=True)

    def teardown(self) -> None:
        if not self._enabled:
            return
        for block in self.blocks:
            block.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()


class HookBlockSwapOffloader:
    """On-demand, backward-aware CPU<->GPU block swap driven by module hooks.

    Unlike ``BlockSwapOffloader`` (which relies on a model's pipeline-layer ``forward`` calling
    ``wait_for_block``/``submit_move_blocks_forward``), this offloader registers hooks directly on
    the leaf modules of each swappable block, so it works even when ``to_layers`` flattens a block
    into several pipeline layers (as SDXL's UNet does). It keeps at most
    ``num_blocks - blocks_to_swap`` blocks resident on the GPU; a forward-pre and full-backward-pre
    hook pulls a block onto the GPU before it runs (covering activation-checkpointing recompute),
    and an LRU policy evicts the least-recently-used resident block to CPU when the cap is exceeded.

    Intended for **full fine-tuning with gradient_release**: each parameter's optimizer step runs
    inside the backward (``register_post_accumulate_grad_hook``) while its block is resident, so the
    weights are updated on-GPU before the block is evicted. A monolithic ``optimizer.step()`` would
    instead need every trainable block resident at once, which defeats the swap — hence the gate.
    Initial CPU layout is applied by ``apply_training_layout`` (call it *after* DeepSpeed has placed
    the model on the GPU, e.g. via ``prepare_block_swap_training``).
    """

    def __init__(
        self,
        blocks: list[nn.Module] | nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device | str = "cuda",
    ):
        self.blocks: list[nn.Module] = list(blocks)
        self.num_blocks = len(self.blocks)
        self.blocks_to_swap = min(max(int(blocks_to_swap), 0), self.num_blocks)
        self.resident_cap = max(1, self.num_blocks - self.blocks_to_swap)
        self.device = torch.device(device)
        self._enabled = self.blocks_to_swap > 0
        self._resident: "OrderedDict[int, None]" = OrderedDict()
        self._block_of_module: dict[int, int] = {}
        self._handles: list = []
        if self._enabled:
            self._register_hooks()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _register_hooks(self) -> None:
        for idx, block in enumerate(self.blocks):
            for module in block.modules():
                if module is block:
                    continue
                if next(module.parameters(recurse=False), None) is None:
                    continue  # only modules that directly own parameters actually run/transfer
                self._block_of_module[id(module)] = idx
                self._handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
                self._handles.append(module.register_full_backward_pre_hook(self._backward_pre_hook))

    def _ensure_resident(self, block_idx: int) -> None:
        if block_idx in self._resident:
            self._resident.move_to_end(block_idx)
            return
        self.blocks[block_idx].to(self.device)
        self._resident[block_idx] = None
        while len(self._resident) > self.resident_cap:
            evict_idx, _ = self._resident.popitem(last=False)
            self.blocks[evict_idx].to("cpu")

    def _forward_pre_hook(self, module: nn.Module, args) -> None:
        self._ensure_resident(self._block_of_module[id(module)])

    def _backward_pre_hook(self, module: nn.Module, grad_output) -> None:
        self._ensure_resident(self._block_of_module[id(module)])

    def apply_training_layout(self) -> None:
        """Push every swappable block to CPU; blocks are pulled back on demand by the hooks."""
        if not self._enabled:
            return
        for block in self.blocks:
            block.to("cpu")
        self._resident.clear()

    def teardown(self) -> None:
        """Remove hooks and move all blocks back to the GPU (for full-model eval / save)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for block in self.blocks:
            block.to(self.device)
        self._resident.clear()
        self._enabled = False

    # diffusion-pipe-style API compatibility: this offloader uses hooks, so the layer-driven
    # wait/submit calls (used by Cosmos's TransformerLayer.forward) are no-ops here.
    def wait_for_block(self, block_idx: int) -> None:
        pass

    def submit_move_blocks_forward(self, block_idx: int) -> None:
        pass
