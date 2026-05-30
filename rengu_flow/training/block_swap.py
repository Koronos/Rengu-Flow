"""CPU block swap during training and inference (shared by SDXL UNet and Cosmos DiT)."""

from __future__ import annotations

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
