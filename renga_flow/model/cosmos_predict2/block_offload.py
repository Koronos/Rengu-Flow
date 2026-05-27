"""CPU block swap for Cosmos DiT preview inference (not training)."""

from __future__ import annotations

import torch


class CosmosBlockOffloader:
    """Move DiT blocks to CPU between preview forward steps to reduce peak VRAM."""

    def __init__(
        self,
        blocks: torch.nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device | str = "cuda",
    ):
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.blocks_to_swap = min(max(int(blocks_to_swap), 0), self.num_blocks)
        self.device = torch.device(device)
        self._enabled = self.blocks_to_swap > 0

        if self._enabled:
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
        torch.cuda.current_stream().synchronize()
