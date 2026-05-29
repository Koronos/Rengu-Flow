"""Backward-compatible re-export; use rengu_flow.training.block_swap.BlockSwapOffloader."""

from rengu_flow.training.block_swap import BlockSwapOffloader

# Legacy name used in docs and tests.
CosmosBlockOffloader = BlockSwapOffloader

__all__ = ["BlockSwapOffloader", "CosmosBlockOffloader"]
