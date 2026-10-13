"""Pipeline-layer building blocks shared by the DiT pipelines (DeepSpeed pipe partition /
activation checkpointing).

What a layer carries between stages (the inter-layer tuple) is model-specific and stays in the
model's ``layers.py``; this module holds only the wrapper every per-block layer repeats and the
``requires_grad`` marking rule of the first layer.
"""

from __future__ import annotations

import torch
from torch import nn

from rengu_flow.model.base import make_contiguous
from rengu_flow.utils.common import cuda_autocast


def mark_inter_layer_grads(outputs, pipe_parallel: bool, carriers) -> None:
    """Make the first layer's floating outputs require grad.

    DeepSpeed pipe (> 1 stage) backprops every floating inter-stage tensor, so with
    ``pipe_parallel`` all of ``outputs`` are marked; single-stage marks only ``carriers`` (what
    actually carries gradient — never a constant RoPE table, or every block would backprop into
    a leaf nobody reads)."""
    for tensor in outputs if pipe_parallel else carriers:
        if tensor is not None and torch.is_floating_point(tensor):
            tensor.requires_grad_(True)


class OffloadedBlockLayer(nn.Module):
    """One DiT block as a pipeline layer.

    ``forward`` runs ``run_block(inputs)`` under ``cuda_autocast`` between the block-swap
    offloader's ``wait_for_block`` / ``submit_move_blocks_forward`` (skipped when ``offloader`` is
    ``None``). ``run_block`` returns the leading items of the new inter-layer tuple; they are made
    contiguous and the remaining input items (e.g. a TREAD route window) pass through untouched.
    Subclasses keep the name ``TransformerLayer`` so ``checkpointable_layers`` still matches.
    """

    def __init__(self, block, block_idx: int, offloader=None):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    def run_block(self, inputs) -> tuple:
        """The block's forward over ``inputs``: the new leading items of the tuple."""
        raise NotImplementedError

    def forward(self, inputs):
        with cuda_autocast():
            if self.offloader is not None:
                self.offloader.wait_for_block(self.block_idx)
            outputs = tuple(self.run_block(inputs))
            if self.offloader is not None:
                self.offloader.submit_move_blocks_forward(self.block_idx)
            return make_contiguous(*outputs) + tuple(inputs[len(outputs) :])
