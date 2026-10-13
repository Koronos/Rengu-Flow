"""Pipeline layers for the Nanosaur2 DiT (DeepSpeed pipe partition / activation checkpointing).

Layer order: ``InitialLayer`` -> dense blocks ``0..F-1`` -> ``SparseStartLayer`` -> sparse
blocks ``F..N-H-1`` -> ``SparseEndLayer`` -> dense blocks ``N-H..N-1`` -> ``FinalLayer``. Every
layer runs the DiT's own pieces (``embed`` / ``run_block`` / ``final``), so the stack is the
monolithic ``forward`` cut at block boundaries.

Inter-layer tuple (tensors only): ``(s, txt, pe, mod, txt_bias, condition, x, t, sparse_skip)``.

SPRINT path-drop (``sparse_skip``, one bool per row): rows that skip keep their activation from
before the sparse blocks. Single-stage runs (``subset=True``) send only the non-skipping rows
through the sparse blocks (no compute for skipped rows; with no row skipping the sparse window is
a plain pass over the batch); the window carries the full-batch tensors as 5 extra items and
``SparseEndLayer`` scatters the result back. Pipeline-parallel runs (``subset=False``) need fixed
inter-stage shapes, so they run every row and select with ``torch.where``.
"""

from __future__ import annotations

import torch
from torch import nn

from rengu_flow.model.base import make_contiguous
from rengu_flow.model.dit_common.layers import OffloadedBlockLayer, mark_inter_layer_grads
from rengu_flow.model.nanosaur2.dit import run_block
from rengu_flow.utils.common import cuda_autocast

BASE_ITEMS = 9


class InitialLayer(nn.Module):
    """Text refinement + conditioning + image embedding (``Nanosaur2Transformer2DModel.embed``)."""

    def __init__(self, model, pipe_parallel: bool = False):
        super().__init__()
        self.s_embedder = model.s_embedder
        self.t_embedder = model.t_embedder
        self.y_embedder = model.y_embedder
        self.shared_encoder_adaLN = model.shared_encoder_adaLN
        self.y_pool_proj = model.y_pool_proj
        self.text_refine_blocks = model.text_refine_blocks
        self.model = [model]
        # DeepSpeed pipe (> 1 stage) backprops every floating inter-stage tensor, so they must
        # all require grad there; single-stage marks only what carries gradient.
        self.pipe_parallel = pipe_parallel

    def forward(self, inputs):
        with cuda_autocast():
            x, t, context, token_weights, sparse_skip = inputs
            t = t.view(-1)
            s, txt, pe, mod, txt_bias, condition = self.model[0].embed(x, t, context, token_weights)
            outputs = make_contiguous(s, txt, pe, mod, txt_bias, condition, x, t, sparse_skip.bool())
            mark_inter_layer_grads(outputs, self.pipe_parallel, (outputs[0], outputs[1], outputs[3], outputs[5]))
            return outputs


class TransformerLayer(OffloadedBlockLayer):
    """Image block ``block_idx`` (``run_block``); extra window items pass through."""

    def __init__(self, model, block_idx: int):
        super().__init__(model.blocks[block_idx], block_idx)
        self.adaLN_offset = model.encoder_adaLN_offsets[block_idx]

    def run_block(self, inputs):
        s, txt, pe, mod, txt_bias = inputs[:5]
        if s.shape[0] == 0:  # every row of the micro-batch skips the sparse path
            return ()
        # Detached: reentrant AC (and pipe stages) hand the passthrough RoPE table back
        # requiring grad; the block must treat it as a constant.
        return (run_block(self.block, self.adaLN_offset, s, txt, pe.detach(), mod, txt_bias),)


class SparseStartLayer(nn.Module):
    """Enter the SPRINT sparse window: keep the dense activation for the residual and, in subset
    mode, narrow the batch to the rows that run the sparse blocks."""

    def __init__(self, subset: bool = True):
        super().__init__()
        self.subset = subset

    def forward(self, inputs):
        s, txt, pe, mod, txt_bias, condition, x, t, sparse_skip = inputs
        if not self.subset or not bool(sparse_skip.any()):
            return tuple(inputs) + (s,)
        rows = (~sparse_skip).nonzero().squeeze(1)
        narrowed = (
            s.index_select(0, rows),
            txt.index_select(0, rows),
            pe,
            mod.index_select(0, rows),
            txt_bias.index_select(0, rows),
        )
        return make_contiguous(*narrowed) + (condition, x, t, sparse_skip) + (s, txt, mod, txt_bias, rows)


class SparseEndLayer(nn.Module):
    """Leave the sparse window: ``s + sprint_out_proj(g - s)`` for the rows that ran it; skipped
    rows keep ``s`` (the dense activation from before the window)."""

    def __init__(self, model, subset: bool = True):
        super().__init__()
        self.sprint_out_proj = model.sprint_out_proj
        self.subset = subset

    def forward(self, inputs):
        with cuda_autocast():
            g = inputs[0]
            sparse_skip = inputs[8]
            if len(inputs) == BASE_ITEMS + 1:
                dense = inputs[BASE_ITEMS]
                out = dense + self.sprint_out_proj(g - dense)
                if not self.subset:
                    out = torch.where(sparse_skip.view(-1, 1, 1), dense.to(out.dtype), out)
                return (out.contiguous(),) + tuple(inputs[1:BASE_ITEMS])
            pe, condition, x, t = inputs[2], inputs[5], inputs[6], inputs[7]
            dense, txt, mod, txt_bias, rows = inputs[BASE_ITEMS:]
            selected = dense.index_select(0, rows)
            update = selected + self.sprint_out_proj(g - selected)
            out = dense.to(update.dtype).index_copy(0, rows, update)
            return make_contiguous(out, txt, pe, mod, txt_bias, condition, x, t, sparse_skip)


class FinalLayer(nn.Module):
    """x-prediction head -> velocity ``(x - x0) / t`` (float32)."""

    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def forward(self, inputs):
        with cuda_autocast():
            s, _txt, _pe, _mod, _txt_bias, condition, x, t, _skip = inputs
            return self.model[0].final(s, condition, x, t)
