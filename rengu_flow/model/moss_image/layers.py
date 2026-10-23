"""Pipeline layers for the MOSS-Image DiT (DeepSpeed pipe partition / activation checkpointing).

``InitialLayer`` -> 42 x ``TransformerLayer`` -> ``FinalLayer``; the stack is the DiT's own pieces
(``embed`` / ``Block`` / ``final``) cut at block boundaries, run over a right-padded text batch
with per-sample RoPE positions and key masking (see ``dit.py``), which is exactly the reference's
per-sample forward.

Inter-layer tuple (tensors only): ``(x, modulation, head_modulation, rope, key_valid | 0-size,
layout)``. ``layout`` is a zero-storage tensor of shape ``(h, w, 0)`` carrying the latent grid as
host shape metadata; the RoPE table and the key mask are constants (never backpropagated into).
The model time input is ``1000 * sigma`` (the features carry ``sigma``).
"""

from __future__ import annotations

import torch
from torch import nn

from rengu_flow.model.base import make_contiguous
from rengu_flow.model.dit_common.layers import OffloadedBlockLayer, mark_inter_layer_grads
from rengu_flow.utils.common import cuda_autocast

TIME_SCALE = 1000.0


class InitialLayer(nn.Module):
    """Embeds latents / text / time and builds modulation, RoPE and the key mask.

    Input: ``(noisy_latents (B, 128, h, w), sigma (B, 1), prompt_embeds (B, L, 4096), text_mask (B, L))``."""

    def __init__(self, model, pipe_parallel: bool = False):
        super().__init__()
        self.video_embedder = model.video_embedder
        self.text_embedder = model.text_embedder
        self.time_embedding = model.time_embedding
        self.shared_block_modulation = model.shared_block_modulation
        self.video_head_modulation = model.video_head_modulation
        self.rope = model.rope
        self.model = [model]
        # DeepSpeed pipe (> 1 stage) backprops every floating inter-stage tensor, so they must all require
        # grad there; single-stage marks only what carries gradient.
        self.pipe_parallel = pipe_parallel

    def forward(self, inputs):
        with cuda_autocast():
            noisy, sigma, embeds, text_mask = inputs
            _, _, h, w = noisy.shape
            x, mod, head, rope, key_valid = self.model[0].embed(noisy, sigma.view(-1) * TIME_SCALE, embeds, text_mask.bool())
            if key_valid is None:
                key_valid = text_mask.new_empty(0, dtype=torch.bool)
            layout = x.new_empty((h, w, 0))
            outputs = make_contiguous(x, mod, head, rope, key_valid, layout)
            # Single stage: the RoPE table and the mask are constants, left out of the grad carriers.
            mark_inter_layer_grads(outputs, self.pipe_parallel, outputs[:3])
            return outputs


class TransformerLayer(OffloadedBlockLayer):
    def run_block(self, inputs):
        x, mod, _head, rope, key_valid, _layout = inputs
        # Reentrant AC hands pass-through outputs back requiring grad: detach the constant table.
        return (self.block(x, rope.detach(), mod, key_valid if key_valid.numel() else None),)


class FinalLayer(nn.Module):
    """Final norm + head modulation + projection over the image tokens (velocity)."""

    def __init__(self, model):
        super().__init__()
        self.final_norm_video = model.final_norm_video
        self.final_linear_video = model.final_linear_video
        self.model = [model]

    def forward(self, inputs):
        with cuda_autocast():
            x, _mod, head, _rope, _key_valid, layout = inputs
            h, w = layout.shape[:2]
            return self.model[0].final(x, head, h, w)
