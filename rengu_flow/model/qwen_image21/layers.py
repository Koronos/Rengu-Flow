"""Pipeline layers for the Qwen-Image 2.1 DiT (DeepSpeed pipe partition / activation checkpointing).

The joint sequence is ``[prompt | target image (h*w)]`` for text-to-image and
``[prompt with the condition images' latent blocks | target]`` for edit batches (the condition
latents fill the prompt's vision slots, in order; see ``QwenImage21Transformer2DModel``). The
inter-layer tuple must be tensors-only, so the non-tensor block metadata is re-derived per layer
from shapes instead of shipped: a zero-storage ``layout`` tensor of shape
``(prefix_len, h, w, start_0, len_0, ..., start_{N-1}, len_{N-1}, 0)`` (host shape metadata, no
device sync) gives the prefill ``segments`` (text runs between the N condition-image blocks, each
block at ``[start_i, start_i + len_i)``) and the modulation mask ``position >= prefix_len`` (the
target is always the trailing ``h*w`` tokens). Text-to-image is the ``N = 0`` case:
``(text_len, h, w, 0)``, one causal text segment. The complex RoPE crosses stages as
``view_as_real`` floats, and an all-valid text mask as a 0-size sentinel.
"""

from __future__ import annotations

import torch
from torch import nn

from rengu_flow.model.base import make_contiguous
from rengu_flow.model.qwen_image21.dit import (
    build_t2i_img_mask,
    pack_latents,
    t2i_img_shapes,
    unpack_latents,
)
from rengu_flow.utils.common import cuda_autocast

# torch's TensorIterator limit is 25 dims; the layout carries 3 + 2N + 1 of them.
MAX_CONDITION_IMAGES = 10


def _layout_dims(segments, prefix_len: int, h: int, w: int) -> tuple[int, ...]:
    """``(prefix_len, h, w, start_0, len_0, ...)`` from the model's prefill segments."""
    blocks = [(start, end - start) for start, end, is_text in segments if not is_text]
    if len(blocks) > MAX_CONDITION_IMAGES:
        raise ValueError(f"qwen_image21 trains with at most {MAX_CONDITION_IMAGES} condition images, got {len(blocks)}")
    return (prefix_len, h, w, *[d for block in blocks for d in block])


def layout_segments(layout: torch.Tensor) -> tuple[int, list[tuple[int, int, bool]]]:
    """``(prefix_len, segments)`` encoded in ``layout``'s shape (inverse of ``_layout_dims``):
    the prefix split into ``(start, end, is_text)`` runs, text between the image blocks."""
    dims = layout.shape[:-1]
    prefix_len = dims[0]
    blocks = [(dims[i], dims[i + 1]) for i in range(3, len(dims), 2)]
    segments = []
    cursor = 0
    for start, length in blocks:
        if start > cursor:
            segments.append((cursor, start, True))
        segments.append((start, start + length, False))
        cursor = start + length
    if cursor < prefix_len:
        segments.append((cursor, prefix_len, True))
    return prefix_len, segments


class InitialLayer(nn.Module):
    """Embeds latents/text/timestep and builds the joint sequence, modulation and RoPE.

    Input: ``(noisy_latents, t, prompt_embeds, text_mask)`` (text-to-image), or those plus
    ``(img_mask, control_latents, control_layout)`` for an edit batch (see
    ``pipeline._edit_inputs``)."""

    def __init__(self, model):
        super().__init__()
        self.img_in = model.img_in
        self.txt_in = model.txt_in
        self.time_text_embed = model.time_text_embed
        self.modulation = model.modulation
        self.pos_embed = model.pos_embed
        self.model = [model]

    def forward(self, inputs):
        with cuda_autocast():
            noisy_latents, t, prompt_embeds, text_mask = inputs[:4]
            text_mask = text_mask.bool()
            bs, _, h, w = noisy_latents.shape
            text_len = prompt_embeds.shape[1]
            # All-valid masks take the maskless path (and skip the per-row RoPE build).
            encoder_mask = None if bool(text_mask.all()) else text_mask
            if len(inputs) == 4:
                hidden = pack_latents(noisy_latents)
                img_shapes = t2i_img_shapes(h, w, bs)
                img_mask = build_t2i_img_mask(text_len, h, w, bs, device=noisy_latents.device)
            else:
                img_mask, control_latents, control_layout = inputs[4:]
                grids = control_layout.shape[1:-1]  # (B, h_0, w_0, ..., 0)
                control_shapes = [(1, grids[i], grids[i + 1]) for i in range(0, len(grids), 2)]
                hidden = torch.cat([control_latents.to(noisy_latents.dtype), pack_latents(noisy_latents)], dim=1)
                img_shapes = [[*control_shapes, (1, h, w)]] * bs
                img_mask = img_mask.bool()
            prepared = self.model[0].prepare_inputs(
                hidden, prompt_embeds, t.view(-1), img_shapes, img_mask, encoder_mask
            )
            key_valid = prepared.key_valid
            if key_valid is None:
                key_valid = text_mask.new_empty(0)
            rope = torch.view_as_real(prepared.rotary_emb)
            dims = _layout_dims(prepared.segments, prepared.prefix_len, h, w)
            layout = prepared.hidden_states.new_empty((*dims, 0))
            outputs = make_contiguous(
                prepared.hidden_states, prepared.temb, prepared.modulation, rope, key_valid, layout
            )
            # The RoPE table is a constant (no parameter behind it): leave it out, or every layer
            # would backprop into a leaf nobody reads.
            for tensor in (outputs[0], outputs[1], outputs[2]):
                tensor.requires_grad_(True)
            return outputs


def _target_token_mask(seq_len: int, prefix_len: int, device) -> torch.Tensor:
    return torch.arange(seq_len, device=device) >= prefix_len


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    def forward(self, inputs):
        with cuda_autocast():
            hidden, temb, modulation, rope, key_valid, layout = inputs
            prefix_len, segments = layout_segments(layout)

            self.offloader.wait_for_block(self.block_idx)
            hidden = self.block(
                hidden,
                modulation,
                # Reentrant AC hands pass-through outputs back requiring grad: detach the
                # constant table so blocks don't backprop into it (see krea2/layers.py).
                rotary_emb=torch.view_as_complex(rope.detach()),
                target_token_mask=_target_token_mask(hidden.shape[1], prefix_len, hidden.device),
                segments=segments,
                key_valid=key_valid if key_valid.numel() else None,  # 0-size sentinel = no padding
            )
            self.offloader.submit_move_blocks_forward(self.block_idx)

            return make_contiguous(hidden, temb, modulation, rope, key_valid, layout)


class FinalLayer(nn.Module):
    """Adaptive norm + projection over the target tokens only (the loss never sees the prompt
    or the condition images)."""

    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out

    def forward(self, inputs):
        with cuda_autocast():
            hidden, temb, _modulation, _rope, _key_valid, layout = inputs
            prefix_len, h, w = layout.shape[:3]
            target = hidden[:, prefix_len:]
            # All-target mask: every row takes its own sample's timestep (not the t=0 row).
            mask = torch.ones(target.shape[1], dtype=torch.bool, device=target.device)
            output = self.proj_out(self.norm_out(target, temb, mask))
            return unpack_latents(output, h, w)
