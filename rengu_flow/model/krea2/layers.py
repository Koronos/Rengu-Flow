"""Pipeline layers for the Krea 2 DiT (DeepSpeed pipe partition / activation checkpointing)."""

from __future__ import annotations

import torch
from torch import nn

from rengu_flow.model.base import make_contiguous
from rengu_flow.model.krea2.dit import pack_latents, prepare_position_ids, unpack_latents
from rengu_flow.utils.common import cuda_autocast


class InitialLayer(nn.Module):
    """Embeds text (fusion + projection), packs image latents, and builds timestep/RoPE tensors."""

    def __init__(self, model):
        super().__init__()
        self.img_in = model.img_in
        self.time_embed = model.time_embed
        self.time_mod_proj = model.time_mod_proj
        self.text_fusion = model.text_fusion
        self.txt_in = model.txt_in
        self.rotary_emb = model.rotary_emb
        self.model = [model]

    def forward(self, inputs):
        with cuda_autocast():
            noisy_latents, t, prompt_embeds, text_mask = inputs
            text_mask = text_mask.bool()
            bs, _, h, w = noisy_latents.shape
            grid_h, grid_w = h // 2, w // 2
            image_seq_len = grid_h * grid_w

            temb = self.time_embed(t.view(-1), dtype=noisy_latents.dtype)
            temb_mod = self.time_mod_proj(torch.nn.functional.gelu(temb, approximate="tanh"))

            text_attn_mask, attn_mask = self.model[0].build_attention_masks(text_mask, image_seq_len)
            text_states = self.txt_in(self.text_fusion(prompt_embeds, attention_mask=text_attn_mask))

            hidden = self.img_in(pack_latents(noisy_latents))
            hidden = torch.cat([text_states, hidden], dim=1)

            position_ids = prepare_position_ids(text_mask.shape[1], grid_h, grid_w, hidden.device)
            freqs_cos, freqs_sin = self.rotary_emb(position_ids)

            # All-valid masks come back as None (fused-SDPA fast path); the inter-layer tuple
            # must stay tensors-only (DeepSpeed pipe comm), so ship a 0-size sentinel instead.
            if attn_mask is None:
                attn_mask = text_mask.new_empty(0)
            grid = torch.tensor([grid_h, grid_w], device=hidden.device)
            outputs = make_contiguous(hidden, temb, temb_mod, freqs_cos, freqs_sin, attn_mask, text_mask, grid)
            for tensor in outputs:
                if torch.is_floating_point(tensor):
                    tensor.requires_grad_(True)
            return outputs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    def forward(self, inputs):
        with cuda_autocast():
            # Items past the base 8 are route-window state (see RouteStartLayer): pass through.
            hidden, temb, temb_mod, freqs_cos, freqs_sin, attn_mask, text_mask, grid = inputs[:8]

            self.offloader.wait_for_block(self.block_idx)
            mask = attn_mask if attn_mask.numel() else None  # 0-size sentinel = no padding
            hidden = self.block(hidden, temb_mod, (freqs_cos, freqs_sin), mask)
            self.offloader.submit_move_blocks_forward(self.block_idx)

            return make_contiguous(hidden, temb, temb_mod, freqs_cos, freqs_sin, attn_mask, text_mask, grid) + tuple(inputs[8:])


class RouteStartLayer(nn.Module):
    """TREAD route entry: shrink the sequence to text + a random subset of image tokens.

    Appends the pre-route state (full hidden/rope/mask + keep index) to the inter-layer
    tuple so RouteEndLayer can scatter back. No-op (8-tuple passthrough) outside
    training-with-grad, so eval/preview/val probes always see the full sequence.
    """

    def __init__(self, drop_ratio: float, disable_after_frac: float = 1.0):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.disable_after_frac = disable_after_frac
        self._progress = 0.0

    def set_training_progress(self, frac: float) -> None:
        """Run fraction [0, 1], pushed by the training loop (TREAD off-ramp)."""
        self._progress = float(frac)

    def forward(self, inputs):
        from rengu_flow.training.token_routing import route_start, sample_keep_index

        # Gate on training mode ONLY (eval/probes/previews call module.eval()). Gating on
        # torch.is_grad_enabled() too made grad-mode an extra shape-divergence axis for the
        # compiled blocks downstream (reentrant AC flips grad mode within one step).
        if not self.training:
            return inputs
        if self._progress >= self.disable_after_frac:
            return inputs  # off-ramp: final stretch trains on the full sequence
        hidden, temb, temb_mod, freqs_cos, freqs_sin, attn_mask, text_mask, grid = inputs
        text_len = text_mask.shape[1]
        keep_idx = sample_keep_index(
            text_len, hidden.shape[1] - text_len, self.drop_ratio, hidden.device
        )
        routed = route_start(hidden, keep_idx)
        cos_r = freqs_cos.index_select(0, keep_idx)
        sin_r = freqs_sin.index_select(0, keep_idx)
        mask_r = attn_mask[..., keep_idx] if attn_mask.numel() else attn_mask
        return (
            routed.contiguous(), temb, temb_mod, cos_r, sin_r, mask_r, text_mask, grid,
            hidden, freqs_cos, freqs_sin, attn_mask, keep_idx,
        )


class RouteEndLayer(nn.Module):
    """TREAD route exit: scatter processed tokens over the saved pre-route sequence
    (identity bypass) and restore the full-sequence rope/mask."""

    def forward(self, inputs):
        from rengu_flow.training.token_routing import route_end

        if len(inputs) == 8:  # RouteStartLayer was a no-op
            return inputs
        routed, temb, temb_mod, _cos, _sin, _mask, text_mask, grid = inputs[:8]
        full_hidden, freqs_cos, freqs_sin, attn_mask, keep_idx = inputs[8:]
        hidden = route_end(routed, full_hidden, keep_idx)
        return make_contiguous(
            hidden, temb, temb_mod, freqs_cos, freqs_sin, attn_mask, text_mask, grid
        )


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer

    def forward(self, inputs):
        with cuda_autocast():
            hidden, temb, _temb_mod, _freqs_cos, _freqs_sin, _attn_mask, text_mask, grid = inputs
            hidden = hidden[:, text_mask.shape[1] :]
            output = self.final_layer(hidden, temb)
            return unpack_latents(output, int(grid[0].item()), int(grid[1].item()))
