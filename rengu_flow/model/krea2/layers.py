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
            hidden, temb, temb_mod, freqs_cos, freqs_sin, attn_mask, text_mask, grid = inputs

            self.offloader.wait_for_block(self.block_idx)
            hidden = self.block(hidden, temb_mod, (freqs_cos, freqs_sin), attn_mask)
            self.offloader.submit_move_blocks_forward(self.block_idx)

            return make_contiguous(hidden, temb, temb_mod, freqs_cos, freqs_sin, attn_mask, text_mask, grid)


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
