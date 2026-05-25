"""DeepSpeed pipeline layers for Cosmos Predict2 DiT."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from renga_flow.model.base import make_contiguous
from renga_flow.utils.common import AUTOCAST_DTYPE, is_main_process


class NoopOffloader:
    """Placeholder when block swap is disabled."""

    def wait_for_block(self, block_idx):
        pass

    def submit_move_blocks_forward(self, block_idx):
        pass


def tokenize(tokenizer, prompts):
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def compute_text_embeddings(text_encoder, input_ids, attn_mask, is_generic_llm=False):
    input_ids = input_ids.to(text_encoder.device)
    attn_mask = attn_mask.to(text_encoder.device)
    outputs = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state
    encoded_text[~attn_mask.bool()] = 0
    return encoded_text


class InitialLayer(nn.Module):
    def __init__(self, model, text_encoder, is_generic_llm):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.pos_embedder = model.pos_embedder
        if model.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = model.extra_pos_embedder
        self.t_embedder = model.t_embedder
        self.t_embedding_norm = model.t_embedding_norm
        self.text_encoder = text_encoder
        self.model = [model]
        self.is_generic_llm = is_generic_llm

    @torch.autocast("cuda", dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_C_T_H_W, timesteps_B_T, *prompt_embeds_or_batch_encoding = inputs

        if torch.is_floating_point(prompt_embeds_or_batch_encoding[0]):
            crossattn_emb, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
        else:
            with torch.no_grad():
                input_ids, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
                crossattn_emb = compute_text_embeddings(
                    self.text_encoder, input_ids, attn_mask, is_generic_llm=self.is_generic_llm
                )

        padding_mask = torch.zeros(
            x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4],
            dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device,
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.model[0].prepare_embedded_sequence(
            x_B_C_T_H_W, fps=None, padding_mask=padding_mask,
        )
        assert extra_pos_emb is None
        assert rope_emb_L_1_1_D is not None

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        outputs = make_contiguous(
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, t5_input_ids, attn_mask,
            t5_attn_mask, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
        )
        for tensor in outputs:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)
        return outputs


class LLMAdapterLayer(nn.Module):
    def __init__(self, llm_adapter):
        super().__init__()
        self.llm_adapter = llm_adapter

    @torch.autocast("cuda", dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        (
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, t5_input_ids, attn_mask,
            t5_attn_mask, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
        ) = inputs

        if self.llm_adapter is not None:
            crossattn_emb = self.llm_adapter(
                source_hidden_states=crossattn_emb,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
            crossattn_emb[~t5_attn_mask.bool()] = 0

        return make_contiguous(
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D,
            adaln_lora_B_T_3D, timesteps_B_T,
        )


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast("cuda", dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T = inputs
        self.offloader.wait_for_block(self.block_idx)
        x_B_T_H_W_D = self.block(
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)
        return make_contiguous(
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D,
            adaln_lora_B_T_3D, timesteps_B_T,
        )


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast("cuda", dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T = inputs
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        return self.unpatchify(x_B_T_H_W_O)
