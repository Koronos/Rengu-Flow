"""Ragged text-embedding helpers shared by the DiT pipelines (krea2, qwen_image21, ...).

Cached text embeddings are stored per caption at their real token length. These helpers
compact an encoder batch to its valid tokens and re-pad cached rows of different lengths
into one batch at collate time. They are agnostic to the per-token feature shape: krea2
caches a ``(layers, dim)`` stack per token, qwen_image21 a single ``(dim,)`` vector.
"""

from __future__ import annotations

import torch


def compact_text_embeddings(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop padded token lanes, keeping only each sample's valid tokens (left-compacted).

    ``hidden_states`` is ``(B, L, *feature)``, ``attention_mask`` ``(B, L)`` bool. Samples are
    re-padded (zeros / False) on the right to the longest valid length in the batch.
    """
    lengths = attention_mask.sum(dim=1)
    max_len = max(int(lengths.max().item()), 1)
    b = hidden_states.shape[0]
    out = hidden_states.new_zeros((b, max_len, *hidden_states.shape[2:]))
    out_mask = attention_mask.new_zeros((b, max_len))
    for i in range(b):
        n = int(lengths[i].item())
        out[i, :n] = hidden_states[i][attention_mask[i]]
        out_mask[i, :n] = True
    return out, out_mask


def pad_text_embeddings(
    embeds: list[torch.Tensor] | torch.Tensor, masks: list[torch.Tensor] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-sample cached embeddings of varying token length into one right-padded batch."""
    if torch.is_tensor(embeds):
        return embeds, masks
    max_len = max(e.shape[0] for e in embeds)
    first = embeds[0]
    out = first.new_zeros((len(embeds), max_len, *first.shape[1:]))
    out_mask = torch.zeros((len(embeds), max_len), dtype=torch.bool)
    for i, (e, m) in enumerate(zip(embeds, masks)):
        out[i, : e.shape[0]] = e
        out_mask[i, : m.shape[0]] = m.bool()
    return out, out_mask
