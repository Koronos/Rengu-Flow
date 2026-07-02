"""Krea 2 text conditioning: Qwen3-VL multi-layer hidden-state taps.

Krea 2 conditions on a *stack* of hidden states tapped from several decoder layers of
Qwen3-VL (not the last hidden state). Prompts use the Qwen-Image chat template tokenized
as a fixed-length block: ``[prefix | prompt | PAD | suffix]`` — the prompt is padded to a
fixed length first and the assistant suffix is appended *after* the padding, matching how
the model was sampled at training time. The first ``PREFIX_IDX`` (system prefix) tokens
are dropped from the encoder outputs.
"""

from __future__ import annotations

import torch

PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_IDX = 34
NUM_SUFFIX_TOKENS = 5

# Indices into the text encoder's ``hidden_states`` tuple (0 is the embedding output) whose
# states are stacked per token. These are the Krea 2 (Qwen3-VL-4B) taps; the pipeline reads
# the checkpoint's model_index.json override when present.
DEFAULT_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


@torch.no_grad()
def encode_prompts(
    text_encoder,
    tokenizer,
    prompts: list[str],
    select_layers: tuple[int, ...] = DEFAULT_SELECT_LAYERS,
    max_sequence_length: int = 512,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize ``prompts`` into the fixed-length Krea 2 layout and tap the selected hidden states.

    Returns ``(hidden_states, attention_mask)`` of shapes
    ``(B, max_sequence_length, len(select_layers), text_hidden_dim)`` and ``(B, L)`` (bool).
    """
    text = [PROMPT_PREFIX + p for p in prompts]
    text_tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length + PREFIX_IDX - NUM_SUFFIX_TOKENS,
        return_tensors="pt",
    ).to(device)
    suffix_tokens = tokenizer([PROMPT_SUFFIX] * len(text), return_tensors="pt").to(device)

    input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
    attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

    # Krea 2 pads in the middle of the template (``[prefix | prompt | PAD | suffix]``), so the
    # suffix tokens sit downstream of the padding. The text features must use positions that count
    # only real tokens (padding does not consume a position) to match how the model was trained;
    # Qwen3-VL's default raw-index positions would place the suffix at ~max_length instead. Build
    # the cumulative-valid-token positions and broadcast across the 3 mRoPE axes (T/H/W are equal
    # for text).
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    hidden_states = torch.stack([outputs.hidden_states[i] for i in select_layers], dim=2)

    return hidden_states[:, PREFIX_IDX:], attention_mask[:, PREFIX_IDX:]


def compact_text_embeddings(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop padded token lanes, keeping only each sample's valid tokens (left-compacted).

    The transformer gives text tokens zero RoPE indices and masks padded keys, so removing the
    padding lanes is mathematically equivalent — and the cached embedding shrinks from the fixed
    512-token block (~30 MB/caption at bf16) to the actual caption length. Samples are re-padded
    (zeros / False) to the longest valid length in the batch; ``prepare_inputs`` pads again at
    collate time when lengths differ across cached samples.
    """
    lengths = attention_mask.sum(dim=1)
    max_len = max(int(lengths.max().item()), 1)
    b, _, num_layers, dim = hidden_states.shape
    out = hidden_states.new_zeros((b, max_len, num_layers, dim))
    out_mask = attention_mask.new_zeros((b, max_len))
    for i in range(b):
        n = int(lengths[i].item())
        out[i, :n] = hidden_states[i][attention_mask[i]]
        out_mask[i, :n] = True
    return out, out_mask


def pad_text_embeddings(
    embeds: list[torch.Tensor] | torch.Tensor, masks: list[torch.Tensor] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-sample cached embeddings of varying token length into one padded batch."""
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
