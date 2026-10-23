"""MOSS-Image text conditioning: a line-for-line port of the reference ``encode_prompt``.

Each prompt is rendered with the Qwen3.5 chat template (``add_generation_prompt=True``,
``enable_thinking=True``, no system prompt), tokenized with truncation at 4096 tokens and forwarded
**on its own, unpadded** (the Gated DeltaNet layers are padding-sensitive); the conditioning is the
``last_hidden_state`` (after the final norm) of every token of the templated sequence. The
unconditional input is the empty prompt through the same template (the reference's default
negative prompt). Rows are right-padded only to batch them.
"""

from __future__ import annotations

import torch

MAX_SEQUENCE_LENGTH = 4096


def text_model_of(text_encoder):
    """The ``Qwen3_5TextModel`` inside whatever wraps it (a ``LazyStreamedEncoder`` or a full
    ``Qwen3_5Model``)."""
    module = text_encoder
    if hasattr(module, "load") and hasattr(module, "is_loaded"):  # LazyStreamedEncoder
        module = module.load()
    return getattr(module, "language_model", module)


def format_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


@torch.no_grad()
def encode_prompts(
    text_encoder, tokenizer, prompts: list[str], device: torch.device | str, max_length: int = MAX_SEQUENCE_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ``(B, L, hidden)`` embeddings and the ``(B, L)`` bool mask of valid tokens."""
    model = text_model_of(text_encoder)
    rows = []
    for prompt in prompts:
        inputs = tokenizer(
            format_prompt(tokenizer, prompt), return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        output = model(**inputs, use_cache=False)
        rows.append(output.last_hidden_state[0])
    length = max(row.shape[0] for row in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
    for index, row in enumerate(rows):
        embeds[index, : row.shape[0]] = row
        mask[index, : row.shape[0]] = True
    return embeds, mask
