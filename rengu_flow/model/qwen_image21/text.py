"""Qwen-Image 2.1 text conditioning (text-to-image).

Reproduces ``QwenImage21Pipeline._get_qwen_prompt_embeds`` of the reference pipeline
(diffusers ``main`` 6256aa7, ``pipelines/qwenimage21/pipeline_qwenimage21.py``) for prompts
without condition images:

- the raw t2i chat template (not ``apply_chat_template``), empty prompts replaced by ``" "``;
- **left** padding for batches (the side the checkpoint was trained with);
- the conditioning is the input of the text decoder's final RMSNorm — captured with a forward
  hook that returns the norm's input (what ``hidden_states[-1]`` meant before transformers 5);
- the ``_drop_idx`` system-message tokens are dropped from each sample's valid tokens, and the
  batch is re-padded on the **right** (the transformer requires right padding).

The reference runs the full ``Qwen3VLForConditionalGeneration``; for text-only input that is
exactly its inner ``Qwen3VLTextModel`` (no position ids are passed on either path), which is all
rengu loads.
"""

from __future__ import annotations

import torch

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# What the processor's chat template renders for the system turn alone; the reference derives
# ``_drop_idx`` by tokenizing exactly this (14 tokens with the Qwen3-VL tokenizer).
SYSTEM_MESSAGE = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"


def drop_index(tokenizer) -> int:
    """Number of leading system-message tokens dropped from the encoder outputs."""
    return len(tokenizer(SYSTEM_MESSAGE).input_ids)


def text_model_of(text_encoder):
    """The ``Qwen3VLTextModel`` inside whatever wraps it (a ``LazyStreamedEncoder``, a
    ``Qwen3VLModel`` or a full ``Qwen3VLForConditionalGeneration``)."""
    module = text_encoder
    if hasattr(module, "load") and hasattr(module, "is_loaded"):  # LazyStreamedEncoder
        module = module.load()
    if hasattr(module, "model") and not hasattr(module, "layers"):  # ...ForConditionalGeneration
        module = module.model
    return getattr(module, "language_model", module)


@torch.no_grad()
def encode_prompts(
    text_encoder,
    tokenizer,
    prompts: list[str],
    device: torch.device | str,
    drop_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``prompts`` into right-padded ``(B, L, hidden)`` embeddings and a ``(B, L)`` bool
    mask of valid tokens (``L`` = longest prompt after dropping the system tokens)."""
    if drop_idx is None:
        drop_idx = drop_index(tokenizer)
    prompts = [" " if not p else p for p in prompts]
    texts = [PROMPT_TEMPLATE_T2I.format(p) for p in prompts]
    inputs = tokenizer(texts, padding=True, padding_side="left", return_tensors="pt").to(device)

    text_model = text_model_of(text_encoder)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = text_model(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False
        )
    finally:
        handle.remove()
    hidden_states = outputs.last_hidden_state

    valid = inputs.attention_mask.bool()
    split = [hidden_states[i][valid[i]][drop_idx:] for i in range(hidden_states.shape[0])]
    max_len = max(e.shape[0] for e in split)
    embeds = hidden_states.new_zeros((len(split), max_len, hidden_states.shape[-1]))
    mask = torch.zeros((len(split), max_len), dtype=torch.bool, device=hidden_states.device)
    for i, e in enumerate(split):
        embeds[i, : e.shape[0]] = e
        mask[i, : e.shape[0]] = True
    return embeds, mask
