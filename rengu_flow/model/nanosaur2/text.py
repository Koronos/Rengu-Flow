"""Nanosaur2 text conditioning: frozen Gemma3-270M, penultimate hidden state with the final norm
applied, and per-token emphasis weights.

Port of the official ComfyUI node's ``text_encoder.py`` (``well9472/Nanosaur2-670M``, MIT):

* tokens are ``[BOS(2)] + sentencepiece(prompt)`` (no EOS, no chat template), truncated to 256;
* ``(text:weight)`` groups are stripped from the prompt and become per-token weights (a token's
  weight is the product of the groups its characters overlap). The weights are not applied to the
  embedding: the DiT turns them into a log-bias on the text keys of its joint attention;
* the conditioning is ``norm(hidden_states[-2])`` of ``Gemma3TextModel`` (ComfyUI's
  ``layer_idx=-2`` with ``layer_norm_hidden_state``).

The tokenizer is the SentencePiece model serialized in the text-encoder checkpoint
(``spiece_model`` key), so no separate tokenizer files are needed.
"""

from __future__ import annotations

import re

import torch

MAX_LENGTH = 256
BOS_TOKEN = 2
PAD_TOKEN = 0


def parse_prompt_emphasis(caption: str) -> tuple[str, list[tuple[int, int, float]]]:
    """Strip "(text:weight)" groups; return the plain text and (start, end, weight) character spans
    into it. Verbatim port of the reference parser (unbalanced or non-numeric groups stay text)."""
    weight_pattern = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
    spans = []
    parts = []
    cursor = 0
    output_len = 0
    idx = 0
    while idx < len(caption):
        if caption[idx] != "(":
            idx += 1
            continue

        depth = 1
        end = idx + 1
        while end < len(caption) and depth > 0:
            if caption[end] == "(":
                depth += 1
            elif caption[end] == ")":
                depth -= 1
            end += 1

        if depth != 0:
            idx += 1
            continue

        inner = caption[idx + 1 : end - 1]
        inner_depth = 0
        colon_idx = -1
        for inner_idx, char in enumerate(inner):
            if char == "(":
                inner_depth += 1
            elif char == ")":
                inner_depth -= 1
            elif char == ":" and inner_depth == 0:
                colon_idx = inner_idx

        emphasized_text = inner[:colon_idx]
        weight_text = inner[colon_idx + 1 :].strip()
        if colon_idx == -1 or not emphasized_text or not weight_pattern.fullmatch(weight_text):
            idx += 1
            continue

        parts.append(caption[cursor:idx])
        output_len += idx - cursor
        parts.append(emphasized_text)
        spans.append((output_len, output_len + len(emphasized_text), float(weight_text)))
        output_len += len(emphasized_text)
        cursor = end
        idx = end

    parts.append(caption[cursor:])
    return "".join(parts), spans


def load_sentencepiece(model_proto: bytes | torch.Tensor):
    """A ``SentencePieceProcessor`` from the checkpoint's serialized ``spiece_model`` bytes."""
    try:
        import sentencepiece
    except ImportError as e:  # pragma: no cover - dependency declared in pyproject
        raise ImportError("nanosaur2 needs the `sentencepiece` package for its Gemma3 tokenizer.") from e
    if torch.is_tensor(model_proto):
        model_proto = model_proto.cpu().numpy().tobytes()
    return sentencepiece.SentencePieceProcessor(model_proto=model_proto, add_bos=False, add_eos=False)


def tokenize_with_weights(spm, text: str, max_length: int = MAX_LENGTH) -> list[tuple[int, float]]:
    """``[(token_id, weight), ...]``: BOS (weight 1) then the prompt tokens with their emphasis."""
    text, spans = parse_prompt_emphasis(text)
    if hasattr(spm, "EncodeAsOffsetMapping"):
        encoded = spm.encode(text, add_bos=False, return_type="offset_mapping")
        tokens = zip(encoded["ids"], encoded["offsets"])
    else:  # sentencepiece < 0.2.2
        encoded = spm.encode(text, add_bos=False, out_type="immutable_proto")
        tokens = [(piece.id, (piece.begin, piece.end)) for piece in encoded.pieces]
    batch = [(BOS_TOKEN, 1.0)]
    for token, (token_begin, token_end) in tokens:
        weight = 1.0
        for begin, end, span_weight in spans:
            if token_begin < end and token_end > begin:
                weight *= span_weight
        batch.append((int(token), weight))
    return batch[:max_length]


def gemma3_config_from_state_dict(state_dict: dict):
    """``Gemma3TextConfig`` of the checkpoint (Gemma3-270M geometry; widths read from the weights)."""
    from transformers import Gemma3TextConfig

    vocab, hidden = state_dict["embed_tokens.weight"].shape
    layers = 0
    while f"layers.{layers}.input_layernorm.weight" in state_dict:
        layers += 1
    head_dim = state_dict["layers.0.self_attn.q_norm.weight"].shape[0]
    pattern = 6  # 5 sliding-window layers, then 1 global (Gemma 3)
    config = Gemma3TextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=state_dict["layers.0.mlp.gate_proj.weight"].shape[0],
        num_hidden_layers=layers,
        num_attention_heads=state_dict["layers.0.self_attn.q_proj.weight"].shape[0] // head_dim,
        num_key_value_heads=state_dict["layers.0.self_attn.k_proj.weight"].shape[0] // head_dim,
        head_dim=head_dim,
        query_pre_attn_scalar=head_dim,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        sliding_window=512,
        layer_types=[
            "full_attention" if (i + 1) % pattern == 0 else "sliding_attention" for i in range(layers)
        ],
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        },
        pad_token_id=PAD_TOKEN,
        bos_token_id=BOS_TOKEN,
        eos_token_id=1,
        attn_implementation="sdpa",
    )
    return config


@torch.no_grad()
def encode_prompts(text_encoder, spm, prompts: list[str], device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per prompt: ``(L, 640)`` conditioning (float32) and ``(L,)`` token weights, at the prompt's
    own token length (BOS only for an empty prompt). Prompts are right-padded into one causal
    batch; padding sits after every real token, so it never reaches them."""
    rows = [tokenize_with_weights(spm, p) for p in prompts]
    lengths = [len(r) for r in rows]
    width = max(lengths)
    ids = torch.full((len(rows), width), PAD_TOKEN, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor([t for t, _ in r])
        mask[i, : len(r)] = 1
    out = text_encoder(
        input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True, use_cache=False
    )
    hidden = text_encoder.norm(out.hidden_states[-2]).float()
    embeds = [hidden[i, :n].cpu() for i, n in enumerate(lengths)]
    weights = [torch.tensor([w for _, w in r], dtype=torch.float32) for r in rows]
    return embeds, weights
