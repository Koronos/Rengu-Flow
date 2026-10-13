"""Canonical SDXL text-encoder names for adapter files (independent of transformers' version).

transformers 4 ``CLIPTextModel`` keeps its layers under a ``text_model`` submodule; transformers 5
made it flat (``CLIPTextModelWithProjection`` still has ``text_model``). ComfyUI / A1111 / kohya
only recognise the canonical ``text_model`` form (``lora_te1_text_model_encoder_layers_N_...``), so
adapter exports always use it, and loads accept the canonical form and the flat one that
transformers-5 exports briefly wrote, translating to whatever the live module is named.

Two namings exist: kohya-flat (``lora_te1_<path with _>``) and dotted (``text_encoder.<path>``,
LoKr). ``text_projection`` lives outside ``text_model`` and is never rewritten.
"""

from __future__ import annotations

_TE_ROOTS = (("lora_te1_", "text_encoder"), ("lora_te2_", "text_encoder_2"))
_OUTSIDE = ("text_projection",)


def _needs_prefix(remainder: str) -> bool:
    return not remainder.startswith(("text_model_", "text_model.") + _OUTSIDE)


def canonical_flat(state: dict) -> dict:
    """Kohya-flat keys -> canonical (``lora_teN_text_model_...``). Idempotent."""
    out = {}
    for key, value in state.items():
        for prefix, _ in _TE_ROOTS:
            if key.startswith(prefix) and _needs_prefix(key[len(prefix):]):
                key = prefix + "text_model_" + key[len(prefix):]
                break
        out[key] = value
    return out


def native_flat(state: dict, text_encoder, text_encoder_2) -> dict:
    """Kohya-flat keys (canonical or flat) -> the live modules' own naming."""
    out = {}
    for key, value in canonical_flat(state).items():
        for (prefix, _), module in zip(_TE_ROOTS, (text_encoder, text_encoder_2)):
            if key.startswith(prefix + "text_model_") and not hasattr(module, "text_model"):
                key = prefix + key[len(prefix + "text_model_"):]
                break
        out[key] = value
    return out


def canonical_dotted(state: dict) -> dict:
    """Dotted keys (``text_encoder.<path>`` / ``text_encoder_2.<path>``) -> canonical. Idempotent."""
    out = {}
    for key, value in state.items():
        for _, root in _TE_ROOTS:
            head = root + "."
            if key.startswith(head) and _needs_prefix(key[len(head):]):
                key = head + "text_model." + key[len(head):]
                break
        out[key] = value
    return out


def native_dotted_for_module(state: dict, module) -> dict:
    """Module-relative dotted keys (root prefix already stripped), canonical or flat -> the
    module's own naming."""
    has = hasattr(module, "text_model")
    out = {}
    for key, value in state.items():
        bare = key[len("text_model."):] if key.startswith("text_model.") else key
        if has and not bare.startswith(_OUTSIDE):
            bare = "text_model." + bare
        out[bare] = value
    return out
