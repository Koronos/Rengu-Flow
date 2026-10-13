"""Local-file loaders for the three Nanosaur2 release files (``well9472/Nanosaur2-670M``):

- ``nanosaur2_diffusion_model.safetensors`` — the DiT, loaded verbatim (official key names);
  the architecture is read from the weights (``dit.detect_config``);
- ``nanosaur2_text_encoder.safetensors`` — Gemma3-270M (``model.*`` keys, RMSNorm weights in
  float32) plus the SentencePiece tokenizer serialized under ``spiece_model``;
- ``nanosaur2_vae.safetensors`` — DINOv2 encoder + LDM decoder + ``latent_mean`` / ``latent_std``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model.dit_common.loading import (
    FP8_DTYPES,
    guard_not_prequantized,
    read_safetensors,
    require_exists,
    strip_comfy_prefix,
)

SPIECE_KEY = "spiece_model"


def _single_file(path, what: str) -> Path:
    p = require_exists(path, what)
    if not p.is_file():
        raise ConfigValidationError(f"model.{what}: expected the .safetensors file, got a folder ({p}).")
    return p


def _guard_not_prequantized(state_dict: dict, what: str) -> None:
    guard_not_prequantized(state_dict, what, kind="fp8", advice="; use the bf16 release file.")


def load_transformer(path, dtype: torch.dtype):
    from rengu_flow.model.nanosaur2.dit import Nanosaur2Transformer2DModel, detect_config

    path = _single_file(path, "transformer_path")
    state_dict = strip_comfy_prefix(read_safetensors(path, dtype))
    _guard_not_prequantized(state_dict, "transformer_path")
    if "s_embedder.proj.weight" not in state_dict or "encoder_adaLN_offsets.0" not in state_dict:
        raise ConfigValidationError(
            "model.transformer_path: not a Nanosaur2 diffusion model (expected keys like "
            "s_embedder.proj.weight, blocks.0.attn.qkv_x.weight, encoder_adaLN_offsets.0)."
        )
    with torch.device("meta"):
        transformer = Nanosaur2Transformer2DModel(**detect_config(state_dict))
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    return transformer


def load_vae(path, dtype: torch.dtype):
    from rengu_flow.model.nanosaur2.vae import Nanosaur2VAE, detect_vae_config

    path = _single_file(path, "vae_path")
    state_dict = read_safetensors(path, dtype)
    if "latent_mean" not in state_dict or "decoder.conv_in.weight" not in state_dict:
        raise ConfigValidationError("model.vae_path: not a Nanosaur2 VAE (expected latent_mean, decoder.*, encoder.*).")
    with torch.device("meta"):
        vae = Nanosaur2VAE(**detect_vae_config(state_dict))
    vae.load_state_dict(state_dict, strict=True, assign=True)
    return vae.eval().requires_grad_(False)


def read_text_encoder_file(path, dtype: torch.dtype | None = None) -> tuple[dict, torch.Tensor]:
    """``(gemma state dict without the "model." prefix, serialized sentencepiece bytes)``, the
    weights cast to ``dtype`` as they are read (``None`` = as stored). fp8 tensors keep their dtype
    (the pre-quantized guard sees them) and so does the uint8 tokenizer entry."""
    path = _single_file(path, "text_encoder_path")
    state_dict = read_safetensors(path, dtype, keep_dtypes=(*FP8_DTYPES, torch.uint8))
    if SPIECE_KEY not in state_dict:
        raise ConfigValidationError(
            f"model.text_encoder_path: {path.name} has no '{SPIECE_KEY}' tokenizer entry; use the "
            "nanosaur2_text_encoder.safetensors release file."
        )
    spiece = state_dict.pop(SPIECE_KEY)
    state_dict = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    state_dict.pop("lm_head.weight", None)
    return state_dict, spiece


def load_text_encoder(path, dtype: torch.dtype = torch.float32):
    """Gemma3-270M as ``transformers.Gemma3TextModel`` in ``dtype`` (float32 by default: the
    encoder is tiny and ComfyUI also computes it in float32 over the bf16 weights)."""
    from transformers import Gemma3TextModel

    from rengu_flow.model.nanosaur2.text import gemma3_config_from_state_dict

    state_dict, _spiece = read_text_encoder_file(path, dtype)
    _guard_not_prequantized(state_dict, "text_encoder_path")
    config = gemma3_config_from_state_dict(state_dict)
    with torch.device("meta"):
        model = Gemma3TextModel(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if unexpected or [k for k in missing if not k.endswith("embed_scale")]:
        raise ConfigValidationError(
            f"model.text_encoder_path: unexpected Gemma3 key layout (missing {missing[:4]}, "
            f"unexpected {unexpected[:4]})."
        )
    # Non-persistent buffers (rotary inv_freq, embed scale) were created on meta: rebuild them.
    _materialize_buffers(model, config)
    return model.eval().requires_grad_(False)


def _materialize_buffers(model, config) -> None:
    from transformers import Gemma3TextModel

    fresh = Gemma3TextModel(_tiny_like(config))
    fresh_buffers = dict(fresh.named_buffers())
    for name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        module_name, _, attr = name.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        source = fresh_buffers.get(name)
        if source is None:
            raise RuntimeError(f"cannot materialize Gemma3 buffer {name}")
        module._buffers[attr] = source.clone()


def _tiny_like(config):
    """Same rotary/scale settings with a 1-layer, tiny-vocab body: cheap source of the buffers."""
    import copy

    tiny = copy.deepcopy(config)
    tiny.vocab_size = 8
    tiny.intermediate_size = 8
    return tiny


def load_tokenizer(path):
    """The SentencePiece tokenizer serialized in the text-encoder file (only that entry is read)."""
    from safetensors import safe_open

    from rengu_flow.model.nanosaur2.text import load_sentencepiece

    path = _single_file(path, "text_encoder_path")
    with safe_open(str(path), framework="pt") as handle:
        if SPIECE_KEY not in handle.keys():
            raise ConfigValidationError(
                f"model.text_encoder_path: {path.name} has no '{SPIECE_KEY}' tokenizer entry; use the "
                "nanosaur2_text_encoder.safetensors release file."
            )
        return load_sentencepiece(handle.get_tensor(SPIECE_KEY))
