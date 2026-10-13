"""Local-file loaders for the Krea 2 components.

Every component accepts what the user already has on disk — rengu never downloads
models or touches the Hub:

- **DiT**: a diffusers-layout folder (``transformer/`` of the HF release) or a single
  ``.safetensors`` in the original Krea key layout — both the official ``raw.safetensors``
  and ComfyUI's ``krea2_raw_bf16.safetensors`` use it (``blocks.N.attn.wq``, ``mod.lin``,
  ``txtfusion`` …). Single files are key-converted to the vendored diffusers naming; the
  architecture is fixed (one public config, ``single_mmdit_large_wide``), so no shape
  inference is needed.
- **Text encoder**: a transformers folder (``text_encoder/``) or a single ``.safetensors``
  (ComfyUI's ``qwen3vl_4b_bf16.safetensors`` or an official-layout export). Both load only
  the text decoder (``Qwen3VLTextModel``; single files use the bundled config) — the vision
  tower is never used for conditioning, so its weights are never read.
- **VAE**: a diffusers folder (``vae/``) or the single ``qwen_image_vae.safetensors``
  (same file cosmos uses), key-converted via diffusers' Wan converter into
  ``AutoencoderKLQwenImage`` with the bundled config.
- **Tokenizer**: the bundled Qwen3-VL one (``dit_common/assets/qwen3vl_tokenizer``, shared with
  qwen_image21); ``model.tokenizer_path`` overrides.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model.dit_common.loading import (
    config_dir_or_bundled,
    guard_not_prequantized,
    load_file_or_folder,
    read_safetensors,
    require_exists,
    strip_comfy_prefix,
)

ASSETS_DIR = Path(__file__).parent / "assets"
# Bundled Qwen3-VL-4B model config (single-file text encoders carry none).
QWEN3VL_ASSETS = ASSETS_DIR / "qwen3vl_4b"
VAE_CONFIG_PATH = ASSETS_DIR / "qwen_image_vae_config.json"

# Inner renames shared by the DiT transformer blocks and the text-fusion blocks
# (original Krea naming -> vendored diffusers naming). Order matters: attn.gate must
# rename before mlp.gate cannot collide because prefixes differ.
_BLOCK_RENAMES = (
    (".attn.wq.", ".attn.to_q."),
    (".attn.wk.", ".attn.to_k."),
    (".attn.wv.", ".attn.to_v."),
    (".attn.wo.", ".attn.to_out.0."),
    (".attn.gate.", ".attn.to_gate."),
    (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
    (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
    (".prenorm.scale", ".norm1.weight"),
    (".postnorm.scale", ".norm2.weight"),
    (".mlp.gate.", ".ff.gate."),
    (".mlp.up.", ".ff.up."),
    (".mlp.down.", ".ff.down."),
)

_TOP_RENAMES = (
    ("first.", "img_in."),
    ("tmlp.0.", "time_embed.linear_1."),
    ("tmlp.2.", "time_embed.linear_2."),
    ("tproj.1.", "time_mod_proj."),
    ("txtmlp.0.scale", "txt_in.norm.weight"),
    ("txtmlp.1.", "txt_in.linear_1."),
    ("txtmlp.3.", "txt_in.linear_2."),
    ("txtfusion.", "text_fusion."),
    ("last.norm.scale", "final_layer.norm.weight"),
    ("last.linear.", "final_layer.linear."),
)


def _guard_not_prequantized(state_dict: dict, what: str) -> None:
    guard_not_prequantized(
        state_dict,
        what,
        kind="fp8/nvfp4 'scaled'",
        advice=(
            ". Use the bf16 file (e.g. krea2_raw_bf16.safetensors or raw.safetensors); for VRAM use "
            "model.transformer_4bit or model.transformer_fp8_matmul instead."
        ),
    )


def is_original_dit_state_dict(state_dict: dict) -> bool:
    return any(k.startswith("blocks.") for k in state_dict) and "first.weight" in state_dict


def convert_dit_original_to_diffusers(state_dict: dict) -> dict:
    """Rename an original-layout Krea 2 DiT state dict (official ``raw.safetensors`` /
    ComfyUI files) to the vendored diffusers naming. Reference for the original names:
    ComfyUI ``comfy/ldm/krea2/model.py`` (which runs them verbatim)."""
    out = {}
    for key, value in state_dict.items():
        new = key
        if new.startswith("blocks."):
            new = "transformer_blocks." + new[len("blocks.") :]
            if new.endswith(".mod.lin"):
                out[new.replace(".mod.lin", ".scale_shift_table")] = value.reshape(6, -1)
                continue
            for old, repl in _BLOCK_RENAMES:
                new = new.replace(old, repl)
        elif key == "last.modulation.lin":
            out["final_layer.scale_shift_table"] = value.reshape(2, -1)
            continue
        else:
            for old, repl in _TOP_RENAMES:
                if new.startswith(old):
                    new = repl + new[len(old) :]
                    break
            if new.startswith("text_fusion."):
                for old, repl in _BLOCK_RENAMES:
                    new = new.replace(old, repl)
        out[new] = value
    return out


def load_transformer(path: str | Path, dtype: torch.dtype):
    """Load the Krea 2 DiT from a diffusers folder or a single original-layout file."""
    from rengu_flow.model.krea2.dit import Krea2Transformer2DModel

    return load_file_or_folder(
        path,
        "transformer_path",
        folder=lambda p: Krea2Transformer2DModel.from_pretrained(p, torch_dtype=dtype),
        file=lambda p: _transformer_from_file(p, dtype),
    )


def _transformer_from_file(path: Path, dtype: torch.dtype):
    from rengu_flow.model.krea2.dit import Krea2Transformer2DModel

    # Tensor by tensor, cast as read; fp8 tensors keep their dtype for the guard.
    state_dict = read_safetensors(path, dtype)
    _guard_not_prequantized(state_dict, "transformer_path")
    # Some re-exports wrap the original keys in a comfy checkpoint prefix.
    state_dict = strip_comfy_prefix(state_dict)
    if is_original_dit_state_dict(state_dict):
        state_dict = convert_dit_original_to_diffusers(state_dict)
    elif not any(k.startswith("transformer_blocks.") for k in state_dict):
        raise ConfigValidationError(
            "model.transformer_path: unrecognized Krea 2 checkpoint key layout (expected the "
            "official/ComfyUI original keys or diffusers keys)."
        )
    with torch.device("meta"):
        transformer = Krea2Transformer2DModel()  # single public config; no shape inference needed
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    return transformer


def load_vae(path: str | Path, dtype: torch.dtype):
    """Load the Qwen-Image VAE from a diffusers folder or the single Wan-layout file."""
    from diffusers import AutoencoderKLQwenImage

    vae = load_file_or_folder(
        path,
        "vae_path",
        folder=lambda p: AutoencoderKLQwenImage.from_pretrained(p, torch_dtype=dtype),
        file=lambda p: _vae_from_file(p, dtype),
    )
    vae.eval().requires_grad_(False)
    return vae


def _vae_from_file(path: Path, dtype: torch.dtype):
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
    from safetensors.torch import load_file

    state_dict = load_file(path)
    if any(k.startswith("encoder.down_blocks.") for k in state_dict):
        converted = state_dict  # already diffusers-layout
    else:
        converted = convert_wan_vae_to_diffusers(state_dict)
    config = {k: v for k, v in json.loads(VAE_CONFIG_PATH.read_text()).items() if not k.startswith("_")}
    vae = AutoencoderKLQwenImage.from_config(config)
    vae.load_state_dict(converted)
    return vae.to(dtype)


def load_text_encoder(path: str | Path, dtype: torch.dtype):
    """Load the Qwen3-VL text decoder (``Qwen3VLTextModel``) from a transformers folder or a
    single file (ComfyUI ``qwen3vl_4b_bf16.safetensors`` or official-layout exports; the
    bundled config). Vision weights are never read (conditioning never runs the vision tower)."""
    from transformers import AutoConfig

    from rengu_flow.model.dit_common.qwen3vl import checkpoint_files, load_qwen3vl_text_model

    path = require_exists(path, "text_encoder_path")
    config_dir = config_dir_or_bundled(path, QWEN3VL_ASSETS)
    files = checkpoint_files(path)
    if not files:
        raise ConfigValidationError(f"model.text_encoder_path: no .safetensors found in {path}.")
    # Text-decoder tensors only, read shard by shard (a folder's vision tower is never
    # loaded); ComfyUI "scaled fp8" entries are dequantized (shared helper).
    return load_qwen3vl_text_model(files, AutoConfig.from_pretrained(config_dir).text_config, dtype)


def load_tokenizer(path: str | Path | None):
    from rengu_flow.model.dit_common.qwen3vl import load_qwen3vl_tokenizer

    return load_qwen3vl_tokenizer(path, "tokenizer_path")
