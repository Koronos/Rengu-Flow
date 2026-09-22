"""Local-file loaders for the Qwen-Image 2.1 components.

Every component accepts what the user already has on disk — rengu never downloads models or
resolves repo ids:

- **DiT**: the diffusers ``transformer/`` folder (sharded) or a single ``.safetensors``. Single
  files may use the diffusers key layout or ComfyUI's (``qwen_image_2.1_bf16.safetensors``),
  which fuses each block's SwiGLU ``img_mlp.gate_layer`` + ``img_mlp.proj`` into one
  ``img_mlp.gate_up`` weight (gate rows first) — split back here. Pre-quantized files
  (``*_int8_convrot``, fp8 "scaled") cannot be trained and are refused.
- **Text encoder**: the transformers ``text_encoder/`` folder (Qwen3-VL-8B, sharded) or a
  single ``.safetensors`` (ComfyUI ``qwen3vl_8b_bf16.safetensors``). Only the text decoder
  (``Qwen3VLTextModel``) is loaded — t2i conditioning never runs the vision tower.
- **VAE**: the diffusers ``vae/`` folder or a single diffusers-layout ``.safetensors``
  (``AutoencoderKLQwenImage21``). ComfyUI's original-layout VAE file is not converted.
- **Tokenizer**: ``model.processor_path`` (the ``processor/`` folder), else
  ``<diffusers_path>/processor``, else the bundled Qwen3-VL tokenizer (byte-identical
  tokenization of the Qwen-Image 2.1 template).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from rengu_flow.config.validation import ConfigValidationError

ASSETS_DIR = Path(__file__).parent / "assets"
QWEN3VL_8B_ASSETS = ASSETS_DIR / "qwen3vl_8b"
VAE_CONFIG_PATH = ASSETS_DIR / "vae_config.json"
TRANSFORMER_CONFIG_PATH = ASSETS_DIR / "transformer_config.json"

_QUANTIZED_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.uint8}
_QUANT_KEY_SUFFIXES = ("scale_weight", "weight_scale", "comfy_quant", "input_scale")


def _require_exists(path: str | Path, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise ConfigValidationError(
            f"model.{what}: {p} does not exist. Point it at a local .safetensors file or "
            "folder you already downloaded — rengu never downloads models or resolves repo ids."
        )
    return p


def _config_kwargs(path: Path) -> dict:
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


def _guard_not_prequantized(state_dict: dict, what: str) -> None:
    quantized = _QUANTIZED_DTYPES & {v.dtype for v in state_dict.values()}
    marked = any(k.endswith(_QUANT_KEY_SUFFIXES) for k in state_dict)
    if quantized or marked:
        raise ConfigValidationError(
            f"model.{what} points at a pre-quantized (int8/fp8) file, which cannot be trained. "
            "Use the bf16 file (e.g. qwen_image_2.1_bf16.safetensors or the diffusers "
            "transformer/ folder); for VRAM use model.transformer_fp8_matmul or "
            "model.transformer_4bit instead."
        )


def split_fused_mlp(state_dict: dict) -> dict:
    """ComfyUI layout -> diffusers layout: ``img_mlp.gate_up`` (gate rows, then up rows) becomes
    ``img_mlp.gate_layer`` + ``img_mlp.proj`` (the halves ComfyUI's LoRA loader also addresses)."""
    out = {}
    for key, value in state_dict.items():
        if ".img_mlp.gate_up." in key:
            gate, up = value.chunk(2, dim=0)
            out[key.replace(".gate_up.", ".gate_layer.")] = gate
            out[key.replace(".gate_up.", ".proj.")] = up
        else:
            out[key] = value
    return out


def load_transformer(path: str | Path, dtype: torch.dtype):
    """Load the Qwen-Image 2.1 DiT from a diffusers folder or a single file."""
    from rengu_flow.model.qwen_image21.dit import QwenImage21Transformer2DModel

    path = _require_exists(path, "transformer_path")
    if not path.is_file():
        return QwenImage21Transformer2DModel.from_pretrained(path, torch_dtype=dtype)

    from safetensors.torch import load_file

    state_dict = load_file(path)
    _guard_not_prequantized(state_dict, "transformer_path")
    # Some re-exports wrap the keys in a comfy checkpoint prefix.
    state_dict = {re.sub(r"^(model\.)?diffusion_model\.", "", k): v for k, v in state_dict.items()}
    if not any(k.startswith("transformer_blocks.") for k in state_dict):
        raise ConfigValidationError(
            "model.transformer_path: unrecognized Qwen-Image 2.1 checkpoint key layout (expected "
            "diffusers keys or ComfyUI's qwen_image_2.1 keys, e.g. transformer_blocks.0.attn.to_q)."
        )
    state_dict = split_fused_mlp(state_dict)
    from accelerate import init_empty_weights

    # include_buffers=False: the rope/timestep freqs are not in the checkpoint and must stay real.
    with init_empty_weights(include_buffers=False):
        transformer = QwenImage21Transformer2DModel(**_config_kwargs(TRANSFORMER_CONFIG_PATH))
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    return transformer


def load_vae(path: str | Path, dtype: torch.dtype):
    """Load the Qwen-Image 2.1 VAE from a diffusers folder or a diffusers-layout single file."""
    from rengu_flow.model.qwen_image21.vae import AutoencoderKLQwenImage21

    path = _require_exists(path, "vae_path")
    if not path.is_file():
        vae = AutoencoderKLQwenImage21.from_pretrained(path, torch_dtype=dtype)
    else:
        from safetensors.torch import load_file

        state_dict = load_file(path)
        if not any(k.startswith("encoder.down_blocks.") for k in state_dict):
            raise ConfigValidationError(
                "model.vae_path: this looks like ComfyUI's original-layout Qwen-Image 2.1 VAE "
                "(qwen_image_2.1_vae_bf16.safetensors), which rengu does not convert. Point "
                "vae_path at the diffusers vae/ folder of Qwen/Qwen-Image-2.1 (or set "
                "model.diffusers_path to the whole download)."
            )
        vae = AutoencoderKLQwenImage21.from_config(_config_kwargs(VAE_CONFIG_PATH))
        vae.load_state_dict(state_dict)
        vae = vae.to(dtype)
    vae.eval().requires_grad_(False)
    return vae


def load_text_encoder(path: str | Path, dtype: torch.dtype):
    """Load the Qwen3-VL-8B text decoder (``Qwen3VLTextModel``) from a transformers folder or a
    single file, reading only the text-decoder tensors."""
    from transformers import AutoConfig

    from rengu_flow.model.dit_common.qwen3vl import checkpoint_files, load_qwen3vl_text_model

    path = _require_exists(path, "text_encoder_path")
    config_dir = path if (not path.is_file() and (path / "config.json").exists()) else QWEN3VL_8B_ASSETS
    text_config = AutoConfig.from_pretrained(config_dir).text_config
    files = checkpoint_files(path)
    if not files:
        raise ConfigValidationError(f"model.text_encoder_path: no .safetensors found in {path}.")
    if path.is_file():
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            # ComfyUI's int8_convrot / w4a8 files store integer weights (scaled fp8 files are
            # dequantized by the shared remap; integer schemes are not).
            integer = any(handle.get_slice(k).get_dtype() in ("I8", "U8") for k in handle.keys())
        if integer:
            raise ConfigValidationError(
                "model.text_encoder_path points at an int8/w4a8 quantized Qwen3-VL file; use "
                "qwen3vl_8b_bf16.safetensors or the diffusers text_encoder/ folder."
            )
    return load_qwen3vl_text_model(files, text_config, dtype)


def load_tokenizer(path: str | Path | None):
    from transformers import AutoTokenizer

    if path:
        return AutoTokenizer.from_pretrained(_require_exists(path, "processor_path"))
    from rengu_flow.model.krea2.loading import QWEN3VL_ASSETS  # same Qwen3-VL tokenizer

    return AutoTokenizer.from_pretrained(QWEN3VL_ASSETS)
