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
  (``Qwen3VLTextModel``) is loaded for text-to-image; the vision tower (``Qwen3VLVisionModel``)
  is read separately, and only when a caption comes with condition images (edit training). Both
  sources carry it (``model.visual.*``).
- **VAE**: the diffusers ``vae/`` folder or a single ``.safetensors`` (``AutoencoderKLQwenImage21``)
  in the diffusers layout or the original (Wan-style) layout of ComfyUI's
  ``qwen_image_2.1_vae_bf16.safetensors``, which :func:`convert_original_vae_state_dict` renames
  (``downsamples.N.downsamples.M.residual.K`` -> ``down_blocks.N.resnets.M.<norm|conv>``, ...) and
  whose ``[out, in, 1, kh, kw]`` conv kernels are squeezed to the 2D kernels of the image-only VAE.
- **Tokenizer**: ``model.processor_path`` (the ``processor/`` folder), else
  ``<diffusers_path>/processor``, else the bundled Qwen3-VL tokenizer (byte-identical
  tokenization of the Qwen-Image 2.1 template).
- **Processor** (edit only: the Qwen3-VL image processor next to that tokenizer): the image
  processor config of the same ``processor/`` folder, else the bundled copy of the release's
  ``processor/preprocessor_config.json``.
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


# Original (Wan-style) VAE key layout -> diffusers AutoencoderKLQwenImage21, applied in order
# (the pattern of diffusers' convert_wan_vae_to_diffusers, as rules instead of per-key tables).
_ORIGINAL_VAE_KEY_RULES = [
    (r"^conv1\.", "quant_conv."),
    (r"^conv2\.", "post_quant_conv."),
    (r"^(encoder|decoder)\.conv1\.", r"\1.conv_in."),
    (r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out."),
    (r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out."),
    (r"^(encoder|decoder)\.middle\.0\.", r"\1.mid_block.resnets.0."),
    (r"^(encoder|decoder)\.middle\.1\.", r"\1.mid_block.attentions.0."),
    (r"^(encoder|decoder)\.middle\.2\.", r"\1.mid_block.resnets.1."),
    # The last entry of each level is its resampler (resample.1 / time_conv), the rest resnets.
    (r"^encoder\.downsamples\.(\d+)\.downsamples\.\d+\.(resample|time_conv)\.", r"encoder.down_blocks.\1.downsampler.\2."),
    (r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2."),
    (r"^decoder\.upsamples\.(\d+)\.upsamples\.\d+\.(resample|time_conv)\.", r"decoder.up_blocks.\1.upsampler.\2."),
    (r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.", r"decoder.up_blocks.\1.resnets.\2."),
    (r"\.residual\.0\.", ".norm1."),
    (r"\.residual\.2\.", ".conv1."),
    (r"\.residual\.3\.", ".norm2."),
    (r"\.residual\.6\.", ".conv2."),
    (r"\.shortcut\.", ".conv_shortcut."),
]


def is_original_vae_layout(state_dict: dict) -> bool:
    return "encoder.conv1.weight" in state_dict and not any(k.startswith("encoder.down_blocks.") for k in state_dict)


def convert_original_vae_state_dict(state_dict: dict) -> dict:
    """ComfyUI / original-layout Qwen-Image 2.1 VAE -> diffusers ``AutoencoderKLQwenImage21`` keys.

    The original checkpoint is a causal-3D (Wan-style) VAE whose conv kernels are all one frame
    deep (``[out, in, 1, kh, kw]``); the vendored image-only VAE uses ``Conv2d`` kernels, so the
    singleton time axis is squeezed. Values are otherwise untouched (same dtype, same data)."""
    out = {}
    for key, value in state_dict.items():
        new = key
        for pattern, repl in _ORIGINAL_VAE_KEY_RULES:
            new = re.sub(pattern, repl, new)
        if value.ndim == 5:
            if value.shape[2] != 1:
                raise ConfigValidationError(
                    f"model.vae_path: {key} has a {value.shape[2]}-frame conv kernel; the "
                    "Qwen-Image 2.1 VAE uses single-frame kernels. Is this another model's VAE?"
                )
            value = value.squeeze(2)
        out[new] = value
    return out


def load_vae(path: str | Path, dtype: torch.dtype):
    """Load the Qwen-Image 2.1 VAE from a diffusers folder or a single file (diffusers layout or
    ComfyUI's original layout, converted here)."""
    from rengu_flow.model.qwen_image21.vae import AutoencoderKLQwenImage21

    path = _require_exists(path, "vae_path")
    if not path.is_file():
        vae = AutoencoderKLQwenImage21.from_pretrained(path, torch_dtype=dtype)
    else:
        from safetensors.torch import load_file

        state_dict = load_file(path)
        if is_original_vae_layout(state_dict):
            state_dict = convert_original_vae_state_dict(state_dict)
        vae = AutoencoderKLQwenImage21.from_config(_config_kwargs(VAE_CONFIG_PATH))
        expected = set(vae.state_dict())
        missing, unexpected = expected - set(state_dict), set(state_dict) - expected
        if missing or unexpected:
            raise ConfigValidationError(
                "model.vae_path: not a Qwen-Image 2.1 VAE (expected ComfyUI's "
                "qwen_image_2.1_vae_bf16.safetensors, a diffusers-layout file or the diffusers "
                f"vae/ folder); {len(missing)} missing keys (e.g. {sorted(missing)[:3]}), "
                f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})."
            )
        vae.load_state_dict(state_dict, strict=True)
        vae = vae.to(dtype)
    vae.eval().requires_grad_(False)
    return vae


def load_qwen3vl_config(path: str | Path):
    """The full ``Qwen3VLConfig`` of the text encoder: the folder's ``config.json`` when it has
    one, else the bundled Qwen3-VL-8B config (single files carry none)."""
    from transformers import AutoConfig

    path = Path(path)
    config_dir = path if (not path.is_file() and (path / "config.json").exists()) else QWEN3VL_8B_ASSETS
    return AutoConfig.from_pretrained(config_dir)


def load_text_encoder(path: str | Path, dtype: torch.dtype):
    """Load the Qwen3-VL-8B text decoder (``Qwen3VLTextModel``) from a transformers folder or a
    single file, reading only the text-decoder tensors."""
    from rengu_flow.model.dit_common.qwen3vl import checkpoint_files, load_qwen3vl_text_model

    path = _require_exists(path, "text_encoder_path")
    text_config = load_qwen3vl_config(path).text_config
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


def load_vision_encoder(path: str | Path, dtype: torch.dtype):
    """Load the Qwen3-VL-8B vision tower (``Qwen3VLVisionModel`` + deepstack mergers) from the
    same checkpoint as the text decoder, reading only its ``visual.*`` tensors (~0.6B params)."""
    from rengu_flow.model.dit_common.qwen3vl import load_qwen3vl_vision_model, vision_checkpoint_files

    path = _require_exists(path, "text_encoder_path")
    try:
        return load_qwen3vl_vision_model(
            vision_checkpoint_files(path), load_qwen3vl_config(path).vision_config, dtype
        )
    except ValueError as e:
        raise ConfigValidationError(
            f"model.text_encoder_path: {e} Edit training (control_path / preview control_images) "
            "needs it: use ComfyUI's qwen3vl_8b_bf16.safetensors (Comfy-Org/Qwen-Image-2.1, "
            "text_encoders/) or the Qwen/Qwen-Image-2.1 text_encoder/ folder — both include it."
        ) from e


def load_processor(path: str | Path | None, tokenizer):
    """The Qwen3-VL processor used for image-conditioned prompts: ``tokenizer`` plus the image
    processor configured by ``path/preprocessor_config.json`` (the ``processor/`` folder), or the
    bundled copy of the release's config when ``path`` is ``None``."""
    from transformers import AutoImageProcessor, Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    source = _require_exists(path, "processor_path") if path else QWEN3VL_8B_ASSETS
    return Qwen3VLProcessor(
        image_processor=AutoImageProcessor.from_pretrained(source),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),  # required by the processor; never used
        chat_template=getattr(tokenizer, "chat_template", None),
    )


def load_tokenizer(path: str | Path | None):
    from transformers import AutoTokenizer

    if path:
        return AutoTokenizer.from_pretrained(_require_exists(path, "processor_path"))
    from rengu_flow.model.krea2.loading import QWEN3VL_ASSETS  # same Qwen3-VL tokenizer

    return AutoTokenizer.from_pretrained(QWEN3VL_ASSETS)
