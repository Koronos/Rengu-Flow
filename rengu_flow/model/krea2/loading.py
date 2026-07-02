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
  (ComfyUI's ``qwen3vl_4b_bf16.safetensors`` or an official-layout export). Single files
  load the text-only decoder (``Qwen3VLTextModel``) from the bundled config — the vision
  tower is never used for conditioning.
- **VAE**: a diffusers folder (``vae/``) or the single ``qwen_image_vae.safetensors``
  (same file cosmos uses), key-converted via diffusers' Wan converter into
  ``AutoencoderKLQwenImage`` with the bundled config.
- **Tokenizer**: bundled (``assets/qwen3vl_4b``); ``model.tokenizer_path`` overrides.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from rengu_flow.config.validation import ConfigValidationError

ASSETS_DIR = Path(__file__).parent / "assets"
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


def _looks_like_file(path: str | Path) -> bool:
    return Path(path).is_file()


def _require_exists(path: str | Path, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise ConfigValidationError(
            f"model.{what}: {p} does not exist. Point it at a local .safetensors file or "
            "folder you already downloaded — rengu never downloads models or resolves repo ids."
        )
    return p


def _guard_not_prequantized(state_dict: dict, what: str) -> None:
    fp8 = {torch.float8_e4m3fn, torch.float8_e5m2} & {v.dtype for v in state_dict.values()}
    scaled = any(k.endswith("scale_weight") for k in state_dict)
    if fp8 or scaled:
        raise ConfigValidationError(
            f"model.{what} points at a pre-quantized (fp8/nvfp4 'scaled') file, which cannot be "
            "trained. Use the bf16 file (e.g. krea2_raw_bf16.safetensors or raw.safetensors); "
            "for VRAM use model.transformer_4bit or model.transformer_fp8_matmul instead."
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

    path = _require_exists(path, "transformer_path")
    if not _looks_like_file(path):
        return Krea2Transformer2DModel.from_pretrained(path, torch_dtype=dtype)

    from safetensors.torch import load_file

    state_dict = load_file(path)
    _guard_not_prequantized(state_dict, "transformer_path")
    # Some re-exports wrap the original keys in a comfy checkpoint prefix.
    state_dict = {re.sub(r"^(model\.)?diffusion_model\.", "", k): v for k, v in state_dict.items()}
    if is_original_dit_state_dict(state_dict):
        state_dict = convert_dit_original_to_diffusers(state_dict)
    elif not any(k.startswith("transformer_blocks.") for k in state_dict):
        raise ConfigValidationError(
            "model.transformer_path: unrecognized Krea 2 checkpoint key layout (expected the "
            "official/ComfyUI original keys or diffusers keys)."
        )
    with torch.device("meta"):
        transformer = Krea2Transformer2DModel()  # single public config; no shape inference needed
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    return transformer


def load_vae(path: str | Path, dtype: torch.dtype):
    """Load the Qwen-Image VAE from a diffusers folder or the single Wan-layout file."""
    from diffusers import AutoencoderKLQwenImage

    path = _require_exists(path, "vae_path")
    if not _looks_like_file(path):
        vae = AutoencoderKLQwenImage.from_pretrained(path, torch_dtype=dtype)
    else:
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
        from safetensors.torch import load_file

        state_dict = load_file(path)
        if any(k.startswith("encoder.down_blocks.") for k in state_dict):
            converted = state_dict  # already diffusers-layout
        else:
            converted = convert_wan_vae_to_diffusers(state_dict)
        config = {
            k: v
            for k, v in json.loads(VAE_CONFIG_PATH.read_text()).items()
            if not k.startswith("_")
        }
        vae = AutoencoderKLQwenImage.from_config(config)
        vae.load_state_dict(converted)
        vae = vae.to(dtype)
    vae.eval().requires_grad_(False)
    return vae


def load_text_encoder(path: str | Path, dtype: torch.dtype):
    """Load the Qwen3-VL conditioner from a transformers folder or a single file.

    Single files (ComfyUI ``qwen3vl_4b_bf16.safetensors`` or official-layout exports) load
    the text-only decoder from the bundled config; vision weights in the file are ignored
    (conditioning never runs the vision tower)."""
    path = _require_exists(path, "text_encoder_path")
    if not _looks_like_file(path):
        from transformers import Qwen3VLModel

        model = Qwen3VLModel.from_pretrained(path, torch_dtype=dtype)
    else:
        from transformers import AutoConfig, Qwen3VLTextModel

        from safetensors.torch import load_file

        state_dict = load_file(path)
        # ComfyUI "scaled fp8" files store each quantized Linear as an fp8 `.weight` plus a
        # scalar `.weight_scale` and a `.comfy_quant` marker: dequantize to compute dtype.
        scales = {
            k[: -len(".weight_scale")]: v.float()
            for k, v in state_dict.items()
            if k.endswith(".weight_scale")
        }
        remapped = {}
        for k, v in state_dict.items():
            base = k[: -len(".weight")] if k.endswith(".weight") else None
            k = re.sub(r"^model\.", "", k)
            if k.startswith(("visual.", "lm_head.")) or k.endswith((".weight_scale", ".comfy_quant")):
                continue
            if k.startswith("language_model."):
                k = k[len("language_model.") :]
            if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                v = v.float() * scales.get(base, torch.tensor(1.0))
            remapped[k] = v.to(dtype)
        config = AutoConfig.from_pretrained(QWEN3VL_ASSETS).text_config
        from accelerate import init_empty_weights

        # include_buffers=False: params land on meta (replaced below by assign) but
        # non-persistent buffers (the rotary inv_freq, absent from checkpoints) are
        # computed for real at init.
        with init_empty_weights(include_buffers=False):
            model = Qwen3VLTextModel._from_config(config)
        model.load_state_dict(remapped, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model


def load_tokenizer(path: str | Path | None):
    from transformers import AutoTokenizer

    if path:
        return AutoTokenizer.from_pretrained(_require_exists(path, "tokenizer_path"))
    return AutoTokenizer.from_pretrained(QWEN3VL_ASSETS)
