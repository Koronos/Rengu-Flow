"""Local-file loaders for the MOSS-Image components.

rengu never downloads models: every component accepts what the user already has on disk.

- **DiT**: the official diffusers ``transformer/`` folder (sharded safetensors) or a single
  ``.safetensors``. Comfy-Org's ``moss_image_{base,sft}_bf16.safetensors`` fuses each block's
  ``proj.q/k/v`` into ``proj.qkv`` (rows ``q; k; v``): split back into the official names on load.
  Every dimension is detected from the weights. Pre-quantized files (``*_int8_convrot``) cannot be
  trained and are refused.
- **VAE**: the diffusers ``vae/`` folder, Comfy-Org's ``flux2-vae_small.safetensors`` (diffusers
  layout) or ``flux2-vae.safetensors`` (original LDM layout, converted here). The decoder width is
  detected: the release's ``vae/`` and ``flux2-vae_small`` carry FLUX.2's *small* decoder
  (``decoder_block_out_channels``), which the installed diffusers' ``AutoencoderKLFlux2`` cannot
  express, so the decoder is swapped in after construction.
- **Text encoder**: the transformers ``text_encoder/`` folder (``Qwen3_5Model``, sharded) or
  Comfy-Org's ``qwen3.5_9b_bf16.safetensors``. Only the text decoder (``Qwen3_5TextModel``, what
  ``last_hidden_state`` of the reference comes from) is loaded: the vision tower, LM head and MTP
  head are never read.
- **Tokenizer**: ``model.tokenizer_path``, else ``<diffusers_path>/tokenizer``, else the Qwen3.5
  tokenizer + chat template bundled in ``assets/tokenizer`` (copied from the Base repo).
"""

from __future__ import annotations

import json
import re
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
from rengu_flow.model.dit_common.qwen3vl import checkpoint_files

ASSETS_DIR = Path(__file__).parent / "assets"
TOKENIZER_ASSETS = ASSETS_DIR / "tokenizer"
TEXT_ENCODER_ASSETS = ASSETS_DIR / "text_encoder"
VAE_CONFIG_PATH = ASSETS_DIR / "vae_config.json"

_QUANTIZED_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.uint8}
_QUANT_KEY_SUFFIXES = (
    "weight_scale",
    "scale_weight",
    "comfy_quant",
    "weight_codebook",
    "weight_s_channel",
    "comfy_attention.config",
)
_QUANTIZED_DTYPE_NAMES = {"I8", "U8", "F8_E4M3", "F8_E5M2"}

# FLUX.2's small decoder (the release's vae/ and Comfy-Org's flux2-vae_small).
SMALL_DECODER_CHANNELS = (96, 192, 384, 384)


def _config_kwargs(path: Path) -> dict:
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


# ---- DiT ------------------------------------------------------------------------------------


def split_fused_qkv(state_dict: dict) -> dict:
    """ComfyUI layout -> official layout: ``layers.N.proj.qkv.weight`` (rows ``q; k; v``) becomes
    ``proj.q`` / ``proj.k`` / ``proj.v`` (done tensor by tensor, so the peak is one extra layer)."""
    for key in [k for k in state_dict if k.endswith(".proj.qkv.weight")]:
        fused = state_dict.pop(key)
        if fused.shape[0] % 3:
            raise ConfigValidationError(f"model.transformer_path: {key} has {fused.shape[0]} rows, not 3 x dim.")
        prefix = key[: -len("qkv.weight")]
        for name, part in zip("qkv", fused.chunk(3, dim=0)):
            state_dict[f"{prefix}{name}.weight"] = part.clone()
    return state_dict


def _guard_not_prequantized(state_dict: dict, what: str) -> None:
    guard_not_prequantized(
        state_dict,
        what,
        kind="int8 / w4a8",
        advice=(
            ". Use the bf16 file (moss_image_base_bf16.safetensors or the diffusers transformer/ "
            "folder); for VRAM use model.transformer_fp8_matmul or model.transformer_4bit instead."
        ),
        dtypes=_QUANTIZED_DTYPES,
        key_suffixes=_QUANT_KEY_SUFFIXES,
    )


def _read_transformer_state(path: Path, dtype: torch.dtype) -> dict:
    state: dict = {}
    for file in checkpoint_files(path):
        state.update(read_safetensors(file, dtype, keep_dtypes=_QUANTIZED_DTYPES))
    if not state:
        raise ConfigValidationError(f"model.transformer_path: no .safetensors found in {path}.")
    return state


def load_transformer(path: str | Path, dtype: torch.dtype):
    """Load the MOSS-Image DiT from the official diffusers folder or a single file (official or
    Comfy-Org fused-qkv layout)."""
    return load_file_or_folder(
        path,
        "transformer_path",
        folder=lambda p: _transformer_from_state(_read_transformer_state(p, dtype), p),
        file=lambda p: _transformer_from_state(_read_transformer_state(p, dtype), None),
    )


def _transformer_from_state(state_dict: dict, folder: Path | None):
    from accelerate import init_empty_weights

    from rengu_flow.model.moss_image.dit import MOSSImageTransformer, detect_config

    _guard_not_prequantized(state_dict, "transformer_path")
    state_dict = split_fused_qkv(strip_comfy_prefix(state_dict))
    if "layers.0.proj.q.weight" not in state_dict or "shared_block_modulation.weight" not in state_dict:
        raise ConfigValidationError(
            "model.transformer_path: unrecognized MOSS-Image checkpoint key layout (expected the official "
            "keys, e.g. layers.0.proj.q.weight, or Comfy-Org's moss_image_*_bf16 with layers.0.proj.qkv.weight)."
        )
    rope_base = 10000.0
    if folder is not None and (folder / "config.json").is_file():
        rope_base = float(json.loads((folder / "config.json").read_text()).get("rope_base", rope_base))
    config = detect_config(state_dict, rope_base)
    state_dict.pop("rope.bands", None)  # derived constant (fp32), rebuilt by the model
    # include_buffers=False: rope.bands is derived, not read from the file, and must stay real.
    with init_empty_weights(include_buffers=False):
        transformer = MOSSImageTransformer(**config)
    expected = {k for k in transformer.state_dict() if k != "rope.bands"}
    missing, unexpected = expected - set(state_dict), set(state_dict) - expected
    if missing or unexpected:
        raise ConfigValidationError(
            f"model.transformer_path: not a MOSS-Image DiT of the detected size {config}; "
            f"{len(missing)} missing keys (e.g. {sorted(missing)[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})."
        )
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    return transformer


# ---- VAE --------------------------------------------------------------------------------------

_LDM_MID = {"q": "to_q", "k": "to_k", "v": "to_v", "proj_out": "to_out.0", "norm": "group_norm"}


def is_ldm_vae_layout(state_dict: dict) -> bool:
    """Original (LDM) layout of Comfy-Org's ``flux2-vae.safetensors``: ``encoder.down.N.block.M``."""
    return any(k.startswith("encoder.down.") for k in state_dict)


def convert_ldm_vae_state_dict(state_dict: dict) -> dict:
    """Original LDM VAE keys -> diffusers ``AutoencoderKLFlux2`` keys (values untouched, except the
    mid-block attention's 1x1 convolutions, which become the Linear weights of diffusers' attention)."""
    levels = 1 + max((int(m.group(1)) for k in state_dict if (m := re.match(r"decoder\.up\.(\d+)\.", k))), default=0)
    out = {}
    for key, value in state_dict.items():
        new = key
        new = re.sub(r"^encoder\.quant_conv\.", "quant_conv.", new)
        new = re.sub(r"^decoder\.post_quant_conv\.", "post_quant_conv.", new)
        new = re.sub(r"^(encoder|decoder)\.norm_out\.", r"\1.conv_norm_out.", new)
        new = re.sub(r"^(encoder|decoder)\.mid\.block_(\d)\.", lambda m: f"{m.group(1)}.mid_block.resnets.{int(m.group(2)) - 1}.", new)
        mid = re.match(r"^(encoder|decoder)\.mid\.attn_1\.(q|k|v|proj_out|norm)\.", new)
        if mid:
            new = f"{mid.group(1)}.mid_block.attentions.0.{_LDM_MID[mid.group(2)]}." + new[mid.end() :]
            if value.ndim == 4 and mid.group(2) != "norm":
                value = value.squeeze(-1).squeeze(-1)
        new = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", new)
        new = re.sub(r"^encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0.", new)
        new = re.sub(
            r"^decoder\.up\.(\d+)\.block\.(\d+)\.",
            lambda m: f"decoder.up_blocks.{levels - 1 - int(m.group(1))}.resnets.{m.group(2)}.",
            new,
        )
        new = re.sub(
            r"^decoder\.up\.(\d+)\.upsample\.",
            lambda m: f"decoder.up_blocks.{levels - 1 - int(m.group(1))}.upsamplers.0.",
            new,
        )
        new = new.replace(".nin_shortcut.", ".conv_shortcut.")
        out[new] = value
    return out


def _vae_from_state(state_dict: dict, vae_cfg: dict, dtype: torch.dtype, what: str):
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLFlux2
    from diffusers.models.autoencoders.vae import Decoder

    vae_cfg = dict(vae_cfg)
    decoder_channels = vae_cfg.pop("decoder_block_out_channels", None)
    with init_empty_weights(include_buffers=False):
        vae = AutoencoderKLFlux2.from_config(vae_cfg)
        if decoder_channels is not None and tuple(decoder_channels) != tuple(vae_cfg["block_out_channels"]):
            vae.decoder = Decoder(
                in_channels=vae_cfg["latent_channels"],
                out_channels=vae_cfg["out_channels"],
                up_block_types=vae_cfg["up_block_types"],
                block_out_channels=tuple(decoder_channels),
                layers_per_block=vae_cfg["layers_per_block"],
                norm_num_groups=vae_cfg["norm_num_groups"],
                act_fn=vae_cfg["act_fn"],
                mid_block_add_attention=vae_cfg["mid_block_add_attention"],
            )
    expected = set(vae.state_dict())
    state_dict = dict(state_dict)
    state_dict.setdefault("bn.num_batches_tracked", torch.zeros((), dtype=torch.long))
    missing, unexpected = expected - set(state_dict), set(state_dict) - expected
    if missing or unexpected:
        raise ConfigValidationError(
            f"model.{what}: not a FLUX.2 VAE as MOSS-Image uses (expected the diffusers vae/ folder or Comfy-Org's "
            f"flux2-vae*.safetensors); {len(missing)} missing keys (e.g. {sorted(missing)[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})."
        )
    vae.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in state_dict.items()}, assign=True)
    return vae.eval().requires_grad_(False)


def _read_vae_state(files: list[Path]) -> dict:
    from safetensors.torch import load_file

    state: dict = {}
    for file in files:
        state.update(load_file(str(file)))
    return state


def _vae_from_folder(path: Path, dtype: torch.dtype):
    vae_cfg = _config_kwargs(path / "config.json") if (path / "config.json").is_file() else _config_kwargs(VAE_CONFIG_PATH)
    files = checkpoint_files(path)
    if not files:
        raise ConfigValidationError(f"model.vae_path: no .safetensors found in {path}.")
    return _vae_from_state(_read_vae_state(files), vae_cfg, dtype, "vae_path")


def _vae_from_file(path: Path, dtype: torch.dtype):
    state = _read_vae_state([path])
    if is_ldm_vae_layout(state):
        state = convert_ldm_vae_state_dict(state)
    vae_cfg = _config_kwargs(VAE_CONFIG_PATH)
    width = state["decoder.conv_in.weight"].shape[0] if "decoder.conv_in.weight" in state else None
    if width == vae_cfg["block_out_channels"][-1]:
        vae_cfg["decoder_block_out_channels"] = list(vae_cfg["block_out_channels"])
    elif width == SMALL_DECODER_CHANNELS[-1]:
        vae_cfg["decoder_block_out_channels"] = list(SMALL_DECODER_CHANNELS)
    else:
        raise ConfigValidationError(f"model.vae_path: unrecognized FLUX.2 VAE decoder width {width}.")
    return _vae_from_state(state, vae_cfg, dtype, "vae_path")


def load_vae(path: str | Path, dtype: torch.dtype = torch.float32):
    """Load the FLUX.2 VAE (frozen, eval: its ``bn`` statistics must not update). MOSS-Image runs it
    in fp32 (``force_upcast``); ``dtype`` is for tests."""
    return load_file_or_folder(
        path, "vae_path", folder=lambda p: _vae_from_folder(p, dtype), file=lambda p: _vae_from_file(p, dtype)
    )


# ---- text encoder -------------------------------------------------------------------------------


def _text_key(key: str) -> str | None:
    """Checkpoint key -> ``Qwen3_5TextModel`` key, or ``None`` when it is not a text-decoder key.
    Accepts the transformers layout (``language_model.layers.N...``), Comfy-Org's (``model.language_model...``,
    plus ``visual.*`` / ``lm_head`` / ``mtp.*`` which are dropped) and bare keys."""
    key = re.sub(r"^model\.", "", key)
    if key.startswith(("visual.", "lm_head.", "mtp.")):
        return None
    if key.startswith("language_model."):
        key = key[len("language_model.") :]
    return key


def load_text_config(path: str | Path):
    """The ``Qwen3_5TextConfig`` of the text encoder: the folder's ``config.json`` when it has one,
    else the bundled Qwen3.5-9B config (single files carry none)."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(config_dir_or_bundled(Path(path), TEXT_ENCODER_ASSETS)).text_config


def load_text_encoder(path: str | Path, dtype: torch.dtype):
    """Load the Qwen3.5-9B text decoder (``Qwen3_5TextModel``) from a transformers folder or a single
    file, reading only its tensors. Frozen, eval."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import Qwen3_5TextModel

    path = require_exists(path, "text_encoder_path")
    files = checkpoint_files(path)
    if not files:
        raise ConfigValidationError(f"model.text_encoder_path: no .safetensors found in {path}.")
    state_dict = {}
    for file in files:
        with safe_open(str(file), framework="pt") as handle:
            keys = list(handle.keys())
            if any(k.endswith(_QUANT_KEY_SUFFIXES) for k in keys) or any(
                handle.get_slice(k).get_dtype() in _QUANTIZED_DTYPE_NAMES for k in keys
            ):
                raise ConfigValidationError(
                    "model.text_encoder_path points at a pre-quantized (int8 / w4a8) Qwen3.5 file, which cannot "
                    "encode captions here; use qwen3.5_9b_bf16.safetensors or the diffusers text_encoder/ folder."
                )
            for k in keys:
                new = _text_key(k)
                if new is not None:
                    state_dict[new] = handle.get_tensor(k).to(dtype)
    text_config = load_text_config(path)
    # include_buffers=False: the rotary inv_freq buffers are not in checkpoints and stay real.
    with init_empty_weights(include_buffers=False):
        model = Qwen3_5TextModel._from_config(text_config)
    expected = set(model.state_dict())
    missing, unexpected = expected - set(state_dict), set(state_dict) - expected
    if missing or unexpected:
        raise ConfigValidationError(
            f"model.text_encoder_path: not a Qwen3.5 text encoder of the expected architecture; {len(missing)} "
            f"missing keys (e.g. {sorted(missing)[:3]}), {len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})."
        )
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model


def load_tokenizer(path: str | Path | None):
    """The tokenizer folder's ``AutoTokenizer`` (``path`` must exist), else the bundled Qwen3.5 one."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(require_exists(path, "tokenizer_path") if path else TOKENIZER_ASSETS)
