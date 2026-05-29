"""Compat shims for diffusers single-file loading with transformers 5.x."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def strip_text_model_state_dict_prefix(state_dict: dict) -> dict:
    """Map ``text_model.*`` keys to flat keys for transformers 5 ``CLIPTextModel``."""
    if not state_dict:
        return state_dict
    if not any(k.startswith("text_model.") for k in state_dict):
        return state_dict
    out: dict = {}
    for key, value in state_dict.items():
        if key.startswith("text_model."):
            out[key[len("text_model.") :]] = value
        else:
            out[key] = value
    return out


def apply_diffusers_transformers_v5_single_file_patch() -> None:
    """Patch diffusers ``create_diffusers_clip_model_from_ldm`` for flat ``CLIPTextModel``."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import transformers
        from packaging.version import Version
    except ImportError:
        return

    if Version(transformers.__version__) < Version("5.0.0"):
        return

    import diffusers.loaders.single_file_utils as sfu

    def _position_embedding_dim(model) -> int:
        if hasattr(model, "text_model"):
            return model.text_model.embeddings.position_embedding.weight.shape[-1]
        return model.embeddings.position_embedding.weight.shape[-1]

    def patched(cls, checkpoint, subfolder="", config=None, torch_dtype=None, local_files_only=None, is_legacy_loading=False):
        import torch
        from contextlib import nullcontext

        from accelerate import init_empty_weights

        from diffusers.loaders.single_file_utils import (
            CHECKPOINT_KEY_NAMES,
            convert_ldm_clip_checkpoint,
            convert_open_clip_checkpoint,
            empty_device_cache,
            fetch_diffusers_config,
            is_accelerate_available,
            is_clip_model,
            is_clip_sd3_model,
            is_clip_sdxl_model,
            is_open_clip_model,
            is_open_clip_sd3_model,
            is_open_clip_sdxl_model,
            is_open_clip_sdxl_refiner_model,
            load_model_dict_into_meta,
        )

        if config:
            config = {"pretrained_model_name_or_path": config}
        else:
            config = fetch_diffusers_config(checkpoint)

        if is_legacy_loading:
            logger.warning(
                "Detected legacy CLIP loading behavior. Please run `from_single_file` with "
                "`local_files_only=False once to update the local cache directory with the necessary "
                "CLIP model config files. Attempting to load CLIP model from legacy cache directory."
            )
            if is_clip_model(checkpoint) or is_clip_sdxl_model(checkpoint):
                config["pretrained_model_name_or_path"] = "openai/clip-vit-large-patch14"
                subfolder = ""
            elif is_open_clip_model(checkpoint):
                config["pretrained_model_name_or_path"] = "stabilityai/stable-diffusion-2"
                subfolder = "text_encoder"
            else:
                config["pretrained_model_name_or_path"] = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
                subfolder = ""

        model_config = cls.config_class.from_pretrained(**config, subfolder=subfolder, local_files_only=local_files_only)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = cls(model_config)

        position_embedding_dim = _position_embedding_dim(model)
        flat_clip = not hasattr(model, "text_model")

        if is_clip_model(checkpoint):
            diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint)
        elif (
            is_clip_sdxl_model(checkpoint)
            and checkpoint[CHECKPOINT_KEY_NAMES["clip_sdxl"]].shape[-1] == position_embedding_dim
        ):
            diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint)
        elif (
            is_clip_sd3_model(checkpoint)
            and checkpoint[CHECKPOINT_KEY_NAMES["clip_sd3"]].shape[-1] == position_embedding_dim
        ):
            diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint, "text_encoders.clip_l.transformer.")
            diffusers_format_checkpoint["text_projection.weight"] = torch.eye(position_embedding_dim)
        elif is_open_clip_model(checkpoint):
            prefix = "cond_stage_model.model."
            diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)
        elif (
            is_open_clip_sdxl_model(checkpoint)
            and checkpoint[CHECKPOINT_KEY_NAMES["open_clip_sdxl"]].shape[-1] == position_embedding_dim
        ):
            prefix = "conditioner.embedders.1.model."
            diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)
        elif is_open_clip_sdxl_refiner_model(checkpoint):
            prefix = "conditioner.embedders.0.model."
            diffusers_format_checkpoint = convert_open_clip_checkpoint(model, checkpoint, prefix=prefix)
        elif (
            is_open_clip_sd3_model(checkpoint)
            and checkpoint[CHECKPOINT_KEY_NAMES["open_clip_sd3"]].shape[-1] == position_embedding_dim
        ):
            diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint, "text_encoders.clip_g.transformer.")
        else:
            raise ValueError("The provided checkpoint does not seem to contain a valid CLIP model.")

        if flat_clip:
            diffusers_format_checkpoint = strip_text_model_state_dict_prefix(diffusers_format_checkpoint)

        if is_accelerate_available():
            load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
            empty_device_cache()
        else:
            model.load_state_dict(diffusers_format_checkpoint, strict=False)

        if torch_dtype is not None:
            model.to(torch_dtype)

        return model

    sfu.create_diffusers_clip_model_from_ldm = patched
    _PATCHED = True
    logger.debug("Applied diffusers single-file patch for transformers 5 CLIPTextModel")
