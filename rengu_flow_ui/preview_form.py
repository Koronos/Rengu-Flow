"""Preview entry field metadata for the training config UI."""

from __future__ import annotations

from typing import Any

# Capability gates derived from each model's ``preview_spec`` (registry/model_spec.py
# ModelCapability.__post_init__), so no model list lives here.
# Edge case: the memory knobs below are honored by every DiT preview runner (preview_dit_memory).
WHEN_DIT_PREVIEW = {"capability": "preview_dit_memory"}
# preview_blocks_to_swap streams DiT blocks through the preview offloader (memory_knobs).
WHEN_DIT_PREVIEW_BLOCK_SWAP = {"capability": "preview_block_swap"}
# preview_offload_dit_for_decode (memory_knobs).
WHEN_DECODE_OFFLOAD_PREVIEW = {"capability": "preview_decode_offload"}
# [preview] guidance_mode: models whose spec declares that preview option.
WHEN_GUIDANCE_MODE_PREVIEW = {"capability": "preview_guidance_mode"}
# Edit previews (condition images per prompt): ``preview_spec.edit``.
WHEN_EDIT_PREVIEW = {"capability": "edit"}


def _entry_field(
    path: str,
    label: str,
    ftype: str,
    *,
    default: Any = None,
    description: str = "",
    required: bool = False,
    importance: str = "advanced",
    when: dict[str, Any] | None = None,
    min_value: float | None = None,
    placeholder: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": path,
        "label": label,
        "type": ftype,
        "description": description,
        "importance": "required" if required else importance,
        "required": required,
        "recommended": importance == "recommended",
    }
    if default is not None:
        out["default"] = default
    if when:
        out["when"] = when
    if min_value is not None:
        out["min"] = min_value
    if placeholder:
        out["placeholder"] = placeholder
    return out


def get_preview_entry_fields() -> list[dict[str, Any]]:
    """Fields for one ``[[preview.prompts]]`` row (paths are relative to the table)."""
    return [
        _entry_field(
            "name",
            "TensorBoard tag",
            "string",
            importance="recommended",
            placeholder="portrait",
            description="Optional name for preview/… in TensorBoard. Defaults to prompt_N.",
        ),
        _entry_field(
            "prompt",
            "Prompt",
            "string",
            required=True,
            importance="required",
            placeholder="Describe the image to generate during training…",
        ),
        _entry_field(
            "negative_prompt",
            "Negative prompt",
            "string",
            description="Overrides the global negative prompt for this preview only.",
        ),
        _entry_field(
            "control_images",
            "Condition images (edit preview)",
            "string_list",
            when=WHEN_EDIT_PREVIEW,
            placeholder="path/to/source.png",
            description=(
                "Image paths to edit: the prompt becomes the edit instruction and the preview edits "
                "these images (in order) instead of generating from text. Each is resized like a "
                "training control image (own aspect, area preview.control_resolution²); the output "
                "keeps the last image's aspect at the width × height area."
            ),
        ),
        _entry_field("width", "Width", "integer", min_value=1),
        _entry_field("height", "Height", "integer", min_value=1),
        _entry_field("num_inference_steps", "Inference steps", "integer", min_value=1),
        _entry_field("guidance_scale", "Guidance scale", "number", min_value=0),
        _entry_field("seed", "Seed", "integer"),
        _entry_field("seed_stride", "Seed stride", "integer"),
        _entry_field("preview_every_n_steps", "Every N steps", "integer", min_value=1),
        _entry_field("preview_every_n_epochs", "Every N epochs", "integer", min_value=1),
        _entry_field(
            "preview_offload_text_encoder",
            "Offload text encoder",
            "boolean",
            default=True,
            when=WHEN_DIT_PREVIEW,
        ),
        _entry_field(
            "preview_blocks_to_swap",
            "Blocks to swap",
            "integer",
            default=0,
            min_value=0,
            when=WHEN_DIT_PREVIEW_BLOCK_SWAP,
        ),
        _entry_field(
            "preview_offload_dit_for_decode",
            "Offload DiT for VAE decode",
            "boolean",
            default=False,
            when=WHEN_DECODE_OFFLOAD_PREVIEW,
        ),
        _entry_field(
            "preview_save_png",
            "Save PNG to run folder",
            "boolean",
            default=False,
        ),
    ]


PREVIEW_GLOBAL_PATHS: frozenset[str] = frozenset(
    {
        "preview.enabled",
        "preview.negative_prompt",
        "preview.width",
        "preview.height",
        "preview.control_resolution",
        "preview.num_inference_steps",
        "preview.guidance_scale",
        "preview.guidance_mode",
        "preview.seed",
        "preview.seed_stride",
        "preview.preview_every_n_steps",
        "preview.preview_every_n_epochs",
        "preview.preview_before_first_step",
        "disable_block_swap_for_preview",
        "preview.preview_offload_text_encoder",
        "preview.preview_blocks_to_swap",
        "preview.preview_offload_dit_for_decode",
        "preview.preview_save_png",
    }
)
