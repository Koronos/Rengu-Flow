"""SDXL model descriptor (torch-free; registered via ``registry/builtin_models.py``)."""

from __future__ import annotations

from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES
from rengu_flow.registry import field_library
from rengu_flow.registry.model_spec import AdapterExport, ComponentPath, ModelCapability, PreviewSpec

# Kohya-flat roots of the adapted networks (canonical ``text_model`` text-encoder names; TE2's
# text_projection sits outside text_model).
_KOHYA_ROOTS = ("lora_unet_", "lora_te1_text_model_", "lora_te2_text_model_", "lora_te2_text_projection")

SPEC = ModelCapability(
    type_id="sdxl",
    display_name="SDXL",
    adapters=["lora", "lokr", *LYCORIS_ADAPTER_TYPES],
    full_finetune=True,
    features={"block_swap": True},
    # Text embeddings depend on clip_skip and on the checkpoint's own CLIP weights; v1 =
    # every 75-token chunk padded to 77 tokens (caches from before hold unpadded rows).
    text_cache_keys={"clip_skip": None},
    text_cache_version=1,
    text_cache_files=["checkpoint_path"],
    pipeline="rengu_flow.model.sdxl:SDXLPipeline",
    env_prefix="SDXL",
    components=(ComponentPath("checkpoint_path", expect="file"),),
    text_cache="optional",
    # diffusers StableDiffusionXLPipeline; the runner falls back to 20 steps / CFG 7.0 at sample
    # time (utils/preview._run_sdxl_previews) — set_config_defaults writes no [preview] defaults.
    preview_spec=PreviewSpec(
        runner="sdxl",
        steps=20,
        guidance=7.0,
        write_defaults=False,
        memory_knobs=frozenset(),
    ),
    # Legacy diffusion-pipe key (TOML-only, hidden in the UI); no pipeline reads it.
    model_defaults={"guidance": 1.0},
    install_profiles=(),
    user_doc="docs/user/training-sdxl-lora-lokr.md",
    # Kohya-style per-adapter layouts (networks.lora_sdxl / lokr_vendored / lycoris_sdxl).
    export_prefix=None,
    adapter_exports={
        # lora_sdxl.save: kohya LoRA (ComfyUI / A1111).
        "lora": AdapterExport(file="lora.safetensors", style="kohya", roots=_KOHYA_ROOTS),
        # lokr_vendored.save: dotted module paths, LyCORIS-style weights.
        "lokr": AdapterExport(
            style="dotted",
            roots=("unet.", "text_encoder.text_model.", "text_encoder_2.text_model.", "text_encoder_2.text_projection."),
        ),
        # lycoris_sdxl.save: kohya-flat names, the algorithm's own weights.
        "lycoris": AdapterExport(style="kohya", roots=_KOHYA_ROOTS),
    },
    testkit="rengu_flow.model.sdxl_testkit",
    model_fields=[
        field_library.component_path(
            "checkpoint_path",
            "Base model (.safetensors)",
            "Full SDXL weights file — not a LoRA.",
            placeholder="path/to/sdxl.safetensors",
            required=True,
            help=(
                "The full pretrained SDXL you downloaded (often a single large .safetensors). This "
                "is not a LoRA and not your dataset. Diffusers-style directories are also accepted."
            ),
        ),
        {
            "path": "model.guidance",
            "label": "Guidance",
            "type": "number",
            "default": 1.0,
            "ui": False,
            "description": "Parsed in TOML only; training uses preview.guidance_scale for CFG.",
            "help": (
                "Parsed and defaulted in config but not used by SDXL training. Use "
                "preview.guidance_scale for sample images during training."
            ),
        },
        {
            "path": "model.freeze_text_encoders",
            "label": "Freeze text encoders",
            "type": "boolean",
            "description": "UNet-only style training when omitted adapter.",
            "help": (
                "Keeps both text encoders frozen so their weights don't update — reduces VRAM and "
                "speeds up each training step. Turn off only if you need the model to respond "
                "differently to new prompt words (e.g. training a new concept with a novel trigger "
                "token)."
            ),
            "doc": "docs/user/full-model-training-sdxl.md",
        },
        {
            "path": "model.v_pred",
            "label": "V-prediction",
            "type": "boolean",
            "default": False,
            "description": (
                "Trains toward velocity instead of noise — required for v-pred checkpoints (e.g. "
                "NoobAI vpred). Wrong setting trains toward the wrong target: previews wash out or "
                "oversaturate."
            ),
        },
        {
            "path": "model.zero_terminal_snr",
            "label": "Zero terminal SNR",
            "type": "boolean",
            "placeholder": "empty = on with v_pred, off otherwise",
            "description": (
                "Rescales the noise schedule so the last timestep is pure noise "
                "(https://arxiv.org/abs/2305.08891) and samples previews with trailing timestep "
                "spacing. Defaults to the v_pred value (v-pred checkpoints are trained this way); "
                "set it explicitly to override."
            ),
        },
        {
            "path": "model.clip_skip",
            "label": "CLIP skip",
            "type": "integer",
            "min": 0,
            "placeholder": "empty = standard -2 layer",
            "description": (
                "Diffusers semantics: hidden_states[-(clip_skip + 2)]. Empty (or 0) already uses "
                "the penultimate layer, which is what A1111's 'Clip skip 2' means; clip_skip=1 "
                "here is A1111's 3, clip_skip=2 its 4. So for anime checkpoints trained with A1111 "
                "'Clip skip 2', leave this empty. Changing it re-encodes the text cache."
            ),
        },
        {
            "path": "model.min_snr_gamma",
            "label": "Min-SNR gamma",
            "type": "number",
            "placeholder": "empty = off",
            "description": (
                "Min-SNR loss weighting (typical value 5): caps the loss weight of low-SNR "
                "(high-noise) timesteps so they don't dominate training. Composes with "
                "debiased_estimation_loss when both are set — this trainer applies both "
                "independently, not as alternatives."
            ),
        },
        {
            "path": "model.debiased_estimation_loss",
            "label": "Debiased estimation loss",
            "type": "boolean",
            "default": False,
            "description": (
                "Debiased-estimation loss weighting by timestep SNR. Composes with min_snr_gamma "
                "when both are set — this trainer applies both independently, not as alternatives."
            ),
        },
        {
            "path": "model.unet_lr",
            "label": "UNet LR override",
            "type": "number",
            "placeholder": "empty = optimizer.lr",
            "description": (
                "Per-group LR override for the UNet parameter group; empty uses optimizer.lr. 0 "
                "sets that group's lr to 0 (no updates) while the other groups keep training."
            ),
        },
        {
            "path": "model.text_encoder_1_lr",
            "label": "Text encoder 1 LR override",
            "type": "number",
            "placeholder": "empty = optimizer.lr",
            "description": (
                "Per-group LR override for text_encoder (CLIP-L); empty uses optimizer.lr. 0 sets "
                "that group's lr to 0 (no updates)."
            ),
        },
        {
            "path": "model.text_encoder_2_lr",
            "label": "Text encoder 2 LR override",
            "type": "number",
            "placeholder": "empty = optimizer.lr",
            "description": (
                "Per-group LR override for text_encoder_2 (OpenCLIP-bigG); empty uses "
                "optimizer.lr. 0 sets that group's lr to 0 (no updates)."
            ),
        },
        field_library.diffusion_model_dtype(
            "Autocast dtype for the UNet forward pass; defaults to Model dtype.",
            label="UNet forward dtype",
            help=(
                "Sets training forward autocast (main.py). SDXL has no separate checkpoint-load "
                "dtype path (unlike Cosmos/Krea 2's transformer_dtype) — this only changes the "
                "forward-pass compute dtype, not how weights are loaded from disk."
            ),
        ),
    ],
)
