"""Qwen-Image 2.1 model descriptor (torch-free; registered via ``registry/builtin_models.py``)."""

from __future__ import annotations

from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES
from rengu_flow.registry import field_library
from rengu_flow.registry.model_spec import (
    LINEAR_ONLY_ADAPTERS,
    ComponentPath,
    ModelCapability,
    PreviewSpec,
    QuantSpec,
)

_QUANT = QuantSpec(fp8="tensorwise", nf4=True, grad_mode=True)
_QUANT_FIELDS = field_library.quant_fields(
    _QUANT,
    nf4={
        "description": (
            "Stores the frozen DiT block linears as 4-bit NF4 (~4 GB instead of ~14 GB bf16); the "
            "adapter trains on top at full precision. Pair with adapter type LoKr (LyCORIS kinds "
            "refuse a quantized base)."
        ),
        "help": (
            "Adapter training only — the quantized base stays frozen while the adapter trains in "
            "full precision on top. The 7B DiT drops from ~14 GB bf16 to ~4 GB. Mutually exclusive "
            "with model.transformer_fp8_matmul. Use adapter type LoKr: the LyCORIS kinds reject a "
            "quantized base."
        ),
    },
    fp8={
        "description": (
            "Tensorwise-scaled e4m3 quantization of the frozen DiT's block linears (mutually "
            "exclusive with transformer_4bit): the base drops from ~14.2 to ~7.3 GB. Needs an "
            "sm89+ GPU (RTX 40xx / Ada); pair with blocks_to_swap on 8-16 GB cards."
        ),
        "help": (
            "Adapter training only. The 7B DiT drops from ~14.2 GB to ~7.3 GB. Needs an sm89+ GPU "
            "(RTX 40xx / Ada); on older cards use transformer_4bit with LoKr instead. Mutually "
            "exclusive with model.transformer_4bit. The 8 GB recipe pairs it with blocks_to_swap = "
            "24 and activation checkpointing."
        ),
    },
    grad_mode={
        "description": (
            "Input-gradient GEMM precision through the frozen fp8 base, only used when "
            "transformer_fp8_matmul is on: fp8 is faster, bf16 (default) keeps the clean gradient."
        ),
        "help": (
            "'bf16' (default) keeps the backward's input-gradient GEMM unquantized; 'fp8' runs it "
            "through the same e4m3 tensorwise kernel as the forward (faster). Only takes effect "
            "when model.transformer_fp8_matmul is true."
        ),
    },
)

# Qwen-Image 2.1 (Qwen/Qwen-Image-2.1): 7B single-stream block-causal DiT (~14 GB bf16) +
# Qwen3-VL-8B text encoder (~17.5 GB, cached + layer-streamed) + RGBA 16x VAE. Text-to-image and
# image-conditioned editing (dataset control_path; preview control_images).
SPEC = ModelCapability(
    type_id="qwen_image21",
    display_name="Qwen-Image 2.1",
    adapters=["lora", "lokr", *LYCORIS_ADAPTER_TYPES],
    full_finetune=True,
    # edit: trains on [[directory]] control_path (condition images) and renders previews
    # with control_images.
    # Linear-only: no Conv modules and only non-affine LayerNorm / custom RMSNorm in the
    # DiT (dit.py), so train_conv/use_tucker/train_norm are no-ops.
    features={"block_swap": True, "edit": True, LINEAR_ONLY_ADAPTERS: True},
    branding_note=(
        "Use ComfyUI's bf16 files (qwen_image_2.1_bf16, qwen_image_2.1_vae_bf16, "
        "qwen3vl_8b_bf16) or the Qwen/Qwen-Image-2.1 diffusers download. Trains text-to-image "
        "and image editing (dataset control_path), also mixed in one run."
    ),
    model_validation={
        "one_of": [
            ["transformer_path", "diffusers_path"],
            ["vae_path", "diffusers_path"],
            ["text_encoder_path", "diffusers_path"],
        ],
    },
    adapter_layer_groups=[
        "attention",
        "feedforward",
        "modulation",
        "text_projection",
        "image_in_out",
    ],
    adapter_module_roots=[
        "transformer_blocks",
        "modulation",
        "time_text_embed",
        "txt_in",
        "img_in",
        "norm_out",
        "proj_out",
        "pos_embed",
    ],
    pipeline="rengu_flow.model.qwen_image21:QwenImage21Pipeline",
    env_prefix="QWEN_IMAGE21",
    # Each component: a single file or its diffusers subfolder; diffusers_path = the whole
    # Qwen/Qwen-Image-2.1 download (fills any component left empty).
    components=(
        ComponentPath("transformer_path", expect="any"),
        ComponentPath("vae_path", expect="any"),
        ComponentPath("text_encoder_path", expect="any"),
        ComponentPath("diffusers_path", expect="dir"),
    ),
    bundle_key="diffusers_path",
    text_cache="required",
    # Always tensorwise e4m3: no fp8_matmul_dtype knob.
    quant=_QUANT,
    # Samples without CFG (reference true_cfg_scale = 1.0); its reference 40 steps are trimmed
    # to 28 for previews. Renders edit previews (control_images).
    preview_spec=PreviewSpec(
        runner="dit",
        steps=28,
        guidance=1.0,
        edit=True,
        help={
            "num_inference_steps": {
                "description": "Denoising steps per preview image (Qwen-Image 2.1 default 28).",
                "help": (
                    "Euler steps over the reference sigma schedule (resolution-aware shift, last "
                    "step at 0.02). The reference sampler uses 40; 28 keeps previews faster."
                ),
            },
            "guidance_scale": {
                "description": "CFG scale for previews; Qwen-Image 2.1 samples without CFG (default 1.0).",
                "help": (
                    "1.0 runs one conditional pass per step, like the reference. Above 1.0 CFG runs "
                    "with the negative prompt (an empty one is encoded as a single space) as "
                    "neg + g*(pos - neg), doubling the preview cost."
                ),
            },
        },
    ),
    install_profiles=("cosmos",),
    user_doc="docs/user/training-qwen-image21.md",
    export_prefix="transformer.",
    testkit="rengu_flow.model.qwen_image21.testkit",
    # (L,) vision-slot mask of edit prompts, one entry per embedding position.
    ragged_cache_keys=frozenset({"image_pad_mask"}),
    model_fields=[
        # The component paths are not "required": each is one_of(<component>_path,
        # diffusers_path) — the diffusers folder below fills any left empty.
        field_library.component_path(
            "transformer_path",
            "Main model (.safetensors)",
            (
                "The 14 GB DiT you train: ComfyUI's qwen_image_2.1_bf16.safetensors (a diffusers "
                "transformer/ folder also works). Optional when the diffusers folder below is set."
            ),
            placeholder="path/to/qwen_image_2.1_bf16.safetensors",
            path_expect="any",
            importance="recommended",
            help=(
                "ComfyUI's qwen_image_2.1_bf16.safetensors (its fused img_mlp.gate_up is split on "
                "load), a diffusers-layout file or the diffusers transformer/ folder. "
                "Pre-quantized int8_convrot / fp8 files cannot be trained; use the bf16 file."
            ),
        ),
        field_library.component_path(
            "vae_path",
            "Image VAE (.safetensors)",
            (
                "Qwen-Image 2.1's own RGBA VAE: ComfyUI's qwen_image_2.1_vae_bf16.safetensors (or "
                "the diffusers vae/ folder). Optional when the diffusers folder below is set."
            ),
            placeholder="path/to/qwen_image_2.1_vae_bf16.safetensors",
            path_expect="any",
            importance="recommended",
            help=(
                "ComfyUI's qwen_image_2.1_vae_bf16.safetensors (original layout, converted on "
                "load), a diffusers-layout file or the diffusers vae/ folder. Training reads your "
                "images through it to build the latent cache."
            ),
        ),
        field_library.component_path(
            "text_encoder_path",
            "Text encoder — Qwen3-VL-8B (.safetensors)",
            (
                "ComfyUI's qwen3vl_8b_bf16.safetensors (or the transformers text_encoder/ folder); "
                "includes the vision tower edit training needs. Optional when the diffusers folder "
                "below is set."
            ),
            placeholder="path/to/qwen3vl_8b_bf16.safetensors",
            path_expect="any",
            importance="recommended",
            help=(
                "ComfyUI's qwen3vl_8b_bf16.safetensors or the diffusers text_encoder/ folder; both "
                "carry the vision tower that edit training (control_path) uses. Captions are "
                "encoded once into the cache; int8 / w4a8 files are refused."
            ),
        ),
        field_library.bundle_folder(
            "diffusers_path",
            (
                "The whole Qwen/Qwen-Image-2.1 diffusers download (transformer/, vae/, "
                "text_encoder/, processor/); fills any of the three file fields above left empty. "
                "Set either this or the three files."
            ),
            placeholder="path/to/Qwen-Image-2.1",
            help=(
                "Optional. Point at the folder holding transformer/, vae/, text_encoder/ and "
                "processor/ (e.g. the Hugging Face cache snapshot) and leave the file fields empty "
                "— each resolves to <folder>/<subfolder>; files you set explicitly always win. "
                "Never a repo id; nothing is downloaded."
            ),
        ),
        field_library.component_path(
            "processor_path",
            "Processor / tokenizer override",
            (
                "Folder with the tokenizer files; defaults to <diffusers folder>/processor, else "
                "the bundled Qwen3-VL tokenizer."
            ),
            path_expect="dir",
            show_if_set=True,
            help=(
                "Defaults to <diffusers_path>/processor, else the Qwen3-VL tokenizer and image "
                "processor bundled with rengu (identical tokenization of the Qwen-Image 2.1 prompt "
                "template)."
            ),
        ),
        {
            "path": "model.text_encoder_offload",
            "label": "Text encoder placement",
            "type": "select",
            "options": ["auto", "stream", "none"],
            "default": "auto",
            "description": (
                "How the 17.5 GB Qwen3-VL-8B encoder runs while captions are cached: auto "
                "(default) streams its 36 decoder layers from pinned host RAM when it does not fit "
                "in free VRAM, stream always does, none loads it whole onto the GPU."
            ),
            "help": (
                "auto (default): stream the 36 decoder layers one at a time from pinned host RAM "
                "when the encoder does not fit in free VRAM (~2 GB VRAM, ~16 GB host RAM); stream: "
                "always; none: load it whole onto the GPU (needs ~20 GB free). The encoder loads "
                "only when captions still need encoding and is unloaded after caching."
            ),
        },
        _QUANT_FIELDS["fp8"],
        _QUANT_FIELDS["nf4"],
        _QUANT_FIELDS["grad_mode"],
        field_library.transformer_dtype(
            help=(
                "The VAE, the Qwen3-VL text encoder and the adapters keep model.dtype. Leave empty "
                "unless you need a different DiT load precision."
            ),
        ),
        field_library.shift(
            (
                "Fixed timestep shift (> 0), overriding the resolution-aware dynamic shift of the "
                "reference scheduler (exponential, mu 0.5 at 256 latent tokens to 0.9 at 8192; "
                "1024x1024 = 4096 tokens). Empty keeps the dynamic default."
            ),
            placeholder="empty = resolution-aware dynamic shift",
            help=(
                "By default training timesteps follow the reference scheduler's exponential shift: "
                "mu interpolates from 0.5 at 256 latent tokens to 0.9 at 8192 (1024x1024 = 4096 "
                "tokens, mu ~0.70). A number here replaces it for every resolution."
            ),
        ),
        field_library.sigmoid_scale(
            (
                "Scale applied to the logit-normal sample before sigmoid, only used when "
                "timestep_sample_method is logit_normal."
            ),
            help=(
                "Only used with timestep_sample_method = logit_normal. Raising it pushes sampled "
                "timesteps toward the extremes (near 0 or 1)."
            ),
        ),
        field_library.timestep_sample_method(
            "How training timesteps are sampled per step: logit_normal (default) or uniform.",
            help=(
                "logit_normal concentrates steps on mid-range noise levels; uniform samples t "
                "evenly in [0, 1]. The resolution-aware shift (or model.shift) is applied on top "
                "of either."
            ),
        ),
    ],
)
