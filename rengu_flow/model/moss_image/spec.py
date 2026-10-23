"""MOSS-Image model descriptor (torch-free; registered via ``registry/builtin_models.py``)."""

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

_DOC = "docs/user/training-moss-image.md"

_QUANT = QuantSpec(fp8="tensorwise", nf4=True, grad_mode=True)
_QUANT_FIELDS = field_library.quant_fields(
    _QUANT,
    nf4={
        "description": (
            "Stores the frozen DiT block linears as 4-bit NF4 (~3 GB instead of ~10 GB bf16); the adapter "
            "trains on top at full precision. Pair with adapter type LoKr (LyCORIS kinds refuse a "
            "quantized base)."
        ),
        "help": (
            "Adapter training only: the quantized base stays frozen while the adapter trains in full "
            "precision on top. The 5B DiT drops from ~10.2 GB bf16 to ~3 GB. Mutually exclusive with "
            "model.transformer_fp8_matmul. Use adapter type LoKr: the LyCORIS kinds reject a quantized base."
        ),
    },
    fp8={
        "description": (
            "Tensorwise-scaled e4m3 quantization of the frozen DiT's block linears (mutually exclusive "
            "with transformer_4bit): the base drops from ~10.2 to ~5.2 GB. Needs an sm89+ GPU "
            "(RTX 40xx / Ada); pair with blocks_to_swap to fit smaller cards."
        ),
        "help": (
            "Adapter training only. The 5B DiT drops from ~10.2 GB to ~5.2 GB. Needs an sm89+ GPU "
            "(RTX 40xx / Ada); on older cards use transformer_4bit with LoKr instead. Mutually exclusive "
            "with model.transformer_4bit. Pair it with blocks_to_swap and activation checkpointing "
            "on smaller cards."
        ),
    },
    grad_mode={
        "description": (
            "Input-gradient GEMM precision through the frozen fp8 base, only used when "
            "transformer_fp8_matmul is on: fp8 is faster, bf16 (default) keeps the clean gradient."
        ),
        "help": (
            "'bf16' (default) keeps the backward's input-gradient GEMM unquantized; 'fp8' runs it through "
            "the same e4m3 tensorwise kernel as the forward (faster). Only takes effect when "
            "model.transformer_fp8_matmul is true."
        ),
    },
)

# MOSS-Image (OpenMOSS-Team/MOSS-Image-Base / -SFT, Apache-2.0): 5B single-stream DiT (10.2 GB bf16) +
# Qwen3.5-9B text encoder (cached + layer-streamed) + FLUX.2 VAE (16x with the 2x2 patchify).
SPEC = ModelCapability(
    type_id="moss_image",
    display_name="MOSS-Image",
    adapters=["lora", "lokr", *LYCORIS_ADAPTER_TYPES],
    full_finetune=True,
    # Linear-only: no Conv modules and only custom RMSNorm in the DiT (dit.py), so train_conv /
    # use_tucker / train_norm are no-ops.
    features={"block_swap": True, LINEAR_ONLY_ADAPTERS: True},
    transformer_blocks=42,
    branding_note=(
        "Train MOSS-Image-Base (OpenMOSS-Team/MOSS-Image-Base, the pretrained model meant for LoRA / "
        "finetuning; MOSS-Image-SFT loads the same way). Use the bf16 files: the diffusers download or "
        "Comfy-Org's moss_image_base_bf16, qwen3.5_9b_bf16 and flux2-vae_small (never the int8_convrot / "
        "w4a8 ones)."
    ),
    model_validation={
        "one_of": [
            ["transformer_path", "diffusers_path"],
            ["vae_path", "diffusers_path"],
            ["text_encoder_path", "diffusers_path"],
        ],
    },
    # The default adapter scope is the 42 blocks' attention + feed-forward linears (what KohakuMoss
    # defaults to and ComfyUI maps); the other groups add the rest of the network.
    adapter_layer_groups=[
        "attention",
        "feedforward",
        "text_refiner",
        "text_projection",
        "modulation",
        "image_in_out",
    ],
    adapter_default_layer_groups=["attention", "feedforward"],
    adapter_module_roots=[
        "video_embedder",
        "text_embedder",
        "time_embedding",
        "shared_block_modulation",
        "layers",
        "final_norm_video",
        "final_linear_video",
        "video_head_modulation",
    ],
    pipeline="rengu_flow.model.moss_image:MOSSImagePipeline",
    env_prefix="MOSS_IMAGE",
    # Each component: a single file or its diffusers subfolder; diffusers_path = the whole
    # OpenMOSS-Team/MOSS-Image-Base download (fills any component left empty).
    components=(
        ComponentPath("transformer_path", expect="any"),
        ComponentPath("vae_path", expect="any"),
        ComponentPath("text_encoder_path", expect="any"),
        ComponentPath("diffusers_path", expect="dir"),
    ),
    bundle_key="diffusers_path",
    # The encoder's file keys the text cache (swapping it re-encodes).
    text_cache_files=["text_encoder_path"],
    # A tokenizer override changes the token ids, hence the embeddings: it re-encodes the captions.
    text_cache_keys={"tokenizer_path": None},
    text_cache="required",
    quant=_QUANT,
    # The reference sampler: Euler, 50 steps, true CFG 3.0, fixed shift 5, empty negative prompt,
    # negative branch skipping block 22 (pipeline_moss_image.py __call__ defaults).
    preview_spec=PreviewSpec(
        runner="dit",
        steps=50,
        guidance=3.0,
        help={
            "num_inference_steps": {
                "description": "Denoising steps per preview image (MOSS-Image default 50).",
                "help": (
                    "Euler steps over the reference sigma schedule (fixed shift 5.0 snapped to the 1000-step "
                    "training grid). 50 is the reference; each step runs two forwards when CFG is on."
                ),
            },
            "guidance_scale": {
                "description": "True CFG scale for previews (MOSS-Image default 3.0; 1.0 = off).",
                "help": (
                    "uncond + g * (cond - uncond) in fp32, as the reference (3.0 default, 3-6 recommended). The "
                    "negative branch skips transformer block 22, as the reference does by default. 1.0 runs one "
                    "forward per step."
                ),
            },
        },
    ),
    # Reference recipe: the report's final-stage sigma sampler (Table 6 / Eq. 4), no shift.
    model_defaults={
        "timestep_sample_method": "moss",
        "timestep_mu": -1.09,
        "timestep_std": 0.95,
        "timestep_cutoff": 0.95,
        "timestep_uniform_mix": 0.01,
    },
    refused_dtypes={
        "float16": "a few joint-stream activations exceed the fp16 range (65504) and overflow; use bfloat16",
    },
    install_profiles=("cosmos",),
    user_doc=_DOC,
    export_prefix="diffusion_model.",
    testkit="rengu_flow.model.moss_image.testkit",
    model_fields=[
        # The component paths are not "required": each is one_of(<component>_path, diffusers_path).
        field_library.component_path(
            "transformer_path",
            "Main model (.safetensors)",
            (
                "The 10.2 GB DiT you train: Comfy-Org's moss_image_base_bf16.safetensors (or the diffusers "
                "transformer/ folder). Optional when the diffusers folder below is set."
            ),
            placeholder="path/to/moss_image_base_bf16.safetensors",
            path_expect="any",
            importance="recommended",
            help=(
                "Comfy-Org's moss_image_base_bf16.safetensors (its fused q/k/v is split on load), a single "
                "file in the official layout or the OpenMOSS diffusers transformer/ folder. Pre-quantized "
                "int8_convrot files cannot be trained; use the bf16 file."
            ),
        ),
        field_library.component_path(
            "vae_path",
            "Image VAE (.safetensors)",
            (
                "The FLUX.2 VAE: Comfy-Org's flux2-vae_small.safetensors (the release's own decoder) or "
                "flux2-vae.safetensors, or the diffusers vae/ folder. Optional when the diffusers folder "
                "below is set."
            ),
            placeholder="path/to/flux2-vae_small.safetensors",
            path_expect="any",
            importance="recommended",
            help=(
                "flux2-vae_small.safetensors (diffusers layout, what the OpenMOSS vae/ folder holds) or "
                "flux2-vae.safetensors (original layout, converted on load). Training reads your images "
                "through its encoder to build the latent cache. The release vae/ and flux2-vae_small are the "
                "bf16-rounded version of flux2-vae (batch-norm statistics included): latents differ by that "
                "rounding, so use flux2-vae_small to match the release."
            ),
        ),
        field_library.component_path(
            "text_encoder_path",
            "Text encoder: Qwen3.5-9B (.safetensors)",
            (
                "Comfy-Org's qwen3.5_9b_bf16.safetensors (or the transformers text_encoder/ folder). Optional "
                "when the diffusers folder below is set."
            ),
            placeholder="path/to/qwen3.5_9b_bf16.safetensors",
            path_expect="any",
            importance="recommended",
            help=(
                "qwen3.5_9b_bf16.safetensors (19.3 GB) or the diffusers text_encoder/ folder. Captions are "
                "encoded once into the cache; the vision tower and LM head are never read. int8 / w4a8 files "
                "are refused."
            ),
        ),
        field_library.bundle_folder(
            "diffusers_path",
            (
                "The whole OpenMOSS-Team/MOSS-Image-Base download (transformer/, vae/, text_encoder/, "
                "tokenizer/); fills any of the three file fields above left empty. Set either this or the "
                "three files."
            ),
            placeholder="path/to/MOSS-Image-Base",
            help=(
                "Optional. Point at the folder holding transformer/, vae/, text_encoder/ and tokenizer/ (e.g. "
                "the Hugging Face cache snapshot) and leave the file fields empty: each resolves to "
                "<folder>/<subfolder>; files you set explicitly always win. Never a repo id; nothing is "
                "downloaded."
            ),
        ),
        field_library.component_path(
            "tokenizer_path",
            "Tokenizer override",
            (
                "Folder with the Qwen3.5 tokenizer files; defaults to <diffusers folder>/tokenizer, else the "
                "tokenizer and chat template bundled with rengu."
            ),
            path_expect="dir",
            show_if_set=True,
            help=(
                "Defaults to <diffusers_path>/tokenizer, else the Qwen3.5 tokenizer and chat template "
                "bundled with rengu (copied from the Base repo; identical tokenization)."
            ),
        ),
        {
            "path": "model.text_encoder_offload",
            "label": "Text encoder placement",
            "type": "select",
            "options": ["auto", "stream", "none"],
            "default": "auto",
            "description": (
                "How the 17 GB Qwen3.5-9B encoder runs while captions are cached: auto (default) streams its "
                "32 decoder layers from pinned host RAM when it does not fit in free VRAM, stream always does, "
                "none loads it whole onto the GPU."
            ),
            "help": (
                "auto (default): stream the 32 decoder layers one at a time from pinned host RAM when the "
                "encoder does not fit in free VRAM (~2 GB VRAM, ~17 GB host RAM); stream: always; none: load it "
                "whole onto the GPU (needs ~20 GB free). The encoder loads only when captions still need "
                "encoding and is unloaded after caching."
            ),
        },
        _QUANT_FIELDS["fp8"],
        _QUANT_FIELDS["nf4"],
        _QUANT_FIELDS["grad_mode"],
        field_library.transformer_dtype(
            help=(
                "The VAE (always fp32), the Qwen3.5 text encoder and the adapters keep their own precision. "
                "bf16 is the supported precision: fp16 overflows (a few joint-stream activations exceed "
                "65504) and is refused. Leave empty unless you need a different DiT load precision."
            ),
        ),
        field_library.timestep_sample_method(
            (
                "How training sigmas are sampled: moss (default) is the report's final-stage sampler (a "
                "truncated logit-normal on the 1000-step grid mixed with 1% uniform); logit_normal and uniform "
                "are the generic alternatives."
            ),
            options=("moss", "logit_normal", "uniform"),
            default="moss",
            help=(
                "moss (default): MOSS-Image technical report, Table 6 final stage / Eq. 4, parametrized by the "
                "four timestep_* fields below; no shift. logit_normal concentrates steps on mid-range noise "
                "levels (scaled by sigmoid_scale); uniform samples evenly in [0, 1]. model.shift is applied on "
                "top of any of them when set."
            ),
        ),
        {
            "path": "model.timestep_mu",
            "label": "Sigma sampler mean (mu)",
            "type": "number",
            "default": -1.09,
            "description": (
                "Mean of the logit-normal Z in S = 1 - sigmoid(Z), only used when timestep_sample_method is "
                "moss. Default -1.09: MOSS-Image technical report, Table 6 (final stage) / Eq. 4."
            ),
            "help": (
                "Only used with timestep_sample_method = moss. Raising it moves the sampled sigmas toward less "
                "noise, lowering it toward more; the report's final stage uses -1.09 (median sigma ~0.78)."
            ),
        },
        {
            "path": "model.timestep_std",
            "label": "Sigma sampler std (s)",
            "type": "number",
            "default": 0.95,
            "gt": 0,
            "description": (
                "Standard deviation of Z, only used when timestep_sample_method is moss. Default 0.95: "
                "MOSS-Image technical report, Table 6 (final stage) / Eq. 4."
            ),
            "help": (
                "Only used with timestep_sample_method = moss. Larger values spread the sampled sigmas toward "
                "both extremes; the report's final stage uses 0.95."
            ),
        },
        {
            "path": "model.timestep_cutoff",
            "label": "Sigma sampler cutoff (a)",
            "type": "number",
            "default": 0.95,
            "gt": 0,
            "max": 0.999,
            "description": (
                "Cutoff a in (0, 1): S is truncated at a and rescaled onto the whole sigma grid, only used when "
                "timestep_sample_method is moss. Default 0.95: MOSS-Image technical report, Table 6 / Eq. 4."
            ),
            "help": (
                "Only used with timestep_sample_method = moss. S = 1 - sigmoid(Z) is truncated at this value "
                "and divided by it, so the grid's top is reached by the noisiest tail; the report's final "
                "stage uses 0.95."
            ),
        },
        {
            "path": "model.timestep_uniform_mix",
            "label": "Sigma sampler uniform mix (rho)",
            "type": "number",
            "default": 0.01,
            "min": 0,
            "max": 1,
            "description": (
                "Share of the sampler drawn uniformly over the 1000 grid points, only used when "
                "timestep_sample_method is moss. Default 0.01: MOSS-Image technical report, Table 6 / Eq. 4."
            ),
            "help": (
                "Only used with timestep_sample_method = moss. The pmf is (1 - rho) times the truncated "
                "logit-normal plus rho uniform; the report's final stage uses 0.01 (1%)."
            ),
        },
        field_library.sigmoid_scale(
            (
                "Scale applied to the logit-normal sample before sigmoid, only used when "
                "timestep_sample_method is logit_normal."
            ),
            help=(
                "Only used with timestep_sample_method = logit_normal. Raising it pushes sampled timesteps "
                "toward the extremes (near 0 or 1)."
            ),
        ),
        field_library.shift(
            (
                "Fixed timestep shift (> 0) applied on top of the sampled sigmas. Empty (default) trains with "
                "no shift, as the report's final stage; the 5.0 of the sampler is for previews only."
            ),
            placeholder="empty = no shift (reference)",
            help=(
                "Advanced: leave empty. The report trains its final stage at shift 1 (none); the shift 5.0 the "
                "reference sampler uses applies to previews only and is fixed."
            ),
        ),
    ],
)
