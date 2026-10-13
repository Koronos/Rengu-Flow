"""Krea 2 model descriptor (torch-free; registered via ``registry/builtin_models.py``)."""

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
            "Stores the frozen DiT block linears as 4-bit NF4 (~7 GB instead of ~26 GB bf16); the "
            "adapter trains on top at full precision. A 16 GB card needs 4-bit or fp8 "
            "(transformer_fp8_matmul) — pair with adapter type LoKr (LyCORIS kinds refuse a "
            "quantized base)."
        ),
        "help": (
            "Adapter training only — the quantized base stays frozen while the adapter trains in "
            "full precision on top. Krea 2's DiT drops from ~26 GB bf16 to ~7 GB (text fusion "
            "stays bf16). A 16 GB card needs 4-bit or fp8 (model.transformer_fp8_matmul, the "
            "faster and more accurate one on RTX 40xx); they are mutually exclusive. Use adapter "
            "type LoKr: it is quantization-aware, while the LyCORIS kinds reject a quantized base."
        ),
    },
    fp8={
        "description": (
            "Tensorwise-scaled e4m3 quantization of the frozen DiT's big linears (mutually "
            "exclusive with transformer_4bit): 1 byte/param drops the base 25.6 -> 12.9 GB, and "
            "the cuBLASLt kernel hits ~2x bf16 GEMM throughput under compile_scope = \"block\". "
            "Quant error is 2.65% RMS vs NF4's 9.55%; pair with model.fp8_grad_mode for the "
            "backward precision."
        ),
        "help": (
            "Adapter training only: the DiT drops from ~25.6 to ~12.9 GB (2.65% RMS weight error "
            "vs NF4's 9.55%), and with compile = true + compile_scope = \"block\" the matmuls run "
            "~2x faster on RTX 40xx (sm89+). Mutually exclusive with transformer_4bit; use adapter "
            "type LoKr."
        ),
    },
    grad_mode={
        "description": (
            "Input-gradient GEMM precision through the frozen fp8 base, only used when "
            "transformer_fp8_matmul is on: fp8 measures ~15-20% faster per step, bf16 (default) "
            "keeps the clean gradient."
        ),
        "help": (
            "'bf16' (default) keeps the backward's input-gradient GEMM unquantized for a clean "
            "gradient; 'fp8' runs it through the same e4m3 tensorwise kernel as the forward, "
            "measured ~15-20% faster per step. Only takes effect when model.transformer_fp8_matmul "
            "is true."
        ),
    },
)

# Krea 2 (open-weights 12B DiT). Same LyCORIS caveats as cosmos apply (DyLoRA vs
# activation checkpointing, OFT VRAM). At 12B the DiT does not fit consumer VRAM in
# bf16 — pair adapters with model.transformer_4bit/fp8 and blocks_to_swap.
SPEC = ModelCapability(
    type_id="krea2",
    display_name="Krea 2",
    adapters=["lora", "lokr", *LYCORIS_ADAPTER_TYPES],
    full_finetune=True,
    features={"block_swap": True, "tread": True, LINEAR_ONLY_ADAPTERS: True},
    transformer_blocks=28,
    # ponytail: only the knob users change on purpose; keying on encoder/checkpoint
    # paths would re-encode every cache on a file move. Add them if encoders diverge.
    text_cache_keys={"max_sequence_length": 512},
    branding_note=(
        "Use the Krea 2 Raw open-weights files (raw.safetensors or ComfyUI's "
        "krea2_raw_bf16.safetensors, plus the Qwen3-VL text encoder and Qwen-Image "
        "VAE files). Train on Raw; the distilled Turbo checkpoint is for inference."
    ),
    model_validation={
        "one_of": [
            ["transformer_path", "checkpoint_path"],
            ["vae_path", "checkpoint_path"],
            ["text_encoder_path", "checkpoint_path"],
        ],
    },
    adapter_layer_groups=[
        "text_fusion",
        "attention",
        "feedforward",
        "time_modulation",
        "image_in_out",
    ],
    adapter_module_roots=[
        "transformer_blocks",
        "text_fusion",
        "txt_in",
        "img_in",
        "time_embed",
        "time_mod_proj",
        "rotary_emb",
        "final_layer",
    ],
    pipeline="rengu_flow.model.krea2:Krea2Pipeline",
    env_prefix="KREA2",
    # Each component: a single file or its diffusers subfolder; checkpoint_path = the whole
    # Krea-2-Raw diffusers folder (fills any component left empty).
    components=(
        ComponentPath("transformer_path", expect="any"),
        ComponentPath("vae_path", expect="any"),
        ComponentPath("text_encoder_path", expect="any"),
        ComponentPath("checkpoint_path", expect="dir"),
    ),
    bundle_key="checkpoint_path",
    # The tapped Qwen3-VL hidden-state stack cannot run inside the training graph.
    text_cache="required",
    # Always tensorwise e4m3 (sm89's _scaled_mm rejects e5m2): no fp8_matmul_dtype knob.
    quant=_QUANT,
    # Reference settings: 28 steps, CFG 4.5 (cond + g*(cond - uncond)).
    preview_spec=PreviewSpec(runner="dit", steps=28, guidance=4.5),
    install_profiles=("cosmos",),
    user_doc="docs/user/training-krea2.md",
    export_prefix="transformer.",
    testkit="rengu_flow.model.krea2.testkit",
    model_fields=[
        # The component paths are not "required": each is one_of(<component>_path,
        # checkpoint_path) — the diffusers folder below fills any left empty.
        field_library.component_path(
            "transformer_path",
            "Main model (.safetensors)",
            (
                "The big DiT checkpoint you train (official raw.safetensors or ComfyUI file). A "
                "diffusers transformer folder also works. Optional when the diffusers folder below "
                "is set."
            ),
            placeholder="path/to/krea2_raw_bf16.safetensors",
            importance="recommended",
            help=(
                "The official raw.safetensors or ComfyUI's krea2_raw_bf16.safetensors — original "
                "key layout is converted automatically. A diffusers transformer folder also works. "
                "Pre-quantized fp8/nvfp4 'scaled' files cannot be trained; use the bf16 file."
            ),
        ),
        field_library.component_path(
            "vae_path",
            "Image VAE (.safetensors)",
            (
                "Qwen-Image VAE — the same file Cosmos/Anima setups use. Encodes images to latents "
                "for training. Optional when the diffusers folder below is set."
            ),
            placeholder="path/to/qwen_image_vae.safetensors",
            importance="recommended",
            help=(
                "One .safetensors (ComfyUI layout, converted automatically) or a diffusers VAE "
                "folder. Training reads your images through this file to build latent caches."
            ),
        ),
        field_library.component_path(
            "text_encoder_path",
            "Text encoder — Qwen3-VL (.safetensors)",
            (
                "Turns captions into conditioning (ComfyUI file or transformers folder). Tokenizer "
                "is bundled. Optional when the diffusers folder below is set."
            ),
            placeholder="path/to/qwen3vl_4b_bf16.safetensors",
            path_expect="any",
            importance="recommended",
            help=(
                "ComfyUI's qwen3vl_4b_bf16.safetensors works as-is (vision weights are ignored; "
                "the tokenizer is bundled with rengu). Captions are encoded once into the cache; "
                "the encoder never runs during training steps."
            ),
        ),
        field_library.bundle_folder(
            "checkpoint_path",
            (
                "Full Krea-2-Raw diffusers folder (transformer/, vae/, text_encoder/); fills any "
                "of the three component paths left empty. Set either this or the three files above."
            ),
            help=(
                "Optional. If you downloaded the whole krea/Krea-2-Raw folder (transformer/, vae/, "
                "text_encoder/, tokenizer/ subfolders), point here and leave the per-component "
                "fields empty — each one resolves as <checkpoint_path>/<subfolder>. Files you set "
                "explicitly always win. Never a repo id; nothing is downloaded."
            ),
        ),
        field_library.component_path(
            "tokenizer_path",
            "Tokenizer override",
            "Folder with tokenizer files; defaults to the bundled Qwen3-VL tokenizer.",
            show_if_set=True,
            help=(
                "rengu ships the Qwen3-VL tokenizer, so leave this empty. Changing it re-encodes "
                "the text-embedding cache."
            ),
        ),
        {
            "path": "model.max_sequence_length",
            "label": "Max caption tokens",
            "type": "integer",
            "default": 512,
            "min": 1,
            "description": (
                "Prompt token budget before truncation (default 512). Lower it to shrink the "
                "text-embedding cache; captions longer than this lose their tail."
            ),
            "help": (
                "Also caps the text-embedding cache entry size — Krea 2 caches a 12-layer "
                "hidden-state stack per token, so halving this halves the cache for long captions. "
                "Compaction already trims unused padding, so lowering it only matters when "
                "captions actually reach the limit."
            ),
        },
        _QUANT_FIELDS["nf4"],
        field_library.transformer_dtype(
            help=(
                "The VAE and the Qwen3-VL text encoder keep model.dtype; adapters default to "
                "float32 (adapter.dtype). Leave empty unless you need a different DiT load "
                "precision. For VRAM, use transformer_4bit / transformer_fp8_matmul instead."
            ),
        ),
        # TOML-only for krea2: it only changes the global training autocast dtype
        # (main.py) — overlaps model.dtype / transformer_dtype, never sets the load dtype.
        field_library.diffusion_model_dtype(
            (
                "Training autocast dtype override (defaults to model.dtype). Does not change the "
                "DiT load dtype — that is transformer_dtype."
            ),
            label="DiT forward dtype",
            ui=False,
            help=(
                "Sets the training forward autocast dtype (main.py). Defaults to model.dtype. On "
                "Cosmos, when set and transformer_dtype is omitted, it is also copied to "
                "transformer_dtype for the checkpoint load."
            ),
            doc="docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
        ),
        field_library.shift(
            (
                "Advanced — leave empty. Fixed timestep shift (> 0), overriding the reference Krea "
                "2 resolution-aware dynamic shift (mu 0.5 at 256 latent tokens to 1.15 at 6400, "
                "i.e. an equivalent fixed shift of exp(mu) ~1.65 to ~3.16)."
            ),
            placeholder="empty = resolution-aware dynamic shift",
            help=(
                "Empty uses the reference Krea 2 schedule: mu from 0.5 at 256 latent tokens to "
                "1.15 at 6400 (1024x1024 = 4096 tokens), equal to a fixed shift of exp(mu) ~1.65 "
                "to ~3.16. A number (> 0) replaces it at every resolution — only for a deliberate "
                "schedule A/B."
            ),
        ),
        field_library.sigmoid_scale(
            (
                "Advanced — the default 1.0 is the reference. Scale applied to the logit-normal "
                "sample before sigmoid (logit_normal only); raising it pushes sampled timesteps "
                "toward the extremes (near 0 or 1)."
            ),
            help=(
                "Only used with timestep_sample_method = logit_normal. Raising it pushes sampled "
                "timesteps toward the extremes (near 0 or 1)."
            ),
        ),
        field_library.timestep_sample_method(
            (
                "Advanced — logit_normal (default) is the reference Krea 2 training distribution "
                "(mid-range noise levels); uniform samples t evenly in [0, 1]."
            ),
            help=(
                "logit_normal concentrates steps on mid-range noise levels, like the reference "
                "Krea 2 training; uniform samples t evenly in [0, 1]. The resolution-aware shift "
                "(or model.shift) is applied on top of either."
            ),
        ),
        _QUANT_FIELDS["fp8"],
        _QUANT_FIELDS["grad_mode"],
        # model.fp8_matmul_dtype is intentionally NOT exposed for krea2: its fp8
        # path always quantizes tensorwise e4m3 (sm89's _scaled_mm rejects e5m2
        # weights), so the knob would be a no-op here. Cosmos still exposes it.
    ],
)
