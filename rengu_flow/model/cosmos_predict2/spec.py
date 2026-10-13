"""Cosmos Predict2 (alias ``anima``) model descriptor (torch-free; registered via
``registry/builtin_models.py``)."""

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

_QUANT = QuantSpec(fp8="scaled_matmul", nf4=True, grad_mode=False, fp8_dtypes=("e5m2", "e4m3"))
_QUANT_FIELDS = field_library.quant_fields(
    _QUANT,
    nf4={
        "description": (
            "Stores the frozen DiT linears as 4-bit NF4 (~1/4 the VRAM of bf16); the adapter "
            "trains on top at full precision. Turn on when the base model does not fit — pair with "
            "adapter type LoKr (LyCORIS kinds refuse a quantized base)."
        ),
        "help": (
            "Adapter training only — the quantized base stays frozen while the adapter trains in "
            "full precision on top. Roughly quarters the DiT's VRAM. Mutually exclusive with "
            "model.transformer_fp8_matmul. Use adapter type LoKr: it is quantization-aware, while "
            "the LyCORIS kinds reject a quantized base."
        ),
    },
    fp8={
        "description": (
            "fp8 scaled-matmul quantization of the frozen DiT's big linears (mutually exclusive "
            "with transformer_4bit). Measured ~70% SLOWER on Ada (RTX 4080) and fp8-sensitive on "
            "Cosmos — an experimental/compat lever; transformer_4bit is the recommended "
            "VRAM-saving option."
        ),
    },
    fp8_dtype={
        "description": (
            "Weight fp8 format for transformer_fp8_matmul. On Ada (RTX 4080), torch._scaled_mm "
            "rejects e5m2 weights (e4m3 is required there), but Cosmos is fp8-sensitive to e4m3 "
            "outliers — measure before relying on this."
        ),
    },
)

# Cosmos exposes the full LyCORIS catalog. Two carry runtime constraints (not
# exclusions): DyLoRA needs activation_checkpointing = false (its random
# sub-rank per forward breaks checkpoint recompute) and the OFT family's
# staged weight rebuild is VRAM-hungry — pair with blocks_to_swap on small
# cards. See docs/user/training-cosmos-predict2-lora-lokr-finetune.md.
SPEC = ModelCapability(
    type_id="cosmos_predict2",
    display_name="Cosmos Predict2",
    aliases=["anima"],
    adapters=["lora", "lokr", *LYCORIS_ADAPTER_TYPES],
    full_finetune=True,
    # No Conv modules and only non-affine LayerNorm / custom RMSNorm in the adapted blocks
    # (the LLM adapter is never a LyCORIS target): train_conv/use_tucker/train_norm are no-ops.
    features={"block_swap": True, "video": True, LINEAR_ONLY_ADAPTERS: True},
    branding_note=(
        "Use checkpoints released for Cosmos Predict2 / Anima-style bundles "
        "(main, VAE, and Qwen3 text encoder paths below)."
    ),
    # The encoder is whichever of t5_path / llm_path is configured (t5_path wins): swapping
    # it, or replacing the file, must re-encode instead of reusing stale embeddings.
    text_cache_files=["t5_path", "llm_path"],
    # Explicit (not derived from components/bundle_key): the text encoder is Qwen3 or T5.
    model_validation={
        "one_of": [["llm_path", "t5_path"]],
    },
    adapter_layer_groups=["self_attention", "cross_attention", "mlp"],
    # Every top-level child of MiniTrainDIT (as the other DiT specs list theirs): preflight rejects
    # a target_include/exclude whose first segment is not one. llm_adapter is present on Anima
    # (Qwen3 LLM adapter) checkpoints and holds TransformerBlocks the default LoRA/LoKr scope
    # adapts (trained when llm_adapter_lr > 0). extra_pos_embedder is absent: the releases set
    # extra_per_block_abs_pos_emb = False (config.get_dit_config).
    adapter_module_roots=[
        "t_embedder",
        "llm_adapter",
        "blocks",
        "final_layer",
        "t_embedding_norm",
        "x_embedder",
        "pos_embedder",
    ],
    pipeline="rengu_flow.model.cosmos_predict2:CosmosPredict2Pipeline",
    env_prefix="COSMOS",
    components=(
        ComponentPath("transformer_path", expect="file"),
        ComponentPath("vae_path", expect="file"),
        ComponentPath("llm_path", expect="any"),
        ComponentPath("t5_path", expect="file", env=False),
        ComponentPath("llm_adapter_path", expect="file", env=False),
    ),
    text_cache="optional",
    # Scaled-matmul fp8 with a weight-format knob (e5m2 default; sm89 needs e4m3).
    quant=_QUANT,
    preview_spec=PreviewSpec(
        runner="dit",
        steps=20,
        guidance=4.0,
        memory_knobs=frozenset({"offload_text_encoder", "blocks_to_swap", "offload_dit_for_decode"}),
    ),
    diffusion_dtype_sets_load_dtype=True,
    install_profiles=("cosmos",),
    user_doc="docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    export_prefix="diffusion_model.",
    testkit="rengu_flow.model.cosmos_predict2.testkit",
    model_fields=[
        field_library.component_path(
            "transformer_path",
            "Main model (.safetensors)",
            "Large image/diffusion checkpoint (e.g. Anima preview).",
            placeholder="path/to/anima-preview.safetensors",
            required=True,
            help=(
                "Example: Anima preview or another Cosmos Predict2 release as a single "
                ".safetensors on disk. This file is the core diffusion model. Do not point here at "
                "the VAE or text encoder — those have their own fields below."
            ),
        ),
        field_library.component_path(
            "vae_path",
            "Image VAE (.safetensors)",
            "Encodes images to latents for training.",
            placeholder="path/to/qwen_image_vae.safetensors",
            required=True,
            help=(
                "Usually the Qwen image VAE paired with Cosmos/Anima (e.g. "
                "qwen_image_vae.safetensors). Training reads your images through this file to "
                "build latent caches."
            ),
        ),
        field_library.component_path(
            "llm_path",
            "Text encoder — Qwen3",
            "Turns captions into conditioning (file or folder).",
            placeholder="path/to/qwen_3_06b_base.safetensors",
            required=True,
            help=(
                "Converts dataset captions into conditioning vectors. Required for modern "
                "Anima/Cosmos setups that use Qwen3. Set llm_adapter_lr = 0 unless you "
                "intentionally train the small LLM adapter."
            ),
        ),
        field_library.component_path(
            "llm_adapter_path",
            "Text adapter (.safetensors)",
            "Extra weights on top of the text encoder, if you have them.",
            show_if_set=True,
            help=(
                "Leave empty unless your checkpoint pack includes a separate LLM adapter file to "
                "load."
            ),
        ),
        {
            "path": "model.llm_adapter_lr",
            "label": "LLM adapter LR",
            "type": "number",
            "default": 0,
            "visibility": {
                "all": [
                    {
                        "not": {
                            "field": "_has_adapter",
                            "equals": True,
                        },
                    },
                    {
                        "any": [
                            {
                                "form_nonempty": "model.llm_path",
                            },
                            {
                                "form_nonempty": "model.llm_adapter_path",
                            },
                            {
                                "form_nonempty": "model.llm_adapter_lr",
                                "exclude_zero": True,
                            },
                        ],
                    },
                ],
            },
            "description": (
                "Finetune only. The embedded Qwen3 LLM adapter is frozen by default (0); set a "
                "value to train it."
            ),
            "help": (
                "Finetune-only LR for the embedded Qwen3 LLM adapter; frozen by default (0), set a "
                "value to train it."
            ),
        },
        _QUANT_FIELDS["nf4"],
        field_library.transformer_dtype(
            help=(
                "Defaults to model.dtype. Applies to most weights in transformer_path; VAE, text "
                "encoder, and embedder/norm layers still use model.dtype. Leave empty unless you "
                "need a different load precision (VRAM / load errors)."
            ),
        ),
        field_library.diffusion_model_dtype(
            (
                "Autocast dtype for DiT forward; defaults to model.dtype. Sets transformer_dtype "
                "when omitted."
            ),
            label="DiT forward dtype",
            help=(
                "Sets the training forward autocast dtype (main.py). Defaults to model.dtype. On "
                "Cosmos, when set and transformer_dtype is omitted, it is also copied to "
                "transformer_dtype for the checkpoint load."
            ),
        ),
        field_library.component_path(
            "t5_path",
            "Text encoder — T5 (alternative)",
            "Alternative text encoder — a raw T5 checkpoint used instead of llm_path.",
            show_if_set=True,
            help=(
                "Single .safetensors or compatible layout. Pick either llm_path (Qwen3) or "
                "t5_path, not both, unless you know your checkpoint bundle requires a specific "
                "combination."
            ),
        ),
        {
            "path": "model.self_attn_lr",
            "label": "Self-attention LR override",
            "type": "number",
            "placeholder": "empty = optimizer.lr",
            "description": (
                "Per-group LR override for self-attention DiT parameters (adapter or finetune — "
                "both route through this grouping); empty uses optimizer.lr, 0 freezes the group."
            ),
        },
        {
            "path": "model.cross_attn_lr",
            "label": "Cross-attention LR override",
            "type": "number",
            "placeholder": "empty = optimizer.lr",
            "description": (
                "Per-group LR override for cross-attention DiT parameters (adapter or finetune); "
                "empty uses optimizer.lr, 0 freezes the group."
            ),
        },
        {
            "path": "model.mlp_lr",
            "label": "MLP LR override",
            "type": "number",
            "placeholder": "empty = optimizer.lr",
            "description": (
                "Per-group LR override for MLP/feed-forward DiT parameters (adapter or finetune); "
                "empty uses optimizer.lr, 0 freezes the group."
            ),
        },
        {
            "path": "model.mod_lr",
            "label": "AdaLN modulation LR override",
            "type": "number",
            "placeholder": "empty = optimizer.lr",
            "description": (
                "Per-group LR override for the adaLN-modulation DiT parameters (adapter or "
                "finetune); empty uses optimizer.lr, 0 freezes the group."
            ),
        },
        field_library.shift(
            (
                "Fixed timestep-shift transform t' = (t*shift)/(1+(shift-1)*t) (> 0) applied after "
                "sampling; overrides flux_shift when set. Empty leaves timesteps unshifted unless "
                "flux_shift is enabled."
            ),
            placeholder="empty = no shift (unless flux_shift is on)",
        ),
        {
            "path": "model.flux_shift",
            "label": "Flux-style resolution shift",
            "type": "boolean",
            "default": False,
            "description": (
                "Flux-style resolution-aware timestep shift (shifts sampling toward higher noise "
                "at larger latent sizes); ignored whenever model.shift is set."
            ),
        },
        field_library.sigmoid_scale(
            (
                "Scale applied to the logit-normal sample before sigmoid, only used when "
                "timestep_sample_method is logit_normal. Raising it pushes sampled timesteps "
                "toward the extremes (near 0 or 1)."
            ),
        ),
        field_library.timestep_sample_method(
            (
                "How training timesteps are sampled per step: logit_normal (default, concentrates "
                "around mid-range noise levels) or uniform (uniform in [0, 1])."
            ),
        ),
        _QUANT_FIELDS["fp8"],
        _QUANT_FIELDS["fp8_dtype"],
    ],
)
