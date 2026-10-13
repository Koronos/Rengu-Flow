"""Nanosaur2 model descriptor (torch-free; registered via ``registry/builtin_models.py``)."""

from __future__ import annotations

from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES
from rengu_flow.registry import field_library
from rengu_flow.registry.model_spec import (
    LINEAR_ONLY_ADAPTERS,
    ComponentPath,
    ModelCapability,
    PreviewSpec,
)

# Nanosaur2 (well9472/Nanosaur2-670M): 670M SPRINT DiT + Gemma3-270M + semantic DINOv2 VAE.
# Small enough to train in bf16 on 8 GB without quantization or block swap.
SPEC = ModelCapability(
    type_id="nanosaur2",
    display_name="Nanosaur2",
    adapters=["lora", "lokr", *LYCORIS_ADAPTER_TYPES],
    full_finetune=True,
    # Linear-only: no Conv modules and only RMSNorm / non-affine LayerNorm in the DiT.
    features={LINEAR_ONLY_ADAPTERS: True},
    transformer_blocks=18,
    # The encoder file (name, size, mtime) keys the text cache; captions re-encode if
    # it is swapped.
    text_cache_files=["text_encoder_path"],
    preview_options={"guidance_mode": ["alternate", "cfg", "path_drop"]},
    branding_note=(
        "Use the three files of well9472/Nanosaur2-670M (nanosaur2_diffusion_model, "
        "nanosaur2_text_encoder, nanosaur2_vae). Not the DMAD 4-step distilled model."
    ),
    adapter_layer_groups=["attention", "cross_attention", "feedforward", "text_refine"],
    adapter_module_roots=[
        "s_embedder",
        "t_embedder",
        "y_embedder",
        "shared_encoder_adaLN",
        "encoder_adaLN_offsets",
        "y_pool_proj",
        "sprint_out_proj",
        "blocks",
        "text_refine_blocks",
        "final_layer",
    ],
    pipeline="rengu_flow.model.nanosaur2:Nanosaur2Pipeline",
    env_prefix="NANOSAUR2",
    components=(
        ComponentPath("transformer_path", expect="file"),
        ComponentPath("vae_path", expect="file"),
        ComponentPath("text_encoder_path", expect="file"),
    ),
    # Captions are encoded once while caching; the frozen Gemma3 encoder never runs in the
    # training graph.
    text_cache="required",
    # Reference: Euler/simple, CFG 4, "alternate" guidance (CFG on even steps, path-drop on odd
    # ones), negative "oldest, low quality"; 50 steps trimmed to 30. The 670M DiT samples
    # resident (no preview block swap).
    preview_spec=PreviewSpec(
        runner="dit",
        steps=30,
        guidance=4.0,
        memory_knobs=frozenset({"offload_text_encoder"}),
        negative_prompt="oldest, low quality",
        defaults={"guidance_mode": "alternate"},
    ),
    # Reference SPRINT path-drop rates (train_lora.py): captioned / caption-dropped samples.
    model_defaults={"sparse_skip_prob": 0.05, "sparse_skip_prob_uncond": 0.5},
    install_profiles=("cosmos",),
    user_doc="docs/user/training-nanosaur2.md",
    export_prefix="diffusion_model.",
    testkit="rengu_flow.model.nanosaur2.testkit",
    # (L,) per-token prompt-emphasis weights.
    ragged_cache_keys=frozenset({"token_weights"}),
    model_fields=[
        field_library.component_path(
            "transformer_path",
            "Main model (.safetensors)",
            "The Nanosaur2 DiT you train (nanosaur2_diffusion_model.safetensors).",
            placeholder="path/to/nanosaur2_diffusion_model.safetensors",
            required=True,
            help=(
                "nanosaur2_diffusion_model.safetensors from well9472/Nanosaur2-670M (1.33 GB). Train "
                "on this base file, not nanosaur2_dmad_4step_diffusion_model.safetensors (a 4-step "
                "distilled checkpoint for inference only). The VAE and text encoder have their own "
                "fields below."
            ),
            doc="docs/user/training-nanosaur2.md",
        ),
        field_library.component_path(
            "vae_path",
            "Image VAE (.safetensors)",
            "Nanosaur2's semantic DINOv2 VAE (nanosaur2_vae.safetensors).",
            placeholder="path/to/nanosaur2_vae.safetensors",
            required=True,
            help=(
                "nanosaur2_vae.safetensors (0.26 GB): a DINOv2-based semantic VAE with 64 latent "
                "channels and 16x downsampling. It is not interchangeable with the Qwen/Wan image "
                "VAE. Training reads your images through it once to build the latent cache."
            ),
            doc="docs/user/training-nanosaur2.md",
        ),
        field_library.component_path(
            "text_encoder_path",
            "Text encoder — Gemma3-270M (.safetensors)",
            "nanosaur2_text_encoder.safetensors (Gemma3-270M; its tokenizer is inside the file).",
            placeholder="path/to/nanosaur2_text_encoder.safetensors",
            required=True,
            help=(
                "nanosaur2_text_encoder.safetensors (0.54 GB): the frozen Gemma3-270M with its "
                "SentencePiece tokenizer stored inside the file, so no tokenizer folder is needed. "
                "Captions are encoded once into the cache; the encoder never runs during training "
                "steps."
            ),
            doc="docs/user/training-nanosaur2.md",
        ),
        {
            "path": "model.sparse_skip_prob",
            "label": "Path-drop rate (captioned samples)",
            "type": "number",
            "default": 0.05,
            "min": 0,
            "max": 1,
            "description": (
                "Probability that a captioned sample skips the SPRINT sparse middle blocks, so the "
                "model keeps learning the path-drop pass its alternate/path_drop guidance uses. "
                "Reference 0.05."
            ),
        },
        {
            "path": "model.sparse_skip_prob_uncond",
            "label": "Path-drop rate (caption-dropped samples)",
            "type": "number",
            "default": 0.5,
            "min": 0,
            "max": 1,
            "description": (
                "Same, for samples whose caption was dropped (dataset uncond_fraction, 0.1 by "
                "default for this model). Reference 0.5."
            ),
        },
        field_library.transformer_dtype(
            help=(
                "Defaults to model.dtype. Applies to the DiT weights in transformer_path; the VAE "
                "and the adapters keep model.dtype (the text encoder runs in fp32). Leave empty "
                "unless you need a different load precision."
            ),
            doc="docs/user/training-nanosaur2.md",
        ),
        field_library.timestep_sample_method(
            (
                "logit_normal (default) is the reference recipe: sigmoid of a normal centered at "
                "ln 2 (noisier timesteps favored); uniform samples t evenly in [0, 1]."
            ),
        ),
        field_library.sigmoid_scale(
            "Std of the logit-normal before sigmoid (reference 1.0).",
        ),
        field_library.shift(
            "Advanced — leave empty. The reference recipe trains without a shift.",
            placeholder="empty = no shift (reference)",
        ),
    ],
)
