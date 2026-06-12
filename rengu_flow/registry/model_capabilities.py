"""Per-model UI/training capabilities for config forms and validation hints.

Canonical model types register here alongside @register_model. Optional aliases map legacy
type strings to a canonical id for the UI and validator (see ``aliases`` on each capability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES

# Human-readable labels for adapter kinds shown in the UI.
DEFAULT_ADAPTER_LABELS: dict[str, str] = {
    "lora": "LoRA (PEFT)",
    "lokr": "LoKr",
    "lycoris_locon": "LyCORIS · LoCon",
    "lycoris_loha": "LyCORIS · LoHa",
    "lycoris_lokr": "LyCORIS · LoKr",
    "lycoris_dora": "LyCORIS · DoRA",
    "lycoris_dylora": "LyCORIS · DyLoRA",
    "lycoris_glora": "LyCORIS · GLoRA",
    "lycoris_diag_oft": "LyCORIS · Diag-OFT",
    "lycoris_boft": "LyCORIS · BOFT",
}

# Field groups for LyCORIS adapters: (field_spec, frozenset_of_adapter_kinds).
# The schema builder iterates these and filters by the active adapter kind.
_ALL_LYCORIS = frozenset(LYCORIS_ADAPTER_TYPES)
_LYCORIS_WITH_TRAIN_CONV = _ALL_LYCORIS - frozenset({"lycoris_dylora"})
_LOCON_LOHA = frozenset({"lycoris_locon", "lycoris_loha"})
_LOCON_LOHA_DORA_LOKR = frozenset({"lycoris_locon", "lycoris_loha", "lycoris_dora", "lycoris_lokr"})
_OFT_FAMILY = frozenset({"lycoris_diag_oft", "lycoris_boft"})

LYCORIS_FIELD_GROUPS: list[tuple[dict[str, Any], frozenset[str]]] = [
    # Shared by all 8 lycoris kinds
    ({"path": "adapter.dropout", "label": "Dropout", "type": "number", "default": 0.0}, _ALL_LYCORIS),
    ({"path": "adapter.rank_dropout", "label": "Rank dropout", "type": "number", "default": 0.0}, _ALL_LYCORIS),
    ({"path": "adapter.module_dropout", "label": "Module dropout", "type": "number", "default": 0.0}, _ALL_LYCORIS),
    # Shared by all except lycoris_dylora
    ({"path": "adapter.train_conv", "label": "Train conv layers", "type": "boolean", "default": False}, _LYCORIS_WITH_TRAIN_CONV),
    # locon, loha, dora, lokr — tucker/scalar/wd_on_output
    ({"path": "adapter.use_tucker", "label": "Tucker decomposition", "type": "boolean", "default": False}, _LOCON_LOHA_DORA_LOKR),
    ({"path": "adapter.use_scalar", "label": "Trained scalar", "type": "boolean", "default": False}, _LOCON_LOHA_DORA_LOKR),
    ({"path": "adapter.wd_on_output", "label": "DoRA output axis", "type": "boolean", "default": True}, _LOCON_LOHA_DORA_LOKR),
    # dora_wd: locon, loha, lokr (NOT dora — implied by the type)
    ({"path": "adapter.dora_wd", "label": "DoRA decomposition", "type": "boolean", "default": False}, _LOCON_LOHA | frozenset({"lycoris_lokr"})),
    # lokr-only extras
    ({"path": "adapter.factor", "label": "LoKr factor", "type": "integer", "default": -1}, frozenset({"lycoris_lokr"})),
    ({"path": "adapter.full_matrix", "label": "LoKr full_matrix", "type": "boolean", "default": False}, frozenset({"lycoris_lokr"})),
    ({"path": "adapter.decompose_both", "label": "LoKr decompose_both", "type": "boolean", "default": False}, frozenset({"lycoris_lokr"})),
    ({"path": "adapter.unbalanced_factorization", "label": "Unbalanced factorization", "type": "boolean", "default": False}, frozenset({"lycoris_lokr"})),
    # dylora-only
    ({"path": "adapter.block_size", "label": "Block size", "type": "integer", "default": 4}, frozenset({"lycoris_dylora"})),
    # OFT family (diag_oft, boft)
    ({"path": "adapter.constraint", "label": "OFT constraint", "type": "number", "default": 0.0}, _OFT_FAMILY),
    ({"path": "adapter.rescaled", "label": "Rescaled OFT", "type": "boolean", "default": False}, _OFT_FAMILY),
]


def _expand_lycoris_templates(target: dict[str, list[dict[str, Any]]]) -> None:
    """Populate ADAPTER_FIELD_TEMPLATES for each lycoris kind from LYCORIS_FIELD_GROUPS."""
    for spec, kinds in LYCORIS_FIELD_GROUPS:
        for kind in kinds:
            target.setdefault(kind, []).append(spec)


# Shared adapter field templates (network types implemented under rengu_flow.networks)
ADAPTER_FIELD_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "common": [
        {
            "path": "adapter.rank",
            "label": "Rank",
            "type": "integer",
            "default": 16,
            "min": 1,
        },
        {"path": "adapter.dim", "label": "Dim (alias for rank)", "type": "integer", "min": 1},
        {
            "path": "adapter.init_from_existing",
            "label": "Init from existing adapter path",
            "type": "path",
        },
        {"path": "adapter.dtype", "label": "Adapter dtype", "type": "select", "options_key": "dtypes"},
    ],
    "lora": [
        # Shared by path with the lycoris kinds: the schema builder merges equal
        # paths into one field shown for every kind that declares it.
        {"path": "adapter.dropout", "label": "Dropout", "type": "number", "default": 0.0},
    ],
    "lokr": [
        {"path": "adapter.factor", "label": "LoKr factor", "type": "integer", "default": -1},
        {"path": "adapter.decompose_both", "label": "LoKr decompose_both", "type": "boolean", "default": False},
        {"path": "adapter.full_matrix", "label": "LoKr full_matrix", "type": "boolean", "default": False},
    ],
}

_expand_lycoris_templates(ADAPTER_FIELD_TEMPLATES)

model_capability_registry: dict[str, ModelCapability] = {}


@dataclass
class ModelCapability:
    """Training options and config fields for one canonical model type."""

    type_id: str
    display_name: str
    adapters: list[str] = field(default_factory=list)
    full_finetune: bool = True
    preview: bool = False
    aliases: list[str] = field(default_factory=list)
    branding_note: str = ""
    model_fields: list[dict[str, Any]] = field(default_factory=list)
    # Feature flags drive cross-section visibility (block_swap, preview, …) via field_visibility.
    features: dict[str, bool] = field(default_factory=dict)
    # Optional validation overrides — see model_config_rules.py (one_of, by_raw_type, …).
    model_validation: dict[str, Any] = field(default_factory=dict)
    # Per-kind display labels; falls back to DEFAULT_ADAPTER_LABELS then the raw kind string.
    adapter_labels: dict[str, str] = field(default_factory=dict)

    def training_modes(self) -> list[str]:
        modes: list[str] = []
        if self.full_finetune:
            modes.append("full_finetune")
        modes.extend(self.adapters)
        return modes

    def to_dict(self) -> dict[str, Any]:
        resolved_labels: dict[str, str] = {
            kind: self.adapter_labels.get(kind, DEFAULT_ADAPTER_LABELS.get(kind, kind))
            for kind in self.adapters
        }
        return {
            "type_id": self.type_id,
            "display_name": self.display_name,
            "adapters": list(self.adapters),
            "adapter_labels": resolved_labels,
            "full_finetune": self.full_finetune,
            "preview": self.preview,
            "aliases": list(self.aliases),
            "branding_note": self.branding_note,
            "training_modes": self.training_modes(),
            "model_fields": list(self.model_fields),
            "features": dict(self.features),
            "model_validation": dict(self.model_validation),
        }


def register_model_capability(cap: ModelCapability) -> ModelCapability:
    model_capability_registry[cap.type_id.lower()] = cap
    return cap


def normalize_model_type(model_type: str | None) -> str | None:
    """Map a capability alias to its canonical type_id, if registered."""
    if not model_type:
        return model_type
    key = str(model_type).lower()
    if key in model_capability_registry:
        return key
    for cap in model_capability_registry.values():
        if key in [a.lower() for a in cap.aliases]:
            return cap.type_id
    return key


def ensure_default_capability(type_id: str) -> None:
    """Register a minimal UI capability when @register_model runs (no-op if already defined)."""
    key = type_id.lower()
    if key in model_capability_registry:
        return
    register_model_capability(
        ModelCapability(
            type_id=key,
            display_name=key.replace("_", " ").title(),
            adapters=["lora", "lokr"],
            full_finetune=True,
            preview=False,
        )
    )


def get_canonical_model_types() -> list[str]:
    return sorted(model_capability_registry.keys())


def get_alias_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for cap in model_capability_registry.values():
        for alias in cap.aliases:
            out[alias.lower()] = cap.type_id
    return out


def get_capability(model_type: str | None) -> ModelCapability | None:
    canonical = normalize_model_type(model_type)
    if not canonical:
        return None
    return model_capability_registry.get(canonical)


def capabilities_for_api() -> dict[str, dict[str, Any]]:
    return {k: v.to_dict() for k, v in model_capability_registry.items()}


def _register_builtin_capabilities() -> None:
    register_model_capability(
        ModelCapability(
            type_id="sdxl",
            display_name="SDXL",
            adapters=["lora", "lokr", *LYCORIS_ADAPTER_TYPES],
            full_finetune=True,
            preview=True,
            features={"preview": True, "block_swap": True},
            model_fields=[
                {
                    "path": "model.checkpoint_path",
                    "label": "Base model (.safetensors)",
                    "type": "path",
                    "required": True,
                    "placeholder": "path/to/sdxl.safetensors",
                    "description": "Full SDXL weights file — not a LoRA.",
                },
                {
                    "path": "model.guidance",
                    "label": "Guidance",
                    "type": "number",
                    "default": 1.0,
                    "ui": False,
                    "description": "Parsed in TOML only; training uses preview.guidance_scale for CFG.",
                },
                {
                    "path": "model.freeze_text_encoders",
                    "label": "Freeze text encoders",
                    "type": "boolean",
                    "description": "UNet-only style training when omitted adapter.",
                },
            ],
        )
    )
    register_model_capability(
        ModelCapability(
            type_id="cosmos_predict2",
            display_name="Cosmos Predict2",
            aliases=["anima"],
            adapters=["lora", "lokr"],
            full_finetune=True,
            preview=True,
            features={"preview": True, "block_swap": True},
            branding_note=(
                "Use checkpoints released for Cosmos Predict2 / Anima-style bundles "
                "(main, VAE, and Qwen3 text encoder paths below)."
            ),
            model_validation={
                "one_of": [["llm_path", "t5_path"]],
            },
            model_fields=[
                {
                    "path": "model.transformer_path",
                    "label": "Main model (.safetensors)",
                    "type": "path",
                    "required": True,
                    "placeholder": "path/to/anima-preview.safetensors",
                    "description": "Large image/diffusion checkpoint (e.g. Anima preview).",
                },
                {
                    "path": "model.vae_path",
                    "label": "Image VAE (.safetensors)",
                    "type": "path",
                    "required": True,
                    "placeholder": "path/to/qwen_image_vae.safetensors",
                    "description": "Encodes images to latents for training.",
                },
                {
                    "path": "model.llm_path",
                    "label": "Text encoder — Qwen3",
                    "type": "path",
                    "required": True,
                    "placeholder": "path/to/qwen_3_06b_base.safetensors",
                    "description": "Turns captions into conditioning (file or folder).",
                },
                {
                    "path": "model.llm_adapter_path",
                    "label": "Text adapter (.safetensors)",
                    "type": "path",
                    "show_if_set": True,
                    "description": "Extra weights on top of the text encoder, if you have them.",
                },
                {
                    "path": "model.llm_adapter_lr",
                    "label": "LLM adapter LR",
                    "type": "number",
                    "default": 0,
                    "visibility": {
                        "all": [
                            {"not": {"field": "_has_adapter", "equals": True}},
                            {
                                "any": [
                                    {"form_nonempty": "model.llm_path"},
                                    {"form_nonempty": "model.llm_adapter_path"},
                                    {"form_nonempty": "model.llm_adapter_lr", "exclude_zero": True},
                                ],
                            },
                        ],
                    },
                    "description": "Finetune only. The embedded Qwen3 LLM adapter is frozen by default (0); set a value to train it.",
                },
                {
                    "path": "model.cache_text_embeddings",
                    "label": "Cache text embeddings",
                    "type": "boolean",
                    "default": True,
                },
                {
                    "path": "model.transformer_dtype",
                    "label": "Main model load dtype",
                    "type": "select",
                    "options_key": "dtypes",
                    "description": "DiT checkpoint load only; defaults to Model dtype. VAE/text unchanged.",
                },
                {
                    "path": "model.diffusion_model_dtype",
                    "label": "DiT forward dtype",
                    "type": "select",
                    "options_key": "dtypes",
                    "ui": False,
                    "description": "Autocast dtype for DiT forward; defaults to model.dtype. Sets transformer_dtype when omitted.",
                },
            ],
        )
    )


_register_builtin_capabilities()
