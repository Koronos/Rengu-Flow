"""Per-model UI/training capabilities for config forms and validation hints.

Canonical model types register here alongside @register_model. Optional aliases map legacy
type strings to a canonical id for the UI and validator (see ``aliases`` on each capability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Shared adapter field templates (network types implemented under renga_flow.networks)
ADAPTER_FIELD_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "common": [
        {
            "path": "adapter.rank",
            "label": "Rank",
            "type": "integer",
            "default": 16,
            "min": 1,
            "recommended": True,
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
        {"path": "adapter.dropout", "label": "LoRA dropout", "type": "number", "default": 0.0},
    ],
    "lokr": [
        {"path": "adapter.factor", "label": "LoKr factor", "type": "integer", "default": -1},
        {"path": "adapter.decompose_both", "label": "LoKr decompose_both", "type": "boolean", "default": False},
        {"path": "adapter.full_matrix", "label": "LoKr full_matrix", "type": "boolean", "default": False},
    ],
}

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

    def training_modes(self) -> list[str]:
        modes: list[str] = []
        if self.full_finetune:
            modes.append("full_finetune")
        modes.extend(self.adapters)
        return modes

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "display_name": self.display_name,
            "adapters": list(self.adapters),
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
            adapters=["lora", "lokr"],
            full_finetune=True,
            preview=True,
            features={"preview": True},
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
            features={"block_swap": True, "preview": True},
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
                    "label": "Text adapter (.safetensors, optional)",
                    "type": "path",
                    "show_if_set": True,
                    "description": "Extra weights on top of the text encoder, if you have them.",
                },
                {
                    "path": "model.llm_adapter_lr",
                    "label": "LLM adapter LR",
                    "type": "number",
                    "visibility": {
                        "any": [
                            {"form_nonempty": "model.llm_adapter_path"},
                            {"form_nonempty": "model.llm_adapter_lr", "exclude_zero": True},
                        ],
                    },
                    "description": "Set 0 to freeze LLM adapter.",
                },
                {
                    "path": "model.cache_text_embeddings",
                    "label": "Cache text embeddings",
                    "type": "boolean",
                    "default": True,
                },
                {
                    "path": "model.transformer_dtype",
                    "label": "Main model load dtype (optional)",
                    "type": "select",
                    "options_key": "dtypes",
                    "description": "DiT checkpoint load only; defaults to Model dtype. VAE/text unchanged.",
                },
                {
                    "path": "model.diffusion_model_dtype",
                    "label": "Forward dtype (optional, unused)",
                    "type": "select",
                    "options_key": "dtypes",
                    "description": "Not applied by training yet; leave unset.",
                },
            ],
        )
    )


_register_builtin_capabilities()
