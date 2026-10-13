"""Model capability registry: the registered ``ModelCapability`` descriptors + adapter templates.

Each canonical model type is described by one spec module (``SPEC = ModelCapability(...)`` in
``rengu_flow/model/<m>/spec.py``; sdxl: ``rengu_flow/model/sdxl_spec.py``), listed in
``registry/builtin_models.py`` and registered at import time below. Optional aliases map legacy
type strings to a canonical id for the UI and validator (see ``aliases`` on each capability).
The descriptor types live in ``registry/model_spec.py`` (re-exported here). Torch-free.
"""

from __future__ import annotations

import importlib
from typing import Any

from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES
from rengu_flow.registry.model_spec import (  # noqa: F401 — re-exported API
    BASE_RAGGED_CACHE_KEYS,
    AdapterExport,
    DEFAULT_ADAPTER_LABELS,
    LINEAR_ONLY_ADAPTERS,
    ComponentPath,
    ModelCapability,
    PreviewSpec,
    QuantSpec,
    _path_identity,
)

# Field groups for LyCORIS adapters: (field_spec, frozenset_of_adapter_kinds).
# The schema builder iterates these and filters by the active adapter kind.
_ALL_LYCORIS = frozenset(LYCORIS_ADAPTER_TYPES)
_LYCORIS_WITH_TRAIN_CONV = _ALL_LYCORIS - frozenset({"lycoris_dylora"})
_LOCON_LOHA = frozenset({"lycoris_locon", "lycoris_loha"})
_LOCON_LOHA_LOKR = frozenset({"lycoris_locon", "lycoris_loha", "lycoris_lokr"})
_OFT_FAMILY = frozenset({"lycoris_diag_oft", "lycoris_boft"})

# LINEAR_ONLY_ADAPTERS (unless_capability on the conv/norm-only knobs) lives in model_spec.
LYCORIS_FIELD_GROUPS: list[tuple[dict[str, Any], frozenset[str]]] = [
    # Shared by all 7 lycoris kinds
    ({"path": "adapter.dropout", "label": "Dropout", "type": "number", "default": 0.0}, _ALL_LYCORIS),
    ({"path": "adapter.rank_dropout", "label": "Rank dropout", "type": "number", "default": 0.0}, _ALL_LYCORIS),
    ({"path": "adapter.module_dropout", "label": "Module dropout", "type": "number", "default": 0.0}, _ALL_LYCORIS),
    ({"path": "adapter.train_norm", "label": "Train norm layers", "type": "boolean", "default": False, "unless_capability": LINEAR_ONLY_ADAPTERS}, _ALL_LYCORIS),
    (
        {
            "path": "adapter.target_include",
            "label": "Target include patterns",
            "type": "string_list",
            "placeholder": "e.g. *attn*",
        },
        _ALL_LYCORIS,
    ),
    (
        {
            "path": "adapter.target_exclude",
            "label": "Target exclude patterns",
            "type": "string_list",
            "placeholder": "e.g. *ff*",
        },
        _ALL_LYCORIS,
    ),
    # Shared by all except lycoris_dylora
    ({"path": "adapter.train_conv", "label": "Train conv layers", "type": "boolean", "default": False, "unless_capability": LINEAR_ONLY_ADAPTERS}, _LYCORIS_WITH_TRAIN_CONV),
    # locon, loha, lokr — tucker/scalar/wd_on_output
    ({"path": "adapter.use_tucker", "label": "Tucker decomposition", "type": "boolean", "default": False, "unless_capability": LINEAR_ONLY_ADAPTERS}, _LOCON_LOHA_LOKR),
    ({"path": "adapter.use_scalar", "label": "Trained scalar", "type": "boolean", "default": False}, _LOCON_LOHA_LOKR),
    ({"path": "adapter.wd_on_output", "label": "DoRA output axis", "type": "boolean", "default": True}, _LOCON_LOHA_LOKR),
    # DoRA weight decomposition: a toggle on top of locon / loha / lokr.
    ({"path": "adapter.dora_wd", "label": "DoRA decomposition", "type": "boolean", "default": False}, _LOCON_LOHA_LOKR),
    # rs_lora only exists on LoConModule (locon).
    ({"path": "adapter.rs_lora", "label": "Rank-stabilized scale", "type": "boolean", "default": False}, frozenset({"lycoris_locon"})),
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
        # Kohya-style alias of rank, accepted in TOML (defaults.py normalizes it); not a UI
        # field — showing both side by side was two knobs for one value.
        {"path": "adapter.dim", "label": "Dim (alias for rank)", "type": "integer", "min": 1, "ui": False},
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


def get_capability(model_type: str | None) -> ModelCapability | None:
    canonical = normalize_model_type(model_type)
    if not canonical:
        return None
    return model_capability_registry.get(canonical)


def capabilities_for_api() -> dict[str, dict[str, Any]]:
    return {k: v.to_dict() for k, v in model_capability_registry.items()}


def _register_builtin_capabilities() -> None:
    from rengu_flow.registry.builtin_models import BUILTIN_MODEL_SPECS

    for module in BUILTIN_MODEL_SPECS:
        register_model_capability(importlib.import_module(module).SPEC)


_register_builtin_capabilities()
