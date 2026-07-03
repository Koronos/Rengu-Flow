"""Adapter (network) form helpers. Keep in sync with ui/web/src/lib/adapterForm.ts.

Safety net for the form->config render path: when an adapter config arrives with
keys from a different network type (library load, import, raw API), drop the keys
that do not belong to the current ``adapter.type``. The interactive type-change
prune+seed lives in the frontend; this side only prunes. Only ``adapter.*`` keys
are touched, never other sections.
"""

from __future__ import annotations

from typing import Any

from rengu_flow.registry.model_capabilities import ADAPTER_FIELD_TEMPLATES


def allowed_adapter_paths(adapter_type: Any) -> frozenset[str]:
    """The ``adapter.*`` paths valid for this network type (common + per-kind)."""
    keep: set[str] = {
        "adapter.type",
        # Layer selection applies to every family (groups expand into include globs).
        "adapter.layer_groups",
        "adapter.target_include",
        "adapter.target_exclude",
    }
    for spec in ADAPTER_FIELD_TEMPLATES["common"]:
        keep.add(spec["path"])
    for spec in ADAPTER_FIELD_TEMPLATES.get(adapter_type, []):
        keep.add(spec["path"])
    return frozenset(keep)


def prune_adapter_form(form: dict[str, Any], adapter_type: Any | None = None) -> dict[str, Any]:
    """Drop adapter.* keys that do not apply to the current adapter.type.

    Unknown types are left untouched (the validator reports them) so we never strip
    a config we cannot reason about.
    """
    adapter_type = adapter_type if adapter_type is not None else form.get("adapter.type")
    if not adapter_type or adapter_type not in ADAPTER_FIELD_TEMPLATES:
        return form
    allowed = allowed_adapter_paths(adapter_type)
    out = dict(form)
    for key in list(out):
        if key.startswith("adapter.") and key not in allowed:
            out.pop(key, None)
    return out
