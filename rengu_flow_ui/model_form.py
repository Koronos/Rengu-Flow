"""Model section: TOML-only paths, pruning, and form visibility.

Per-model checkpoint fields come from ``ModelCapability.model_fields`` with an
automatic ``when: model.type in [type_id]`` clause. Fields marked ``ui: false`` are
accepted in TOML (and may appear in parsed forms) but are hidden from the web UI
when they are not used for that pipeline (see ``ModelCapability`` field ``ui`` flags).
"""

from __future__ import annotations

from typing import Any

from rengu_flow.registry.model_capabilities import (
    get_capability,
    model_capability_registry,
    normalize_model_type,
)


def toml_only_model_paths_for_type(model_type: str | None) -> frozenset[str]:
    """Flat form paths hidden for ``model_type`` (``ui: false`` on capability field specs)."""
    canonical = normalize_model_type(model_type)
    if not canonical:
        return frozenset()
    cap = get_capability(canonical)
    if not cap:
        return frozenset()
    return frozenset(
        spec["path"]
        for spec in cap.model_fields
        if spec.get("path") and spec.get("ui") is False
    )


def all_toml_only_model_paths() -> frozenset[str]:
    paths: set[str] = set()
    for cap in model_capability_registry.values():
        paths |= set(toml_only_model_paths_for_type(cap.type_id))
    return frozenset(paths)


def hidden_model_paths_by_type() -> dict[str, list[str]]:
    """Document which model.* paths are UI-hidden per canonical type (for tests/docs)."""
    return {
        cap.type_id: sorted(toml_only_model_paths_for_type(cap.type_id))
        for cap in model_capability_registry.values()
        if toml_only_model_paths_for_type(cap.type_id)
    }


def prune_toml_only_model_keys(form: dict[str, Any], model_type: str | None = None) -> dict[str, Any]:
    """Drop ``ui: false`` model.* keys for the selected type (e.g. after type change)."""
    model_type = model_type if model_type is not None else form.get("model.type")
    stale = toml_only_model_paths_for_type(str(model_type) if model_type else None)
    if not stale:
        return form
    out = dict(form)
    for path in stale:
        out.pop(path, None)
    return out


def attach_model_section_visibility(fields: list[dict[str, Any]]) -> None:
    """Normalize ``visibility`` on model section fields (idempotent).

    Model-specific paths already carry ``when: model.type in [...]`` from
    ``config_schema._model_section_fields``. This pass only ensures legacy keys
    are folded into ``visibility`` for the SPA.
    """
    from rengu_flow_ui.field_visibility import normalize_field_visibility

    for field in fields:
        path = field.get("path", "")
        if not path.startswith("model.") or path in ("model.type", "model.dtype"):
            continue
        if field.get("visibility") is not None:
            continue
        vis = normalize_field_visibility(field)
        if vis is not None:
            field["visibility"] = vis
