"""Central rules for which config form fields are shown and which form keys belong to a model.

Paired with ``rengu_flow.registry.model_config_rules`` (TOML validation). Capabilities
live in ``model_capabilities.py``; required keys and ``one_of`` groups are validated there,
while this module handles show/hide in the SPA.

All visibility logic should go through this module. Schema builders attach a normalized
``visibility`` object to each field; the Vue form evaluates the same structure.

Clause types (compose with ``all`` / ``any`` / ``not``):
  - ``field`` + ``in`` | ``equals`` — compare a flat form key (e.g. model.type).
  - ``capability`` — feature flag on the selected model (see ModelCapability.features).
  - ``form_nonempty`` — show when the form already has a non-empty value at ``path``
    (legacy TOML / expert-only fields).
"""

from __future__ import annotations

from typing import Any

from rengu_flow.registry.model_capabilities import (
    model_capability_registry,
    normalize_model_type,
)


def _form_value(form: dict[str, Any], path: str) -> Any:
    return form.get(path)


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def capability_has_feature(capabilities: dict[str, Any] | None, model_type: str | None, feature: str) -> bool:
    if not capabilities or not model_type or not feature:
        return False
    canonical = normalize_model_type(model_type)
    if not canonical:
        return False
    cap = capabilities.get(canonical)
    if not cap:
        return False
    features = cap.get("features") or {}
    return bool(features.get(feature))


def model_supports_adapters(capabilities: dict[str, Any] | None, model_type: str | None) -> bool:
    cap = None
    if capabilities and model_type:
        canonical = normalize_model_type(model_type)
        cap = capabilities.get(canonical or "")
    if not cap:
        return False
    adapters = cap.get("adapters") or []
    return len(adapters) > 0


def eval_visibility_clause(clause: dict[str, Any], form: dict[str, Any], capabilities: dict[str, Any] | None) -> bool:
    if "all" in clause:
        return all(eval_visibility_clause(c, form, capabilities) for c in clause["all"])
    if "any" in clause:
        return any(eval_visibility_clause(c, form, capabilities) for c in clause["any"])
    if "not" in clause:
        return not eval_visibility_clause(clause["not"], form, capabilities)

    if "capability" in clause:
        feature = clause["capability"]
        model_type = _form_value(form, "model.type")
        want = clause.get("equals", True)
        has = capability_has_feature(capabilities, model_type, feature)
        return has if want else not has

    if "form_nonempty" in clause:
        path = clause["form_nonempty"]
        val = _form_value(form, path)
        if not _is_nonempty(val):
            return False
        if clause.get("exclude_zero") and val in (0, 0.0, "0"):
            return False
        return True

    if clause.get("when_model_has_adapter"):
        if not model_supports_adapters(capabilities, _form_value(form, "model.type")):
            return False
        return _form_value(form, "_has_adapter") is True

    field = clause.get("field")
    if field is not None:
        val = _form_value(form, field)
        if "equals" in clause:
            return val == clause["equals"]
        if "in" in clause:
            return val in clause["in"]
    return True


def normalize_field_visibility(field: dict[str, Any]) -> dict[str, Any] | None:
    """Merge legacy ``when`` / ``when_model_has_adapter`` / ``show_if_set`` into one tree."""
    clauses: list[dict[str, Any]] = []

    if field.get("when_model_has_adapter"):
        clauses.append({"when_model_has_adapter": True})

    when = field.get("when")
    if when:
        clauses.append(when)

    when_cap = field.get("when_capability")
    if when_cap:
        if isinstance(when_cap, str):
            clauses.append({"capability": when_cap})
        else:
            clauses.append(when_cap)

    if field.get("show_if_set"):
        path = field.get("path")
        if path:
            entry: dict[str, Any] = {"form_nonempty": path}
            if field.get("show_if_set_exclude_zero"):
                entry["exclude_zero"] = True
            clauses.append(entry)

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"all": clauses}


def field_visible(field: dict[str, Any], form: dict[str, Any], capabilities: dict[str, Any] | None = None) -> bool:
    vis = field.get("visibility")
    if vis is None:
        vis = normalize_field_visibility(field)
    if vis is None:
        return True
    return eval_visibility_clause(vis, form, capabilities or {})


def attach_visibility_to_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Set ``visibility`` on every schema field from legacy keys (idempotent)."""
    for section in schema.get("sections", []):
        for field in section.get("fields", []):
            if field.get("visibility") is None:
                vis = normalize_field_visibility(field)
                if vis is not None:
                    field["visibility"] = vis
    return schema


def model_specific_paths() -> dict[str, frozenset[str]]:
    """Paths owned by each canonical model type (from capability model_fields)."""
    out: dict[str, frozenset[str]] = {}
    for cap in model_capability_registry.values():
        paths = {
            spec["path"]
            for spec in cap.model_fields
            if spec.get("path") and spec.get("ui", True) is not False
        }
        out[cap.type_id] = frozenset(paths)
    return out


def all_model_specific_paths() -> frozenset[str]:
    paths: set[str] = set()
    for owned in model_specific_paths().values():
        paths |= set(owned)
    return frozenset(paths)


def prune_form_for_model(form: dict[str, Any], capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    """Drop model.* keys that do not belong to the selected model.type."""
    from rengu_flow_ui.model_form import prune_toml_only_model_keys

    model_type = normalize_model_type(form.get("model.type"))
    if not model_type:
        return form
    owned = model_specific_paths()
    allowed = owned.get(model_type, frozenset()) | frozenset({"model.type", "model.dtype"})
    stale = all_model_specific_paths() - allowed
    out = dict(form)
    for path in stale:
        out.pop(path, None)
    return prune_toml_only_model_keys(out, model_type)


def models_with_feature(feature: str) -> list[str]:
    return sorted(
        cap.type_id
        for cap in model_capability_registry.values()
        if (cap.features or {}).get(feature)
    )


def when_models_with_feature(feature: str) -> dict[str, Any]:
    """Visibility clause: selected model.type supports ``feature``."""
    types = models_with_feature(feature)
    if not types:
        return {"field": "model.type", "in": []}
    return {"field": "model.type", "in": types}
