"""Select which layers an adapter trains: named layer groups + fnmatch globs.

Two selection levels, composable and shared by every adapter family (PEFT LoRA,
vendored LoKr, LyCORIS catalog):

  * ``adapter.layer_groups`` — model-defined names ("text_adapter", "attention", ...)
    that expand to glob patterns over dotted module paths. Each pipeline publishes its
    map (e.g. ``Krea2Pipeline.ADAPTER_LAYER_GROUPS``); the UI offers the names as
    presets per model.
  * ``adapter.target_include`` / ``adapter.target_exclude`` — raw fnmatch globs on the
    dotted module paths, for anything a named group doesn't cover. Same semantics the
    LyCORIS attach path has always used.

Groups expand into ``target_include`` (union with any explicit patterns) before the
per-family attach code runs, so all three families see one consistent mechanism.
"""

from __future__ import annotations

from fnmatch import fnmatch

from rengu_flow.config.validation import ConfigValidationError


def apply_layer_groups(adapter_config: dict, layer_groups: dict[str, tuple] | None) -> None:
    """Expand ``adapter.layer_groups`` names into ``target_include`` globs (in place).

    Unknown names raise with the model's available groups. No-op when the config sets
    no groups. A model without a published map rejects any group name.
    """
    names = adapter_config.get("layer_groups") or []
    if isinstance(names, str):
        names = [names]
    if not names:
        return
    layer_groups = layer_groups or {}
    unknown = [n for n in names if n not in layer_groups]
    if unknown:
        available = ", ".join(sorted(layer_groups)) or "none for this model"
        raise ConfigValidationError(
            f"adapter.layer_groups {unknown} not recognized. Available: {available}."
        )
    include = list(adapter_config.get("target_include") or [])
    for name in names:
        for pattern in layer_groups[name]:
            if pattern not in include:
                include.append(pattern)
    adapter_config["target_include"] = include


def filter_target_names(
    names: list[str],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    """fnmatch filtering over dotted module paths (mirrors the LyCORIS attach globs)."""
    include = include or ["*"]
    exclude = exclude or []
    return [
        n
        for n in names
        if any(fnmatch(n, p) for p in include)
        and not any(fnmatch(n, p) for p in exclude)
    ]
