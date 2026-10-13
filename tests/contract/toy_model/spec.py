"""Toy model descriptor: the Nanosaur2 spec under another type id and distinct derived values."""

from __future__ import annotations

import copy
from dataclasses import replace

from rengu_flow.model.nanosaur2.spec import SPEC as _BASE



def _field(base: dict) -> dict:
    """Nanosaur2's field without its explicit doc link (the toy's ``user_doc`` is the default)."""
    field = {k: v for k, v in copy.deepcopy(base).items() if k != "doc"}
    if field["path"] == "model.sparse_skip_prob":
        field["default"] = 0.25
    return field


SPEC = replace(
    _BASE,
    type_id="toy_dit",
    display_name="Toy DiT",
    aliases=[],
    env_prefix="TOY_DIT",
    pipeline="toy_model.pipeline:ToyPipeline",
    testkit="toy_model.testkit",
    user_doc="tests/contract/toy_model/README.md",
    install_profiles=("cosmos", "optim"),
    preview_spec=replace(_BASE.preview_spec, steps=11, guidance=2.5, negative_prompt="toy negative"),
    model_defaults={**_BASE.model_defaults, "sparse_skip_prob": 0.25},
    ragged_cache_keys=_BASE.ragged_cache_keys | {"toy_rows"},
    model_fields=[_field(f) for f in _BASE.model_fields],
)
