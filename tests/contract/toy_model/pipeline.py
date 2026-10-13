"""Toy pipeline: the Nanosaur2 pipeline registered under the toy type id."""

from __future__ import annotations

from rengu_flow.model.nanosaur2.pipeline import Nanosaur2Pipeline
from rengu_flow.registry.models import register_model


@register_model("toy_dit")
class ToyPipeline(Nanosaur2Pipeline):
    name = "toy_dit"
