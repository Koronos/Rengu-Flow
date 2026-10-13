"""Toy test kit: the Nanosaur2 kit building the toy pipeline (see ``testkit_common``)."""

from __future__ import annotations

from rengu_flow.model import testkit_common
from rengu_flow.model.nanosaur2 import testkit as _base

NOT_APPLICABLE: dict[str, str] = {}
CAPABILITIES = testkit_common.ALL_CAPABILITIES - NOT_APPLICABLE.keys()
VAE_INPUT_SHAPE = _base.VAE_INPUT_SHAPE
monolithic_forward = _base.monolithic_forward
export_transformer_path = _base.export_transformer_path
preview_config = _base.preview_config
cases = _base.cases


def tiny_config(tmp_path, **over) -> dict:
    config = _base.tiny_config(tmp_path, **over)
    config["model"]["type"] = "toy_dit"
    return config


def tiny_pipeline(config):
    from toy_model.pipeline import ToyPipeline

    pipeline = ToyPipeline(config)
    pipeline.load_diffusion_model()
    if config.get("adapter"):
        config["adapter"] = testkit_common.prepare_adapter_config(config["adapter"], config["model"]["dtype"])
        pipeline.configure_adapter(config["adapter"])
    return pipeline
