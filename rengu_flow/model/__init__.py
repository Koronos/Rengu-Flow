"""Model contract and base for Rengu Flow pipelines."""

from rengu_flow.model.base import BasePipeline, ModelPipelineProtocol, make_contiguous

__all__ = [
    "BasePipeline",
    "ModelPipelineProtocol",
    "make_contiguous",
    "SDXLPipeline",
    "CosmosPredict2Pipeline",
    "Krea2Pipeline",
]


def __getattr__(name: str):
    if name == "SDXLPipeline":
        from rengu_flow.model.sdxl import SDXLPipeline

        return SDXLPipeline
    if name == "CosmosPredict2Pipeline":
        from rengu_flow.model.cosmos_predict2 import CosmosPredict2Pipeline

        return CosmosPredict2Pipeline
    if name == "Krea2Pipeline":
        from rengu_flow.model.krea2 import Krea2Pipeline

        return Krea2Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
