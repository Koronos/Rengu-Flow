"""Model contract and base for Renga Flow pipelines."""

from renga_flow.model.base import BasePipeline, ModelPipelineProtocol, make_contiguous

# Register built-in models so get_model() can resolve them.
from renga_flow.model import sdxl  # noqa: F401
from renga_flow.model import cosmos_predict2  # noqa: F401
from renga_flow.model.sdxl import SDXLPipeline
from renga_flow.model.cosmos_predict2 import CosmosPredict2Pipeline

__all__ = [
    "BasePipeline",
    "ModelPipelineProtocol",
    "make_contiguous",
    "SDXLPipeline",
    "CosmosPredict2Pipeline",
]
