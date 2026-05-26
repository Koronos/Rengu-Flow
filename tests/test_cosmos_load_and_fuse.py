"""Cosmos Predict2: load_and_fuse_adapter is explicitly not supported."""

import pytest

pytest.importorskip("torchvision")

from renga_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline


def test_load_and_fuse_adapter_raises():
    config = {
        "model": {
            "type": "cosmos_predict2",
            "dtype": "bfloat16",
            "transformer_path": "t",
            "vae_path": "v",
            "llm_path": "l",
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }

    class _Stub(CosmosPredict2Pipeline):
        def __init__(self, cfg):
            self.config = cfg
            self.model_config = cfg["model"]

    model = _Stub(config)
    with pytest.raises(NotImplementedError, match="load_and_fuse_adapter"):
        model.load_and_fuse_adapter("/tmp/adapter")
