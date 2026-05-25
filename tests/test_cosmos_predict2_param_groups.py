"""CosmosPredict2Pipeline.get_param_groups without loading checkpoints."""

import torch

from renga_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline


def _param(name: str) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.zeros(1))
    p.original_name = name
    return p


def _pipeline_config():
    return {
        "model": {
            "type": "cosmos_predict2",
            "dtype": "bfloat16",
            "transformer_path": "t",
            "vae_path": "v",
            "llm_path": "l",
            "llm_adapter_lr": 0,
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }


def test_llm_adapter_lr_zero_freezes_adapter_params(monkeypatch):
    """llm_adapter_lr=0 must disable grads on llm_adapter weights."""
    monkeypatch.setattr(
        CosmosPredict2Pipeline,
        "__init__",
        lambda self, config: setattr(self, "config", config)
        or setattr(self, "model_config", config["model"]),
    )
    pipe = CosmosPredict2Pipeline(_pipeline_config())
    params = [
        _param("blocks.0.self_attn.q_proj.weight"),
        _param("blocks.0.llm_adapter.out_proj.weight"),
        _param("llm_adapter.in_proj.weight"),
    ]
    groups = pipe.get_param_groups(params)
    assert params[1].requires_grad is False
    assert params[2].requires_grad is False
    assert params[0].requires_grad is True
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-4


def test_llm_adapter_lr_positive_includes_group(monkeypatch):
    config = _pipeline_config()
    config["model"]["llm_adapter_lr"] = 2e-5
    monkeypatch.setattr(
        CosmosPredict2Pipeline,
        "__init__",
        lambda self, cfg: setattr(self, "config", cfg) or setattr(self, "model_config", cfg["model"]),
    )
    pipe = CosmosPredict2Pipeline(config)
    params = [_param("llm_adapter.out_proj.weight")]
    groups = pipe.get_param_groups(params)
    assert len(groups) == 1
    assert groups[0]["lr"] == 2e-5
