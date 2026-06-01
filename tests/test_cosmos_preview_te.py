"""The cosmos preview text encoder stays resident between previews (no per-preview disk reload)."""

from __future__ import annotations

import torch

import rengu_flow.model.cosmos_predict2.pipeline as pipeline_mod
from rengu_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline


class _FakeTE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


def _bare_pipeline() -> CosmosPredict2Pipeline:
    obj = object.__new__(CosmosPredict2Pipeline)
    obj.model_config = {"dtype": torch.float32, "llm_path": "x"}
    obj._preview_restore_state = None
    obj._preview_offloader = None
    return obj


def test_preview_text_encoder_loaded_from_disk_once(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_load(_model_config):
        calls["n"] += 1
        return None, None, _FakeTE(), False, "cosmos_predict2"

    monkeypatch.setattr(pipeline_mod, "load_text_stack", fake_load)

    obj = _bare_pipeline()
    obj.text_encoder = _FakeTE()
    obj.text_encoder.to("meta")  # freed by caching (cache_text_embeddings)

    devices: list[str] = []
    for _ in range(3):
        obj.ensure_text_encoder_for_preview(device="cpu")
        obj.offload_text_encoder_after_encode({"preview_offload_text_encoder": True})
        obj.restore_after_preview()
        devices.append(next(obj.text_encoder.parameters()).device.type)

    assert calls["n"] == 1, "text encoder must be loaded from disk once, not per preview"
    assert "meta" not in devices, "text encoder must not be parked back on meta between previews"
    assert devices == ["cpu", "cpu", "cpu"]
