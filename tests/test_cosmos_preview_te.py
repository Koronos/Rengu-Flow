"""The cosmos preview text encoder stays resident between previews (no per-preview disk reload)."""

from __future__ import annotations

import pytest
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
    obj.cache_text_embeddings = True
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


def test_training_resident_text_encoder_never_offloaded() -> None:
    """cache_text_embeddings=false: the TE is in the training graph; a preview CPU round-trip
    reassigns param .data under the fused optimizer / compiled graph → cudaErrorIllegalAddress
    on the next step. The offload must be a no-op regardless of preview_offload_text_encoder."""
    obj = _bare_pipeline()
    obj.cache_text_embeddings = False
    obj.text_encoder = _FakeTE()

    obj.ensure_text_encoder_for_preview(device="cpu")
    moves: list = []
    obj.text_encoder.to = lambda *a, **k: moves.append(a)  # type: ignore[method-assign]
    obj.offload_text_encoder_after_encode({"preview_offload_text_encoder": True})

    assert not moves, (
        "training-resident text encoder must not be moved (its param storages are referenced "
        "by the optimizer and compiled graph)"
    )


class _SpyDiT(torch.nn.Module):
    """First parameter lives on the target device (a never-swapped top-level module), like the
    real DiT under block swap; records every whole-module ``.to``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(2, 2)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.moves: list[str] = []

    def to(self, *args, **kwargs):
        self.moves.append(str(args[0]))
        return super().to(*args, **kwargs)


def test_ensure_transformer_force_move_fixes_blocks_parked_off_device() -> None:
    obj = _bare_pipeline()
    obj.transformer = _SpyDiT()
    obj.ensure_transformer_for_preview("cpu")
    assert obj.transformer.moves == []  # first param already on target: no-op (no needless .to)
    obj.ensure_transformer_for_preview("cpu", force_move=True)
    assert obj.transformer.moves == ["cpu"]


class _Offloader:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.suspended = False

    def suspend(self) -> None:
        self.suspended = True


@pytest.mark.parametrize(
    "train_swap,preview_swap,expect_force",
    [(True, 0, True), (True, 4, False), (False, 0, False)],
    ids=["train_swap_no_preview_swap", "train_swap_and_preview_swap", "no_swap"],
)
def test_prepare_preview_memory_forces_the_dit_back_when_training_swap_parked_it(
    monkeypatch, train_swap, preview_swap, expect_force
) -> None:
    obj = _bare_pipeline()
    obj.transformer = _SpyDiT()
    obj._block_swap_offloader = _Offloader(train_swap)
    seen = {}
    monkeypatch.setattr(
        obj, "ensure_transformer_for_preview", lambda device="cuda", *, force_move=False: seen.update(force=force_move)
    )
    monkeypatch.setattr(obj, "_make_preview_offloader", lambda *a, **k: object())
    obj.prepare_preview_memory({"preview_blocks_to_swap": preview_swap})
    assert obj._block_swap_offloader.suspended is train_swap
    assert seen["force"] is expect_force
