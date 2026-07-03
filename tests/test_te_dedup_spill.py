"""Tests for the disk-backed text-embedding dedup spill (per-sample over batched maps)."""

from __future__ import annotations

import torch

from rengu_flow.data.manager import _TextEmbeddingDedup


def _batch_result(lengths: list[int], dim=(12, 8)) -> dict:
    max_len = max(lengths)
    embeds = torch.zeros(len(lengths), max_len, *dim, dtype=torch.bfloat16)
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for i, n in enumerate(lengths):
        embeds[i, :n] = torch.randn(n, *dim, dtype=torch.bfloat16)
        mask[i, :n] = True
    return {"prompt_embeds": embeds, "text_mask": mask, "image_spec": ["x"] * len(lengths)}


def test_spill_batches_with_different_padded_lengths(tmp_path):
    """Regression: batched results are padded to the batch max, so consecutive batches
    with different max lengths (e.g. (4,115,12,2560) then (4,111,12,2560)) must not
    collide in the spill — rows are stored per sample."""
    dedup = _TextEmbeddingDedup(tmp_path / "spill")
    r1 = _batch_result([115, 90])
    dedup.store([(1, "a"), (1, "b")], r1)
    r2 = _batch_result([111, 60])  # smaller batch max: crashed the old whole-batch spill
    dedup.store([(1, "c"), (1, "d")], r2)

    hit = dedup.lookup([(1, "b"), (1, "c")])  # cross-batch reassembly
    assert hit is not None
    assert [t.shape[0] for t in hit["prompt_embeds"]] == [115, 111]
    assert torch.equal(hit["prompt_embeds"][0], r1["prompt_embeds"][1])
    assert torch.equal(hit["prompt_embeds"][1], r2["prompt_embeds"][0])
    assert [m.sum().item() for m in hit["text_mask"]] == [90, 111]
    dedup.close()


def test_spill_partial_hit_returns_none(tmp_path):
    dedup = _TextEmbeddingDedup(tmp_path / "spill")
    dedup.store([(1, "a")], _batch_result([10]))
    assert dedup.lookup([(1, "a"), (1, "zz")]) is None
    assert dedup.lookup([(1, "a")]) is not None
    dedup.close()


def test_spill_keys_are_per_text_encoder(tmp_path):
    dedup = _TextEmbeddingDedup(tmp_path / "spill")
    dedup.store([(1, "same-caption")], _batch_result([10]))
    assert dedup.lookup([(2, "same-caption")]) is None  # other encoder: no cross-hit
    dedup.close()


def test_spill_dir_removed_on_close(tmp_path):
    spill_dir = tmp_path / "spill"
    dedup = _TextEmbeddingDedup(spill_dir)
    dedup.store([(1, "a")], _batch_result([10]))
    dedup.close()
    assert not spill_dir.exists()


def test_pipe_transport_roundtrips_tensors_as_numpy():
    """Regression for the 60GB caching RAM leak: unpickling a torch tensor leaks its
    full storage on torch 2.12, so pipe payloads must travel as numpy (bf16 tagged as a
    uint16 view) and rebuild identically on the other side."""
    import pickle

    import numpy as np
    import torch

    from rengu_flow.data.manager import _from_pipe, _to_pipe

    payload = {
        "prompt_embeds": torch.randn(7, 12, 16, dtype=torch.bfloat16),
        "text_mask": torch.ones(7, dtype=torch.bool),
        "latents": torch.randn(4, 8),
        "image_spec": [(None, "img.jpg")],
        "nested": [torch.randn(3, dtype=torch.bfloat16)],
    }
    packed = _to_pipe(payload)

    def assert_no_tensors(obj):
        assert not torch.is_tensor(obj), "a torch tensor reached the pipe"
        if isinstance(obj, dict):
            for v in obj.values():
                assert_no_tensors(v)
        elif isinstance(obj, (list, tuple)) and not (
            len(obj) == 2 and obj[0] == "__pipe_bf16__"
        ):
            for v in obj:
                assert_no_tensors(v)

    assert_no_tensors(packed)
    restored = _from_pipe(pickle.loads(pickle.dumps(packed)))
    assert torch.equal(restored["prompt_embeds"], payload["prompt_embeds"])
    assert restored["prompt_embeds"].dtype == torch.bfloat16
    assert torch.equal(restored["text_mask"], payload["text_mask"])
    assert torch.equal(restored["latents"], payload["latents"])
    assert torch.equal(restored["nested"][0], payload["nested"][0])
    assert restored["image_spec"] == [(None, "img.jpg")]
