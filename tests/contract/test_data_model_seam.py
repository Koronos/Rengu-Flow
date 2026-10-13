"""Contract C: the data <-> model seam without doubles, for every case of every model kit.

Real images (and control images for edit cases) in a temp folder -> the real ``Dataset`` /
``DatasetManager`` cache through the tiny model's own VAE and text-encoder fns -> the loader's
micro-batch split (GAS 1 and 2) and the model's micro-batch trim -> ``prepare_inputs`` -> every
pipeline layer -> loss and backward. Square and non-square buckets, captions of mixed lengths
(one empty). Every cache written reopens past the stale-layout guard."""

from __future__ import annotations

import gc
import random

import pytest
import torch

from rengu_flow.model.testkit_common import run_layers
from rengu_flow.utils.cache import MANIFEST_NAME, reject_stale_sequence_layout

pytestmark = pytest.mark.no_ui_db


def _text_lengths(batch) -> set[int]:
    rows = batch["prompt_embeds"]
    rows = list(rows.unbind(0)) if torch.is_tensor(rows) else list(rows)
    return {int(r.shape[0]) for r in rows}


def test_dataset_batches_train_through_the_layers(kit_case, tmp_path):
    from rengu_flow.data.dataset import Dataset
    from rengu_flow.data.loader import split_batch
    from rengu_flow.data.manager import DatasetManager
    from rengu_flow.engine import select_backend

    kit, case = kit_case
    if case.write_dataset is None:
        pytest.skip(f"{kit.type_id}/{case.name}: no CPU-encodable media for this case")
    dataset_config = case.write_dataset(tmp_path / "data")
    gc.collect()
    p = kit.pipeline()
    ds = Dataset(dataset_config, p, training_config={"cache_root": str(tmp_path / "cache")})
    manager = DatasetManager(p, backend=select_backend({"engine": "accelerate"}))
    manager.register(ds)
    manager.cache(unload_models=False)
    ds.post_init(0, 1, {None: 2}, 1, {None: 2})
    assert len(ds) > 0
    manifests = sorted((tmp_path / "cache").rglob(MANIFEST_NAME))
    assert manifests
    for manifest in manifests:  # the model's real rows reopen in the current (ragged) layout
        reject_stale_sequence_layout(manifest.parent)

    layers = p.to_layers()
    loss_fn = p.get_loss_fn()
    trim = getattr(type(p), "trim_micro_batch", None)
    grids, lengths, controls = set(), set(), set()
    random.seed(0)
    torch.manual_seed(0)
    for accumulation in (1, 2):
        for i in range(len(ds)):
            batch = ds[i]
            grids.add(tuple(batch["latents"].shape[-2:]))
            lengths |= _text_lengths(batch)
            controls.add(sum(k.startswith("control_latents_") for k in batch))
            features, label = p.prepare_inputs(batch)
            for micro_features, micro_label in split_batch((features, label), accumulation):
                if callable(trim):
                    micro_features = p.trim_micro_batch(micro_features)
                out = run_layers(layers, micro_features)
                pred = out[0] if isinstance(out, tuple) else out  # SDXL appends its timesteps
                assert pred.shape == micro_label[0].shape and torch.isfinite(pred).all()
                loss = loss_fn(out, micro_label)
                assert torch.isfinite(loss)
                loss.backward()
    assert case.expected_grids <= grids, grids
    assert controls == {case.controls}, controls
    if kit.has("ragged_text"):
        assert len(lengths) > 1, f"captions did not produce mixed text lengths: {lengths}"
    for end in (layers[0], layers[-1]):  # the gradient crossed every layer of the pipeline
        assert any(prm.grad is not None for prm in end.parameters()), f"no gradient reached {type(end).__name__}"
