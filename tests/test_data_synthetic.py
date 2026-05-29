"""Tests for SyntheticSDXLDataset."""

import torch

from rengu_flow.data.synthetic import SyntheticSDXLDataset


def test_synthetic_dataset_basic():
    """Len, keys and shapes of getitem are correct."""
    ds = SyntheticSDXLDataset(
        num_batches=2, micro_batch_size=2, latent_channels=4, latent_height=64, latent_width=64
    )
    assert len(ds) == 2
    item = ds[0]
    assert set(item.keys()) == {"latents", "caption", "mask"}
    assert item["latents"].shape == (2, 4, 64, 64)
    assert item["mask"].shape == (2, 1, 64, 64)
    assert len(item["caption"]) == 2


def test_synthetic_dataset_device_and_dtypes():
    ds = SyntheticSDXLDataset(num_batches=1, micro_batch_size=1)
    item = ds[0]
    assert item["latents"].device.type == "cpu"
    assert item["mask"].device.type == "cpu"
    assert item["latents"].dtype == torch.float32
    assert item["mask"].dtype == torch.float32


def test_synthetic_dataset_reproducible():
    ds = SyntheticSDXLDataset(num_batches=3, micro_batch_size=1)
    a = ds[1].copy()
    b = ds[1].copy()
    assert torch.equal(a["latents"], b["latents"])
    assert torch.equal(a["mask"], b["mask"])
