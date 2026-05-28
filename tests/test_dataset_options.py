"""Tests for dataset options ported from diffusion-pipe."""

import json

import torch

from renga_flow.data.cache_utils import seed_from_hash
from renga_flow.data.dataset import (
    SizeBucketDataset,
    directory_subsample_ratio,
    trim_iteration_order_by_subsample_ratio,
)


def test_trim_iteration_order_by_subsample_ratio():
    import datasets

    order = datasets.Dataset.from_dict({"idx": list(range(10))})
    trimmed = trim_iteration_order_by_subsample_ratio(order, 0.25)
    assert len(trimmed) == 2
    assert trimmed["idx"] == [0, 1]
    assert len(trim_iteration_order_by_subsample_ratio(order, 1.0)) == 10


def test_directory_subsample_ratio_defaults_to_full_dataset():
    assert directory_subsample_ratio({}) == 1.0
    assert directory_subsample_ratio({"subsample_ratio": 0.5}) == 0.5
    assert directory_subsample_ratio({"subsample_ratio": 1}) == 1.0


def test_seed_from_hash_deterministic():
    assert seed_from_hash("/data/a") == seed_from_hash("/data/a")
    assert seed_from_hash("/data/a") != seed_from_hash("/data/b")


def test_size_bucket_online_caption_from_dict():
    import datasets

    metadata = datasets.Dataset.from_dict(
        {
            "image_spec": [["", "img.png"]],
            "caption": [["cached cap"]],
        }
    )
    dir_cfg = {"path": "/tmp", "num_repeats": 1}
    class FakeDir:
        def __init__(self):
            self.captions_dict = {"img.png": ["from json", "alt"]}

    ds = SizeBucketDataset(
        metadata,
        dir_cfg,
        (512, 512, 1),
        __import__("pathlib").Path("/tmp/cache"),
        FakeDir(),  # noqa: provides captions_dict
    )
    ds.latent_dataset = [{"latents": torch.zeros(1)}]
    ds.iteration_order = datasets.Dataset.from_dict(
        {
            "image_spec": [["", "img.png"]],
            "latents_idx": [0],
            "caption": ["ignored"],
            "caption_number": [1],
        }
    )
    ds.text_embedding_datasets = []
    ds.uncond_text_embeddings = []
    item = ds[0]
    assert item["caption"] == "alt"
