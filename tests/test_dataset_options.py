"""Tests for dataset options ported from diffusion-pipe."""


import torch

from rengu_flow.data.cache_utils import seed_from_hash
from rengu_flow.data.dataset import (
    SizeBucketDataset,
    directory_subsample_ratio,
)



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
        FakeDir(),  # provides captions_dict
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


def _uncond_size_bucket(tmp_path, fake_dir):
    import datasets

    metadata = datasets.Dataset.from_dict(
        {"image_spec": [["", "img.png"]], "caption": [["a cap"]]}
    )

    class FakeTE:
        def get_text_embeddings(self, image_spec, caption_number):
            return {"emb": "cond"}

    ds = SizeBucketDataset(
        metadata,
        {"path": str(tmp_path), "num_repeats": 1},
        (512, 512, 1),
        tmp_path / "cache",
        fake_dir,
    )
    ds.latent_dataset = [{"latents": torch.zeros(1)}]
    ds.iteration_order = datasets.Dataset.from_dict(
        {
            "image_spec": [["", "img.png"]],
            "latents_idx": [0],
            "caption": ["a cap"],
            "caption_number": [0],
        }
    )
    ds.text_embedding_datasets = [FakeTE()]
    ds.uncond_text_embeddings = [[{"emb": "uncond"}]]
    return ds


def test_uncond_fraction_one_swaps_caption_and_embeddings(tmp_path):
    class FakeDir:
        captions_dict = None
        uncond_fraction = 1.0

    ds = _uncond_size_bucket(tmp_path, FakeDir())
    item = ds[0]
    assert item["caption"] == ""
    assert item["emb"] == "uncond"


def test_uncond_fraction_defaults_to_zero(tmp_path):
    class FakeDir:
        captions_dict = None

    ds = _uncond_size_bucket(tmp_path, FakeDir())
    item = ds[0]
    assert item["caption"] == "a cap"
    assert item["emb"] == "cond"


def test_directory_dataset_resolves_uncond_fraction(tmp_path):
    from rengu_flow.data.dataset import DirectoryDataset

    dataset_config = {
        "resolutions": [512],
        "uncond_fraction": 0.1,
    }
    dd = DirectoryDataset(
        {"path": str(tmp_path), "num_repeats": 1, "uncond_fraction": 0.25},
        dict(dataset_config),
        "sdxl",
        skip_dataset_validation=True,
    )
    assert dd.uncond_fraction == 0.25
    dd2 = DirectoryDataset(
        {"path": str(tmp_path), "num_repeats": 1},
        dict(dataset_config),
        "sdxl",
        skip_dataset_validation=True,
    )
    assert dd2.uncond_fraction == 0.1


def test_tag_dropout_applied_to_sampled_caption(tmp_path):
    from rengu_flow.data.tag_dropout import TagDropoutConfig

    class FakeDir:
        captions_dict = None
        tag_dropout = TagDropoutConfig(enabled=True, default_probability=1.0)

    ds = _uncond_size_bucket(tmp_path, FakeDir())
    item = ds[0]
    assert item["caption"] == ""
    assert item["emb"] == "cond"


def test_directory_dataset_builds_tag_dropout(tmp_path):
    from rengu_flow.data.dataset import DirectoryDataset

    dd = DirectoryDataset(
        {
            "path": str(tmp_path),
            "num_repeats": 1,
            "tag_dropout_enabled": True,
            "tag_dropout_probability": 0.5,
        },
        {"resolutions": [512]},
        "sdxl",
        skip_dataset_validation=True,
    )
    assert dd.tag_dropout.enabled
    assert dd.tag_dropout.default_probability == 0.5
    dd2 = DirectoryDataset(
        {"path": str(tmp_path), "num_repeats": 1},
        {"resolutions": [512]},
        "sdxl",
        skip_dataset_validation=True,
    )
    assert not dd2.tag_dropout.enabled


def test_dataset_tag_dropout_with_cached_text_embeddings_is_allowed(tmp_path):
    """Tag dropout + cached TE is no longer rejected: it is baked into the cache (K>=1)."""
    from rengu_flow.data.dataset import Dataset

    dataset_config = {
        "resolutions": [512],
        "tag_dropout_enabled": True,
        "directory": [{"path": str(tmp_path), "num_repeats": 1}],
    }

    class CachedTEModel:
        name = "sdxl"

        def get_text_encoders(self):
            return ["te"]

    # Cache on + dropout + default K=1 -> one fixed baked variant (diffusion-pipe default), no raise.
    Dataset(dict(dataset_config), CachedTEModel())

    # Cache on + dropout + K >= 2 -> rotating baked variants, no raise.
    k2_config = dict(dataset_config)
    k2_config["cached_caption_variants"] = 4
    Dataset(dict(k2_config), CachedTEModel())

    # Cache off (live encoding) -> live per-sample dropout, no raise.
    class LiveTEModel(CachedTEModel):
        def get_text_encoders(self):
            return []

    Dataset(dict(dataset_config), LiveTEModel())
