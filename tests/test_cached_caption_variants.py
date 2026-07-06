"""Cached caption variants: in-pipeline tag-dropout/shuffle baking at the TE cache step."""

from __future__ import annotations

from types import SimpleNamespace

import datasets
import torch

from rengu_flow.data.cache_utils import content_fingerprint
from rengu_flow.data.dataset import (
    SizeBucketDataset,
    expand_caption_variants,
)
from rengu_flow.data.tag_dropout import TagDropoutConfig

DROP_HALF = TagDropoutConfig(enabled=True, default_probability=0.5)
NO_DROP = TagDropoutConfig()


def _fake_latent_map(example, rank):
    n = len(example["image_spec"])
    return {"latents": torch.zeros(n, 4)}


def _mock_dir_dataset(dataset_config, tag_dropout=NO_DROP, cache_te=True):
    return SimpleNamespace(
        captions_dict=None,
        uncond_fraction=0.0,
        tag_dropout=tag_dropout,
        dataset_config=dataset_config,
        caches_text_embeddings=cache_te,
        _aug_fingerprint="",
    )


# --- pure function -------------------------------------------------------------


def test_expand_identity_when_off():
    caps = ["a, b, c", "d, e"]
    assert expand_caption_variants(caps, 1, NO_DROP, False, seed_key="img.jpg") == caps


def test_expand_multiplies_per_base_caption():
    out = expand_caption_variants(
        ["a, b, c"], 4, DROP_HALF, False, seed_key="img.jpg"
    )
    assert len(out) == 4
    out2 = expand_caption_variants(
        ["a, b, c", "x, y"], 3, DROP_HALF, False, seed_key="img.jpg"
    )
    assert len(out2) == 6  # 2 base * 3 variants


def test_expand_is_deterministic_and_order_independent():
    a = expand_caption_variants(["one, two, three, four"], 5, DROP_HALF, False, seed_key="k")
    b = expand_caption_variants(["one, two, three, four"], 5, DROP_HALF, False, seed_key="k")
    assert a == b  # deterministic
    # different seed_key (different image) yields an independent draw
    c = expand_caption_variants(["one, two, three, four"], 5, DROP_HALF, False, seed_key="other")
    assert a != c


def test_expand_actually_drops_tags():
    out = expand_caption_variants(
        ["alpha, beta, gamma, delta, epsilon"], 8, DROP_HALF, False, seed_key="k"
    )
    # With p=0.5 over 5 tags across 8 draws, at least one variant must differ from the base.
    assert any(v != "alpha, beta, gamma, delta, epsilon" for v in out)


def test_expand_shuffle_reorders_without_dropping():
    out = expand_caption_variants(
        ["alpha, beta, gamma, delta"], 6, NO_DROP, True, seed_key="k"
    )
    base_tags = {"alpha", "beta", "gamma", "delta"}
    for v in out:
        assert {t.strip() for t in v.split(",")} == base_tags  # same tags, no drop
    assert any(v != "alpha, beta, gamma, delta" for v in out)  # some reordered


def test_directory_dataset_flag_does_not_shadow_method(tmp_path):
    """Regression: the cache flag must not shadow DirectoryDataset.cache_text_embeddings()."""
    from rengu_flow.data.dataset import DirectoryDataset

    dd = DirectoryDataset(
        {"path": str(tmp_path), "num_repeats": 1},
        {"resolutions": [512]},
        "sdxl",
        skip_dataset_validation=True,
        cache_text_embeddings=True,
    )
    assert dd.caches_text_embeddings is True
    assert callable(dd.cache_text_embeddings)  # still the method, not the bool flag


# --- integration through SizeBucketDataset -------------------------------------


def _build_bucket(tmp_path, metadata, dataset_config, tag_dropout=NO_DROP, cache_te=True):
    sb = SizeBucketDataset(
        metadata,
        {"path": str(tmp_path), "num_repeats": 1},
        (512, 512, 1),
        tmp_path / "cache",
        _mock_dir_dataset(dataset_config, tag_dropout, cache_te),
    )
    sb.cache_latents(_fake_latent_map, regenerate_cache=True, trust_cache=False)
    return sb


def _metadata(n=3):
    return datasets.Dataset.from_dict(
        {
            "image_spec": [[None, f"img{i}.jpg"] for i in range(n)],
            "caption": [["red, hair, smile, outdoors"] for _ in range(n)],
        }
    )


def test_bucket_bakes_k_variants(tmp_path):
    sb = _build_bucket(
        tmp_path, _metadata(3), {"cached_caption_variants": 4}, tag_dropout=DROP_HALF
    )
    assert sb.caption_variants == 4
    assert len(sb.iteration_order) == 12  # 3 images * 4 variants
    assert sorted({row["caption_number"] for row in sb.iteration_order}) == [0, 1, 2, 3]


def test_bucket_k1_dropout_bakes_single_fixed_variant(tmp_path):
    """K=1 + dropout is allowed (diffusion-pipe default): one fixed baked variant, not rejected."""
    sb = _build_bucket(
        tmp_path, _metadata(3), {"cached_caption_variants": 1}, tag_dropout=DROP_HALF
    )
    assert sb._caption_variants_expanded is True
    assert sb.caption_variants == 1
    assert len(sb.iteration_order) == 3  # 3 images * 1 variant
    base_tags = {"red", "hair", "smile", "outdoors"}
    for row in sb.iteration_order:
        # dropout applied once -> a (possibly trimmed) subset of the base tags
        assert {t.strip() for t in row["caption"].split(",") if t.strip()} <= base_tags


def test_bucket_no_expansion_when_cache_off(tmp_path):
    sb = _build_bucket(
        tmp_path,
        _metadata(3),
        {"cached_caption_variants": 4},
        tag_dropout=DROP_HALF,
        cache_te=False,
    )
    # Cache off -> live dropout path; variants are NOT pre-baked.
    assert sb.caption_variants == 1
    assert len(sb.iteration_order) == 3


def test_bucket_default_is_unchanged(tmp_path):
    sb = _build_bucket(tmp_path, _metadata(3), {})  # K defaults to 1, no dropout
    assert sb.caption_variants == 1
    assert len(sb.iteration_order) == 3


def test_iteration_order_rebuilds_when_k_changes(tmp_path):
    meta = _metadata(2)
    sb3 = _build_bucket(tmp_path, meta, {"cached_caption_variants": 3}, tag_dropout=DROP_HALF)
    assert len(sb3.iteration_order) == 6

    # Same cache dir, trust_cache=True, but K changed: the caption fingerprint sidecar must
    # force a rebuild instead of serving the stale 3-variant order.
    sb5 = SizeBucketDataset(
        _metadata(2),
        {"path": str(tmp_path), "num_repeats": 1},
        (512, 512, 1),
        tmp_path / "cache",
        _mock_dir_dataset({"cached_caption_variants": 5}, DROP_HALF, True),
    )
    sb5.cache_latents(_fake_latent_map, regenerate_cache=False, trust_cache=True)
    assert len(sb5.iteration_order) == 10  # rebuilt for K=5


# --- root-keyed text embeddings (augmentation variants share one root's captions) ------------


def test_caption_variants_shared_across_augmentation_of_same_root():
    """Augmentation variants of one root image get IDENTICAL baked captions (seeded by the root,
    not the per-crop variant_key), so the TE cache can store them once instead of once per crop."""
    from rengu_flow.data.dataset import maybe_expand_caption_variants

    meta = datasets.Dataset.from_dict(
        {
            "image_spec": [[None, "a.jpg"], [None, "a.jpg", "flipx"], [None, "a.jpg", "crop2"]],
            "caption": [["red, hair, smile, outdoors"]] * 3,
        }
    )
    expanded, did = maybe_expand_caption_variants(
        meta, _mock_dir_dataset({"cached_caption_variants": 4}, DROP_HALF)
    )
    assert did
    assert len(expanded["caption"][0]) == 4  # K baked variants per base caption
    # all three augmentation variants of the root -> identical baked caption set
    assert expanded["caption"][0] == expanded["caption"][1] == expanded["caption"][2]


def test_text_embedding_dataset_resolves_by_root_image():
    """Any augmentation variant's full image_spec resolves to the same (root-deduped) TE rows."""
    from rengu_flow.data.dataset import TextEmbeddingDataset

    flattened = {"image_spec": [[None, "a.jpg"], [None, "a.jpg"]]}  # 2 caption rows, one root
    ds = TextEmbeddingDataset([{"prompt_embeds": "row0"}, {"prompt_embeds": "row1"}], flattened)
    assert list(ds.image_spec_to_te_idx) == [(None, "a.jpg")]  # one entry keyed by the root
    for aug in [(None, "a.jpg"), (None, "a.jpg", "flipx"), (None, "a.jpg", "crop2")]:
        assert ds.get_text_embeddings(aug, 0) == {"prompt_embeds": "row0"}
        assert ds.get_text_embeddings(aug, 1) == {"prompt_embeds": "row1"}


def test_cache_text_embeddings_dedups_augmentation_variants(tmp_path):
    """_cache_text_embeddings stores one row per (root, caption): augmentation variants of a
    source image (identical captions) collapse instead of each writing its own embedding."""
    from rengu_flow.data.dataset import _cache_text_embeddings

    specs = []
    for root in ("a.jpg", "b.jpg"):
        specs += [[None, root], [None, root, "v1"], [None, root, "v2"]]
    meta = datasets.Dataset.from_dict({"image_spec": specs, "caption": [["one caption"]] * 6})

    def fake_te_map(example, rank):
        return {"prompt_embeds": torch.zeros(len(example["caption"]), 2)}

    te = _cache_text_embeddings(
        meta, fake_te_map, 1, tmp_path / "cache", True, 1, cache_num_proc=1
    )
    assert len(te.te_dataset) == 2  # 2 roots x 1 caption, NOT 6 (variants collapsed)
    assert set(te.image_spec_to_te_idx) == {(None, "a.jpg"), (None, "b.jpg")}
    assert te.get_text_embeddings((None, "a.jpg", "v2"), 0) is not None  # a crop resolves to root


def test_te_cache_fingerprint_tracks_variant_config(tmp_path):
    """The text-embedding cache is keyed by caption content, so a K/seed change shifts it."""
    cols = ["caption", "image_spec"]
    m3 = _build_bucket(tmp_path / "a", _metadata(2), {"cached_caption_variants": 3}, DROP_HALF)
    m3b = _build_bucket(tmp_path / "b", _metadata(2), {"cached_caption_variants": 3}, DROP_HALF)
    m5 = _build_bucket(tmp_path / "c", _metadata(2), {"cached_caption_variants": 5}, DROP_HALF)
    fp3 = content_fingerprint(m3.metadata_dataset, cols)
    fp3b = content_fingerprint(m3b.metadata_dataset, cols)
    fp5 = content_fingerprint(m5.metadata_dataset, cols)
    assert fp3 == fp3b  # identical config -> identical (cache reused)
    assert fp3 != fp5  # K changed -> fingerprint changed (cache regenerated)
