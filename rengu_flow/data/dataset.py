"""Directory-based dataset with buckets, cache, and multi-caption (from diffusion-pipe)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import datasets
import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from rengu_flow.data.augmentation import (
    augmentation_fingerprint,
    expand_variant_keys,
    is_augmentation_enabled,
    validate_augmentation_for_directory,
    with_variant_key,
)
from rengu_flow.data.augmentation.names import AUG_MVP_VERSION
from rengu_flow.data.cache_utils import (
    _map_and_cache,
    bucket_suffix,
    content_fingerprint,
    dedup_and_sort,
    resolve_cache_num_proc,
    seed_from_hash,
)
from rengu_flow.data.tag_dropout import (
    TagDropoutConfig,
    apply_tag_dropout,
    build_tag_dropout_config,
    join_tags,
    split_tags,
)
from rengu_flow.utils.common import is_main_process, round_to_nearest_multiple
from rengu_flow.utils.paths import path_is_under

logger = logging.getLogger(__name__)

CAPTIONS_JSON_FILE = "captions.json"

# Video extensions from imageio (same as diffusion-pipe utils.common)
VIDEO_EXTENSIONS = set()
try:
    for x in getattr(imageio.config, "video_extensions", []):
        ext = getattr(x, "extension", x) if hasattr(x, "extension") else x
        VIDEO_EXTENSIONS.add(ext)
        VIDEO_EXTENSIONS.add(ext.upper() if isinstance(ext, str) else ext)
except Exception:
    pass
if not VIDEO_EXTENSIONS:
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def shuffle_with_seed(lst: list, seed=None) -> None:
    """Shuffle list in place with optional seed (restores RNG state after)."""
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(lst)
    random.setstate(rng_state)


def _read_captions_from_txt_per_line(caption_file: str) -> list[str]:
    """Read .txt file as one caption per line (rengu-flow behavior). Empty lines skipped."""
    with open(caption_file) as f:
        captions = [line.strip() for line in f if line.strip()]
    return captions if captions else [""]


class TextEmbeddingDataset:
    """Maps (image_spec, caption_number) to cached text embedding indices."""

    def __init__(self, te_dataset, flattened_captions) -> None:
        self.te_dataset = te_dataset
        self.flattened_captions = flattened_captions
        self.image_spec_to_te_idx = defaultdict(list)
        for i, image_spec in enumerate(flattened_captions["image_spec"]):
            self.image_spec_to_te_idx[tuple(image_spec)].append(i)

    def get_text_embeddings(self, image_spec, caption_number):
        return self.te_dataset[
            self.image_spec_to_te_idx[image_spec][caption_number]
        ]


def _cache_text_embeddings(
    metadata_dataset,
    map_fn,
    i: int,
    cache_dir,
    regenerate_cache: bool,
    caching_batch_size: int,
    cache_num_proc: int | None = None,
    cache_keep_in_memory: bool = False,
    cache_format: str = "v2",
):
    """Flatten captions to one row per (image, caption), then map_and_cache."""
    from rengu_flow.data.cache_utils import _map_and_cache, content_fingerprint

    def flatten_captions(example):
        result = {key: [] for key in example}
        for idx, captions in enumerate(example["caption"]):
            for caption in captions:
                result["caption"].append(caption)
                for key, value in example.items():
                    if key == "caption":
                        continue
                    result[key].append(value[idx])
        return result

    flattened = metadata_dataset.map(
        flatten_captions,
        batched=True,
        keep_in_memory=cache_keep_in_memory,
        remove_columns=metadata_dataset.column_names,
    )
    # Text embeddings depend only on the caption text (and which encoder, via i); image_spec
    # keys the per-image lookup. Decoupling from the chained fingerprint means a latent-only
    # change never invalidates the text-embedding cache and vice versa.
    te_fp_override = content_fingerprint(
        flattened,
        [c for c in ("caption", "image_spec") if c in flattened.column_names],
    )
    te_dataset = _map_and_cache(
        flattened,
        map_fn,
        cache_dir,
        cache_file_prefix=f"text_embeddings_{i}_",
        new_fingerprint_args=[i],
        fingerprint_override=te_fp_override,
        regenerate_cache=regenerate_cache,
        caching_batch_size=caching_batch_size,
        num_proc=cache_num_proc,
        keep_in_memory=cache_keep_in_memory,
        cache_format=cache_format,
    )
    assert len(te_dataset) == len(flattened)
    return TextEmbeddingDataset(te_dataset, flattened)


def directory_subsample_ratio(directory_config: dict) -> float:
    """Fraction of a directory's images to use per epoch (diffusion-pipe ``subsample_ratio``)."""
    return float(directory_config.get("subsample_ratio", 1.0))


def trim_iteration_order_by_subsample_ratio(order, subsample_ratio: float):
    """Keep the first ``len * subsample_ratio`` rows of an iteration order (no-op when >= 1.0).

    The metadata is shuffled per bucket before the order is built, so this yields a stable
    pseudo-random subset.
    """
    if subsample_ratio >= 1.0:
        return order
    keep = int(len(order) * subsample_ratio)
    return order.select(range(keep))


def directory_max_images(directory_config: dict) -> int | None:
    """Absolute per-size-bucket image cap for a directory (``max_images``), or None for no cap."""
    value = directory_config.get("max_images")
    if value is None:
        return None
    return int(value)


def directory_subsample_shuffle(directory_config: dict) -> bool:
    """Whether the active sampler (``subsample_ratio`` or ``max_images``) rotates per epoch.

    ``True`` (default) advances the sampled window every epoch so the whole folder is
    eventually used; ``False`` keeps the same images every epoch (the behaviour of the
    retired ``static_sampling = true``).
    """
    if "static_sampling" in directory_config and is_main_process():
        print(
            "[data] 'static_sampling' was renamed; it is ignored. Use "
            "subsample_shuffle = false for the old static_sampling = true behaviour.",
            flush=True,
        )
    return bool(directory_config.get("subsample_shuffle", True))


def effective_sample_cap(
    pool_len: int, max_images: int | None, subsample_ratio: float
) -> int | None:
    """Per-epoch row count for a bucket, or ``None`` when the whole pool is used.

    ``max_images`` (absolute) and ``subsample_ratio`` (fraction) are two ways to limit how many
    images a folder contributes; they are mutually exclusive (enforced in dataset_config). The
    absolute cap wins if both somehow reach here. A ``subsample_ratio`` of ``1`` (or >= 1) means
    "no limit".
    """
    if max_images is not None:
        return max_images
    if subsample_ratio is not None and subsample_ratio < 1.0:
        return max(1, int(pool_len * subsample_ratio))
    return None


def rotation_window_index(
    pos: int, epoch: int, pool_len: int, cap: int | None, static: bool
) -> int:
    """Map a per-epoch slot ``pos`` to an index into the full (shuffled) pool.

    ``cap`` is the per-epoch row count from :func:`effective_sample_cap` (``None`` => no limit,
    behave as before). When ``static`` is False (default), the window start advances by ``cap``
    each epoch so consecutive epochs serve fresh images and cover the whole pool every
    ``ceil(pool_len/cap)`` epochs (wrapping). When ``static`` is True the same first ``cap`` rows
    are served every epoch. With ``cap > pool_len`` the modulo repeats the pool up to ``cap``
    (repeat-to-N). The result depends only on (pos, epoch, pool_len, cap, static), so it is
    deterministic and identical across data-parallel ranks (epoch is synced via
    ``PipelineDataLoader.sync_epoch``).
    """
    if pool_len <= 0:
        return 0
    if cap is None:
        return pos % pool_len
    offset = 0 if static else ((epoch - 1) * cap) % pool_len
    return (offset + pos) % pool_len


def uniform_caption_variants(caption_lists) -> int:
    """Captions-per-image when every image has the same count, else 1.

    Caption variants (one per .txt line) multiply the iteration order; the
    epoch accounting in main divides by this so an "epoch" still means one
    pass over the images, with variants rotating across appearances. A mixed
    dataset (unequal counts) cannot be divided out cleanly -> report 1.
    """
    counts = {len(c) for c in caption_lists}
    return counts.pop() if len(counts) == 1 else 1


def expand_caption_variants(
    captions: list[str],
    num_variants: int,
    tag_dropout: TagDropoutConfig,
    shuffle: bool,
    *,
    seed_key: str,
    delimiter: str = ", ",
) -> list[str]:
    """Bake ``num_variants`` tag-dropout/shuffle variants per base caption.

    Used at the text-embedding caching step so cached embeddings carry dropout/shuffle
    regularization (one per cached variant), rotated across epochs by the existing
    ``caption_number`` machinery. Deterministic and order-independent: each variant is
    seeded by (seed_key, base index, variant index, base caption), so the same config
    always yields the same strings — which keeps the text-embedding cache fingerprint
    (a content hash of the caption column) stable until a knob actually changes.

    Returns a flat list of ``len(captions) * num_variants``. With ``num_variants == 1``,
    dropout disabled and shuffle off this is the identity, so default configs are
    untouched.
    """
    num_variants = max(1, int(num_variants))
    if num_variants == 1 and not tag_dropout.enabled and not shuffle:
        return list(captions)
    out: list[str] = []
    for base_idx, caption in enumerate(captions):
        for variant_idx in range(num_variants):
            seed = int(
                hashlib.md5(
                    f"{seed_key}\x00{base_idx}\x00{variant_idx}\x00{caption}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                16,
            )
            rng = random.Random(seed)
            variant = apply_tag_dropout(caption, tag_dropout, rng, delimiter=delimiter)
            if shuffle:
                parts = split_tags(variant, delimiter)
                rng.shuffle(parts)
                variant = join_tags(parts, delimiter)
            out.append(variant)
    return out


class SizeBucketDataset:
    """Single size bucket from one directory: latents + text embeddings cache, iteration order."""

    def __init__(
        self,
        metadata_dataset,
        directory_config: dict,
        size_bucket: tuple,
        cache_base: Path,
        directory_dataset=None,
        resolution: int | None = None,
    ) -> None:
        # Per-bucket shuffle mixes multi-resolution training better (diffusion-pipe).
        metadata_dataset = metadata_dataset.shuffle(seed=seed_from_hash(size_bucket))
        self.metadata_dataset = metadata_dataset
        self._caption_variants = None
        self._caption_variants_expanded = False
        self.directory_config = directory_config
        self.size_bucket = size_bucket
        # Long-side resolution this bucket was generated for; used by the resolution
        # schedule to filter which buckets are sampled at a given training stage. For
        # AR buckets the exact source resolution is passed in; otherwise derive it from
        # the spatial dims (last element is the frame count).
        self.resolution = (
            int(resolution)
            if resolution is not None
            else int(round(math.sqrt(size_bucket[-3] * size_bucket[-2])))
        )
        self.path = Path(directory_config["path"])
        self.cache_dir = cache_base / f"cache_{bucket_suffix(size_bucket)}"
        self.captions_dict = (
            directory_dataset.captions_dict if directory_dataset is not None else None
        )
        self.uncond_fraction = float(
            getattr(directory_dataset, "uncond_fraction", 0.0)
        )
        self.tag_dropout = (
            getattr(directory_dataset, "tag_dropout", None) or TagDropoutConfig()
        )

        # Cached caption variants: when text embeddings are cached, bake K seeded
        # tag-dropout/shuffle variants per caption into the caption column. Both the
        # text-embedding cache (keyed by caption content) and the iteration order (which
        # rotates caption_number across epochs) read this same column, so the existing
        # multi-caption machinery handles the rest. Only meaningful for the cached path —
        # the live path applies tag dropout per sample in _sample_from_entry instead.
        # K = 1 with dropout/shuffle bakes a single fixed augmented variant for the whole
        # dataset (diffusion-pipe's default behaviour); K >= 2 gives variants that rotate.
        ds_cfg = getattr(directory_dataset, "dataset_config", None) or {}
        cache_text_embeddings = bool(
            getattr(directory_dataset, "caches_text_embeddings", False)
        )
        cached_variants = int(ds_cfg.get("cached_caption_variants", 1) or 1)
        cached_shuffle = bool(ds_cfg.get("cached_caption_shuffle", False))
        if cache_text_embeddings and (
            cached_variants > 1 or cached_shuffle or self.tag_dropout.enabled
        ):
            tag_dropout = self.tag_dropout
            if (
                cached_variants > 1
                and not tag_dropout.enabled
                and not cached_shuffle
                and is_main_process()
            ):
                print(
                    "[data] cached_caption_variants > 1 with no tag dropout and no "
                    "shuffle bakes identical copies (no regularization); set "
                    "tag_dropout_enabled or cached_caption_shuffle, or leave variants at 1.",
                    flush=True,
                )

            def _expand(example):
                return {
                    "caption": expand_caption_variants(
                        list(example["caption"]),
                        cached_variants,
                        tag_dropout,
                        cached_shuffle,
                        seed_key=str(example["image_spec"][-1]),
                    )
                }

            self.metadata_dataset = self.metadata_dataset.map(
                _expand,
                keep_in_memory=True,
                load_from_cache_file=False,
                desc="Expanding caption variants",
            )
            self._caption_variants_expanded = True

        if len(size_bucket) == 4:
            old_cache_dir = cache_base / f"cache_{bucket_suffix(size_bucket[1:])}"
            if old_cache_dir.exists() and not self.cache_dir.exists():
                old_cache_dir.rename(self.cache_dir)

        os.makedirs(self.cache_dir, exist_ok=True)
        self.text_embedding_datasets = []
        self.uncond_text_embeddings = []
        self.num_repeats = int(directory_config["num_repeats"])
        if self.num_repeats <= 0:
            raise ValueError(f"num_repeats must be >0, was {self.num_repeats}")
        self.max_images = directory_max_images(directory_config)
        if self.max_images is not None and self.max_images <= 0:
            raise ValueError(f"max_images must be >0, was {self.max_images}")
        self.subsample_ratio = directory_subsample_ratio(directory_config)
        self.subsample_shuffle = directory_subsample_shuffle(directory_config)
        self._epoch = 1
        # Per-epoch row order for the uncapped (whole-pool) case, so partial passes don't always
        # drop the same images (see _pool_index). Cached per epoch; deterministic per bucket.
        self._epoch_order_seed = seed_from_hash(("epoch_order", size_bucket))
        self._epoch_order_cache: list[int] | None = None
        self._epoch_order_for: int | None = None
        self._aug_fingerprint = getattr(directory_dataset, "_aug_fingerprint", "")

    @property
    def caption_variants(self) -> int:
        """Captions per image in this bucket (1 when images disagree)."""
        if self._caption_variants is None:
            self._caption_variants = uniform_caption_variants(
                self.metadata_dataset["caption"]
            )
        return self._caption_variants

    def cache_latents(
        self,
        map_fn,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        iteration_order_cache_dir = self.cache_dir / "iteration_order"
        latent_fp_args = [AUG_MVP_VERSION, self._aug_fingerprint]
        # Key the latent cache only on what actually determines a latent: the image, its
        # mask/control inputs and the size bucket. Captions live in a separate column and
        # never affect latents, so excluding them keeps caption edits/shuffles from
        # invalidating the (expensive) VAE cache. Augmentation is folded in via
        # latent_fp_args (aug_fingerprint) and via image_spec variant keys.
        latent_fp_override = content_fingerprint(
            self.metadata_dataset,
            [
                c
                for c in ("image_spec", "mask_file", "size_bucket", "is_video", "control_file")
                if c in self.metadata_dataset.column_names
            ],
        )
        if map_fn is None:
            self.latent_dataset = _map_and_cache(
                self.metadata_dataset,
                None,
                self.cache_dir,
                cache_file_prefix="latents_",
                new_fingerprint_args=latent_fp_args,
                fingerprint_override=latent_fp_override,
                regenerate_cache=False,
                caching_batch_size=caching_batch_size,
                num_proc=cache_num_proc,
                keep_in_memory=cache_keep_in_memory,
                cache_format=cache_format,
            )
            self.iteration_order = datasets.load_from_disk(
                str(iteration_order_cache_dir)
            )
            return

        print(f"caching latents: {self.size_bucket}")
        self.latent_dataset = _map_and_cache(
            self.metadata_dataset,
            map_fn,
            self.cache_dir,
            cache_file_prefix="latents_",
            new_fingerprint_args=latent_fp_args,
            fingerprint_override=latent_fp_override,
            regenerate_cache=regenerate_cache,
            caching_batch_size=caching_batch_size,
            num_proc=cache_num_proc,
            keep_in_memory=cache_keep_in_memory,
            cache_format=cache_format,
        )
        assert len(self.latent_dataset) == len(self.metadata_dataset)

        # The iteration order embeds the caption strings and their caption_number slots, so
        # it must rebuild whenever the captions change — including when cached caption
        # variants are (re)baked. trust_cache only guards the existence check, so key the
        # rebuild on a content hash of the caption column stored in a sidecar file.
        caption_fp = content_fingerprint(
            self.metadata_dataset,
            [
                c
                for c in ("caption", "image_spec")
                if c in self.metadata_dataset.column_names
            ],
        )
        caption_fp_file = self.cache_dir / "iteration_order.caption_fp"
        caption_fp_stale = (
            not caption_fp_file.exists() or caption_fp_file.read_text() != caption_fp
        )
        if (
            regenerate_cache
            or not iteration_order_cache_dir.exists()
            or not trust_cache
            or caption_fp_stale
        ):
            print("Building iteration order")
            image_spec_to_latents_idx = {
                tuple(self.metadata_dataset[i]["image_spec"]): i
                for i in range(len(self.metadata_dataset))
            }

            equal_num_captions = True
            num_captions = None
            for idx in range(len(self.metadata_dataset)):
                n = len(self.metadata_dataset[idx]["caption"])
                if num_captions is not None and n != num_captions:
                    equal_num_captions = False
                    break
                num_captions = n

            if equal_num_captions and num_captions is not None:
                by_caption_num = [[] for _ in range(num_captions)]
                seed = 0
                for idx in range(len(self.metadata_dataset)):
                    example = self.metadata_dataset[idx]
                    image_spec = example["image_spec"]
                    captions = list(example["caption"])
                    shuffle_with_seed(captions, seed)
                    seed += 1
                    latents_idx = image_spec_to_latents_idx[tuple(image_spec)]
                    for i, caption in enumerate(captions):
                        by_caption_num[i].append(
                            (image_spec, latents_idx, caption, i)
                        )
                iteration_order_list = []
                for lst in by_caption_num:
                    iteration_order_list.extend(lst)
            else:
                iteration_order_list = []
                for idx in range(len(self.metadata_dataset)):
                    example = self.metadata_dataset[idx]
                    image_spec = example["image_spec"]
                    captions = example["caption"]
                    latents_idx = image_spec_to_latents_idx[tuple(image_spec)]
                    for i, caption in enumerate(captions):
                        iteration_order_list.append(
                            (image_spec, latents_idx, caption, i)
                        )
                shuffle_with_seed(iteration_order_list, 42)

            iteration_order_dict = defaultdict(list)
            for image_spec, latents_idx, caption, caption_number in iteration_order_list:
                iteration_order_dict["image_spec"].append(image_spec)
                iteration_order_dict["latents_idx"].append(latents_idx)
                iteration_order_dict["caption"].append(caption)
                iteration_order_dict["caption_number"].append(caption_number)
            iteration_order = datasets.Dataset.from_dict(iteration_order_dict)
            # The full pool is cached; subsample_ratio/max_images are applied per epoch at access
            # time (see _effective_len / rotation_window_index) so the window can rotate.
            iteration_order.save_to_disk(str(iteration_order_cache_dir))
            caption_fp_file.write_text(caption_fp)

        self.iteration_order = datasets.load_from_disk(
            str(iteration_order_cache_dir)
        )

    def cache_text_embeddings(
        self,
        map_fn,
        i: int,
        regenerate_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        print(f"caching text embeddings: {self.size_bucket}")
        te_dataset = _cache_text_embeddings(
            self.metadata_dataset,
            map_fn,
            i,
            self.cache_dir,
            regenerate_cache,
            caching_batch_size,
            cache_num_proc=cache_num_proc,
            cache_keep_in_memory=cache_keep_in_memory,
            cache_format=cache_format,
        )
        self.text_embedding_datasets.append(te_dataset)

    def add_text_embedding_dataset(self, te_dataset) -> None:
        self.text_embedding_datasets.append(te_dataset)

    def _sample_from_entry(self, entry, latent_dict: dict | None = None) -> dict:
        ret = dict(latent_dict if latent_dict is not None else self.latent_dataset[entry["latents_idx"]])
        use_uncond = (
            self.uncond_fraction > 0 and random.random() < self.uncond_fraction
        )
        if use_uncond:
            caption = ""
        elif self.captions_dict and not self._caption_variants_expanded:
            spec = entry["image_spec"]
            key = spec[-1]
            if key in self.captions_dict:
                caption = self.captions_dict[key][entry["caption_number"]]
            else:
                logger.warning(
                    "Image %s missing from captions_dict; using empty caption.",
                    key,
                )
                caption = ""
        else:
            # entry["caption"] is the baked variant when cached caption variants are active,
            # so caption_number resolves to it directly (captions_dict holds only base lines).
            caption = entry["caption"]
        # When variants are pre-baked, dropout is already in the cached embedding/caption;
        # re-applying it live would diverge the returned string from the cached embedding.
        if (
            not use_uncond
            and self.tag_dropout.enabled
            and not self._caption_variants_expanded
        ):
            caption = apply_tag_dropout(caption, self.tag_dropout, random)
        for ds, uncond_ds in zip(
            self.text_embedding_datasets, self.uncond_text_embeddings
        ):
            emb_dict = (
                uncond_ds[0]
                if use_uncond
                else ds.get_text_embeddings(
                    tuple(entry["image_spec"]), entry["caption_number"]
                )
            )
            ret.update(emb_dict)
        ret["caption"] = caption
        return ret

    @property
    def _pool_len(self) -> int:
        """Number of rows actually available in this bucket (the full pool to rotate over)."""
        return len(self.iteration_order)

    @property
    def _sample_cap(self) -> int | None:
        """Per-epoch row count from max_images or subsample_ratio (None => whole pool)."""
        return effective_sample_cap(
            self._pool_len, self.max_images, self.subsample_ratio
        )

    @property
    def _effective_len(self) -> int:
        """Rows served per epoch: the active sampler's cap, or the whole pool when uncapped."""
        cap = self._sample_cap
        return self._pool_len if cap is None else cap

    def _pool_index(self, idx: int) -> int:
        """Map an upper-layer index (0..len-1) to a row of the full pool, honoring rotation."""
        m = self._effective_len
        if m <= 0:
            return 0
        pos = idx % m
        cap = self._sample_cap
        if cap is None and self.subsample_shuffle:
            # Whole pool served every epoch: reshuffle the row order per epoch (RandomCursor) so
            # the partial passes at schedule-stage boundaries / run end don't always drop the same
            # tail. Coverage is preserved (it's a full permutation); a capped pool keeps its own
            # coverage-guaranteeing window rotation below.
            return self._epoch_pool_order()[pos]
        return rotation_window_index(
            pos, self._epoch, self._pool_len, cap, not self.subsample_shuffle
        )

    def _epoch_pool_order(self) -> list[int]:
        """Cached per-epoch permutation of the whole pool (built once per epoch)."""
        if self._epoch_order_for != self._epoch or self._epoch_order_cache is None:
            from rengu_flow.data.sampling import RandomCursor

            self._epoch_order_for = self._epoch
            self._epoch_order_cache = RandomCursor(
                self._pool_len, seed=self._epoch_order_seed
            ).order(self._epoch)
        return self._epoch_order_cache

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch so a non-static sampler rotates its window."""
        self._epoch = int(epoch)

    def get_items_batch(self, idx_list: list[int]) -> list[dict]:
        """Load multiple training samples; batches latent cache reads per shard."""
        entries = []
        for idx in idx_list:
            entries.append(self.iteration_order[self._pool_index(idx)])
        latent_idxs = [e["latents_idx"] for e in entries]
        latent_dicts = self.latent_dataset.get_many(latent_idxs)
        return [
            self._sample_from_entry(entry, latent_dicts[i])
            for i, entry in enumerate(entries)
        ]

    def __getitem__(self, idx):
        entry = self.iteration_order[self._pool_index(idx)]
        return self._sample_from_entry(entry)

    def __len__(self) -> int:
        return int(self._effective_len * self.num_repeats)


class ConcatenatedBatchedDataset:
    """Concatenation of multiple SizeBucketDatasets (same size bucket); returns batches."""

    def __init__(self, datasets_list: list) -> None:
        self.datasets = datasets_list
        self.post_init_called = False

    def post_init(
        self,
        global_batch_size: dict,
        global_batch_size_image: dict,
        data_parallel_rank: int,
        data_parallel_world_size: int,
    ) -> None:
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        size_bucket = self.datasets[0].size_bucket
        # All datasets in this concat share the same size bucket -> same resolution.
        self.resolution = self.datasets[0].resolution
        iteration_order = []
        for i, ds in enumerate(self.datasets):
            assert ds.size_bucket == size_bucket
            iteration_order.extend([i] * len(ds))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.datasets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = np.array(iteration_order)

        gbs_dict = (
            global_batch_size_image
            if size_bucket[-1] == 1
            else global_batch_size
        )
        if None in gbs_dict:
            self.global_batch_size = gbs_dict[None]
        else:
            bucket_size = math.sqrt(size_bucket[-2] * size_bucket[-3])
            min_diff = float("inf")
            for size, bs in gbs_dict.items():
                diff = abs(size - bucket_size)
                if diff < min_diff:
                    min_diff = diff
                    self.global_batch_size = bs

        assert self.global_batch_size % self.data_parallel_world_size == 0
        self._make_divisible_by(self.global_batch_size)
        self.batch_size = self.global_batch_size // self.data_parallel_world_size
        self.post_init_called = True

    def _make_divisible_by(self, n: int) -> None:
        new_length = (len(self.iteration_order) // n) * n
        self.iteration_order = self.iteration_order[:new_length]
        if new_length == 0 and is_main_process():
            logger.warning(
                "size bucket %s is being completely dropped (not enough images)",
                self.datasets[0].size_bucket,
            )

    def __len__(self) -> int:
        assert self.post_init_called
        return len(self.iteration_order) // self.global_batch_size

    def set_epoch(self, epoch: int) -> None:
        for ds in self.datasets:
            ds.set_epoch(epoch)

    def __getitem__(self, idx):
        assert self.post_init_called
        start_idx = (
            idx * self.global_batch_size
            + self.data_parallel_rank * self.batch_size
        )
        end_idx = start_idx + self.batch_size
        if self.batch_size > 1:
            ds_ids = [int(self.iteration_order[k][0]) for k in range(start_idx, end_idx)]
            if len(set(ds_ids)) == 1:
                ds = self.datasets[ds_ids[0]]
                inner = [int(self.iteration_order[k][1]) for k in range(start_idx, end_idx)]
                if hasattr(ds, "get_items_batch"):
                    return ds.get_items_batch(inner)
        return [
            self.datasets[int(self.iteration_order[k][0])][
                int(self.iteration_order[k][1])
            ]
            for k in range(start_idx, end_idx)
        ]


class ARBucketDataset:
    """AR bucket: creates SizeBucketDatasets per resolution, caches latents and text embeddings."""

    def __init__(
        self,
        ar_frames: tuple,
        resolutions: np.ndarray,
        metadata_dataset,
        directory_config: dict,
        cache_base: Path,
        round_to_multiple: int,
        directory_dataset=None,
    ) -> None:
        self.ar_frames = ar_frames
        self.resolutions = resolutions
        self.metadata_dataset = metadata_dataset
        self.directory_config = directory_config
        self.size_buckets = []
        self.path = Path(directory_config["path"])
        self.cache_base = cache_base
        self.cache_dir = cache_base / f"ar_frames_{bucket_suffix(ar_frames)}"
        self.round_to_multiple = round_to_multiple
        self.directory_dataset = directory_dataset
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_size_bucket_datasets(self) -> list:
        return self.size_buckets

    def cache_latents(
        self,
        map_fn,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        print(f"caching latents: {self.ar_frames}")
        for res in self.resolutions:
            area = res**2
            w = math.sqrt(area * self.ar_frames[0])
            h = area / w
            w = round_to_nearest_multiple(w, self.round_to_multiple)
            h = round_to_nearest_multiple(h, self.round_to_multiple)
            size_bucket = (w, h, self.ar_frames[1])
            naming_size_bucket = (self.ar_frames[0],) + size_bucket
            metadata_with_size = self.metadata_dataset.map(
                lambda ex: {"size_bucket": size_bucket},
                cache_file_name=str(
                    self.cache_dir
                    / f"metadata/metadata_{bucket_suffix(naming_size_bucket)}.arrow"
                ),
                load_from_cache_file=(not regenerate_cache and trust_cache),
                desc="Adding size bucket",
            )
            self.size_buckets.append(
                SizeBucketDataset(
                    metadata_with_size,
                    self.directory_config,
                    naming_size_bucket,
                    self.cache_base,
                    self.directory_dataset,
                    resolution=int(res),
                )
            )
        for ds in self.size_buckets:
            ds.cache_latents(
                map_fn,
                regenerate_cache=regenerate_cache,
                trust_cache=trust_cache,
                caching_batch_size=caching_batch_size,
                cache_num_proc=cache_num_proc,
                cache_keep_in_memory=cache_keep_in_memory,
                cache_format=cache_format,
            )

    def cache_text_embeddings(
        self,
        map_fn,
        i: int,
        regenerate_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        print(f"caching text embeddings: {self.ar_frames}")
        te_dataset = _cache_text_embeddings(
            self.metadata_dataset,
            map_fn,
            i,
            self.cache_dir,
            regenerate_cache,
            caching_batch_size,
            cache_num_proc=cache_num_proc,
            cache_keep_in_memory=cache_keep_in_memory,
            cache_format=cache_format,
        )
        for sb in self.size_buckets:
            sb.add_text_embedding_dataset(te_dataset)


class DirectoryDataset:
    """One directory of media files: metadata, AR/size buckets, cache latents and text embeddings."""

    def __init__(
        self,
        directory_config: dict,
        dataset_config: dict,
        model_name: str,
        framerate: float | None = None,
        round_to_multiple: int = 32,
        skip_dataset_validation: bool = False,
        cache_text_embeddings: bool = False,
    ) -> None:
        self._set_defaults(directory_config, dataset_config)
        self.directory_config = directory_config
        self.dataset_config = dataset_config
        # Whether the model caches text embeddings (drives cached caption-variant baking
        # in SizeBucketDataset; the live path applies tag dropout per sample instead).
        # Note: distinct name from the cache_text_embeddings() method to avoid shadowing it.
        self.caches_text_embeddings = cache_text_embeddings
        if skip_dataset_validation:
            from rengu_flow.data.augmentation import resolve_augmentation_config

            self._resolved_augmentation = resolve_augmentation_config(
                directory_config, dataset_config
            )
        else:
            self._resolved_augmentation = validate_augmentation_for_directory(
                directory_config, dataset_config
            )
        self._aug_fingerprint = augmentation_fingerprint(self._resolved_augmentation)
        self._aug_enabled = is_augmentation_enabled(self._resolved_augmentation)
        self._variant_keys = (
            expand_variant_keys(self._resolved_augmentation)
            if self._aug_enabled
            else [None]
        )
        if not skip_dataset_validation:
            self._validate()
        self.model_name = model_name
        self.framerate = framerate
        self.round_to_multiple = round_to_multiple
        self.enable_ar_bucket = directory_config.get(
            "enable_ar_bucket",
            dataset_config.get("enable_ar_bucket", False),
        )
        self.size_buckets_config = directory_config.get(
            "size_buckets", dataset_config.get("size_buckets")
        )
        self.use_size_buckets = self.size_buckets_config is not None
        if self.use_size_buckets:
            self.size_buckets_config = sorted(
                self.size_buckets_config, key=lambda t: t[-1], reverse=True
            )
            self.size_buckets_config = np.array(self.size_buckets_config)
            self.size_bucket_datasets = []
        else:
            res = directory_config.get(
                "resolutions", dataset_config["resolutions"]
            )
            self.resolutions = dedup_and_sort(
                self._process_user_provided_resolutions(res)
            )
            self.ar_bucket_datasets = []

        # Cache-time tag shuffling was retired: caption multiplication is governed
        # only by .txt lines (caption variants). Bake shuffled/dropped variants with
        # scripts/generate_caption_variants.py (--shuffle-tags) instead.
        for _retired in ("shuffle_tags", "cache_shuffle_num", "cache_shuffle_delimiter"):
            if (
                directory_config.get(_retired, dataset_config.get(_retired)) not in (None, False)
                and is_main_process()
            ):
                print(
                    f"[data] '{_retired}' was retired and is ignored; pre-bake shuffled "
                    "caption variants with scripts/generate_caption_variants.py "
                    "--shuffle-tags (cached, rotates across epochs).",
                    flush=True,
                )
        self.shuffle_metadata = directory_config["shuffle_metadata"]
        self.path = Path(self.directory_config["path"])
        self.mask_path = (
            Path(self.directory_config["mask_path"])
            if "mask_path" in self.directory_config
            else None
        )
        self.control_path = (
            Path(self.directory_config["control_path"])
            if "control_path" in self.directory_config
            else None
        )
        self.default_mask_file = (
            Path(self.directory_config["default_mask_file"])
            if "default_mask_file" in self.directory_config
            else None
        )
        self.cache_dir = self.path / "cache" / self.model_name
        self.grouping_keys_json_file = (
            self.cache_dir / "metadata/grouping_keys.json"
        )

        if not self.path.exists() or not self.path.is_dir():
            raise RuntimeError(f"Invalid path: {self.path}")
        if self.mask_path is not None and (
            not self.mask_path.exists() or not self.mask_path.is_dir()
        ):
            raise RuntimeError(f"Invalid mask_path: {self.mask_path}")
        if self.control_path is not None and (
            not self.control_path.exists() or not self.control_path.is_dir()
        ):
            raise RuntimeError(f"Invalid control_path: {self.control_path}")
        if self.default_mask_file is not None and (
            not self.default_mask_file.exists()
            or not self.default_mask_file.is_file()
        ):
            raise RuntimeError(
                f"Invalid default_mask_file: {self.default_mask_file}"
            )

        if self.use_size_buckets:
            self.ars = np.array(
                [w / h for w, h, _ in self.size_buckets_config]
            )
        elif not self.enable_ar_bucket:
            self.ars = np.array([1.0])
        elif directory_config.get("ar_buckets") or dataset_config.get(
            "ar_buckets"
        ):
            ars = directory_config.get(
                "ar_buckets", dataset_config.get("ar_buckets")
            )
            self.ars = self._process_user_provided_ars(ars)
        else:
            # Fall back to the same defaults the UI schema advertises (see
            # rengu_flow_ui/dataset_schema.py) so enable_ar_bucket works even when a config
            # turns it on without spelling out min_ar/max_ar/num_ar_buckets and provides no
            # explicit ar_buckets. Direct key access here used to raise KeyError instead.
            min_ar = directory_config.get(
                "min_ar", dataset_config.get("min_ar", 0.5)
            )
            max_ar = directory_config.get(
                "max_ar", dataset_config.get("max_ar", 2.0)
            )
            num_ar = directory_config.get(
                "num_ar_buckets", dataset_config.get("num_ar_buckets", 12)
            )
            self.ars = np.geomspace(min_ar, max_ar, num=num_ar)
        self.ars = dedup_and_sort(self.ars)
        self.log_ars = np.log(self.ars)
        frame_buckets = directory_config.get(
            "frame_buckets", dataset_config.get("frame_buckets", [1])
        )
        if 1 not in frame_buckets:
            frame_buckets = list(frame_buckets) + [1]
        frame_buckets.sort()
        self.frame_buckets = np.array(frame_buckets)

        online_captions = directory_config.get(
            "online_captions", dataset_config.get("online_captions", False)
        )
        if online_captions:
            captions_json = self.path / CAPTIONS_JSON_FILE
            if not captions_json.exists():
                raise FileNotFoundError(
                    f"online_captions requires {CAPTIONS_JSON_FILE} in {self.path}"
                )
            with open(captions_json) as f:
                self.captions_dict = json.load(f)
        else:
            self.captions_dict = None

        self.uncond_fraction = float(
            directory_config.get(
                "uncond_fraction", dataset_config.get("uncond_fraction", 0.0)
            )
            or 0.0
        )
        self.tag_dropout = build_tag_dropout_config(
            directory_config, dataset_config, tags_file_base=self.path
        )

    def _set_defaults(
        self, directory_config: dict, dataset_config: dict
    ) -> None:
        directory_config.setdefault(
            "enable_ar_bucket",
            dataset_config.get("enable_ar_bucket", False),
        )
        directory_config.setdefault(
            "directory_caption", dataset_config.get("directory_caption", "")
        )
        directory_config.setdefault(
            "num_repeats", dataset_config.get("num_repeats", 1)
        )
        directory_config.setdefault(
            "shuffle_metadata", dataset_config.get("shuffle_metadata", True)
        )
        directory_config.setdefault(
            "online_captions", dataset_config.get("online_captions", False)
        )
        # Inherit the global max_images cap, but not onto folders that already pick the other
        # (mutually exclusive) sampler via an explicit subsample_ratio < 1.
        try:
            explicit_ratio = float(directory_config.get("subsample_ratio", 1.0))
        except (TypeError, ValueError):
            explicit_ratio = 1.0
        if explicit_ratio >= 1.0 and dataset_config.get("max_images") is not None:
            directory_config.setdefault("max_images", dataset_config.get("max_images"))
        if "static_sampling" in dataset_config and is_main_process():
            print(
                "[data] 'static_sampling' was renamed; it is ignored. Use "
                "subsample_shuffle = false for the old static_sampling = true behaviour.",
                flush=True,
            )
        directory_config.setdefault(
            "subsample_shuffle", dataset_config.get("subsample_shuffle", True)
        )

    def _validate(self) -> None:
        res = self.directory_config.get(
            "resolutions", self.dataset_config.get("resolutions", [])
        )
        if len(res) > 3 and is_main_process():
            logger.warning(
                "Many resolutions set in dataset config; ensure you understand the effect."
            )

    def cache_metadata(
        self,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        cache_num_proc: int | None = None,
    ) -> None:
        def check_grouped():
            if not self.grouping_keys_json_file.exists():
                return False, None
            with open(self.grouping_keys_json_file) as f:
                keys = json.load(f)
            if self.use_size_buckets and not all(
                len(k) == 3 for k in keys
            ):
                return False, keys
            if not self.use_size_buckets and not all(
                len(k) == 2 for k in keys
            ):
                return False, keys
            all_exist = all(
                (
                    self.cache_dir / f"metadata/grouped_metadata_{bucket_suffix(k)}"
                ).exists()
                for k in keys
            )
            return all_exist, keys

        all_exist, unique_keys = check_grouped()
        if regenerate_cache or not all_exist or not trust_cache:
            print(
                "Grouped metadata is not cached. Computing ungrouped metadata and grouping."
            )
            unique_keys = self._group_metadata_and_save_to_disk(
                regenerate_cache=regenerate_cache,
                trust_cache=trust_cache,
                cache_num_proc=cache_num_proc,
            )
        else:
            print("Found grouped metadata cache. Directly loading it.")

        for key in unique_keys:
            grouped_dir = (
                self.cache_dir
                / f"metadata/grouped_metadata_{bucket_suffix(key)}"
            )
            print(f"Loading grouped metadata with grouping key {key}")
            metadata = datasets.load_from_disk(str(grouped_dir))
            if self.use_size_buckets:
                assert len(key) == 3
                self.size_bucket_datasets.append(
                    SizeBucketDataset(
                        metadata,
                        self.directory_config,
                        key,
                        self.cache_dir,
                        self,
                    )
                )
            else:
                self.ar_bucket_datasets.append(
                    ARBucketDataset(
                        key,
                        self.resolutions,
                        metadata,
                        self.directory_config,
                        self.cache_dir,
                        self.round_to_multiple,
                        self,
                    )
                )

    def _group_metadata_and_save_to_disk(
        self,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        cache_num_proc: int | None = None,
    ) -> list:
        metadata_dataset = self._get_ungrouped_metadata(
            regenerate_cache=regenerate_cache,
            trust_cache=trust_cache,
            cache_num_proc=cache_num_proc,
        )
        grouped = defaultdict(lambda: defaultdict(list))
        unique_keys = set()
        for idx in range(len(metadata_dataset)):
            example = metadata_dataset[idx]
            if self.use_size_buckets:
                key = tuple(example["size_bucket"])
            else:
                key = example["ar_bucket"]
                key = (key[0], int(key[1]))
            unique_keys.add(key)
            for k, v in example.items():
                grouped[key][k].append(v)
        unique_keys = list(unique_keys)
        for key, data in grouped.items():
            ds = datasets.Dataset.from_dict(data)
            path = self.cache_dir / f"metadata/grouped_metadata_{bucket_suffix(key)}"
            ds.save_to_disk(str(path))
        with open(self.grouping_keys_json_file, "w") as f:
            json.dump(unique_keys, f)
        return unique_keys

    def _get_ungrouped_metadata(
        self,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        cache_num_proc: int | None = None,
    ):
        metadata_num_proc = resolve_cache_num_proc(cache_num_proc)
        metadata_cache_1 = self.cache_dir / "metadata/metadata_intermediate"
        metadata_cache_2 = self.cache_dir / "metadata/metadata.arrow"

        if (
            regenerate_cache
            or not metadata_cache_1.exists()
            or not trust_cache
        ):
            print("Intermediate metadata is not cached. Enumerating all files.")
            files = sorted(self.path.glob("*"))

            mask_stems = {}
            if self.mask_path is not None:
                mask_stems = {
                    p.stem: p
                    for p in self.mask_path.glob("*")
                    if p.is_file()
                }
            control_stems = {}
            if self.control_path is not None:
                control_stems = {
                    p.stem: p
                    for p in self.control_path.glob("*")
                    if p.is_file()
                }

            def process_file(file):
                if file.suffix != ".tar":
                    return [(None, str(file))]
                with tarfile.open(file) as tar_f:
                    return [(str(file), n) for n in tar_f.getnames()]

            captions_json = self.path / CAPTIONS_JSON_FILE
            has_captions_json = captions_json.exists()

            image_specs = []
            caption_files = []
            mask_files = []
            control_files = []
            for file in tqdm(files, disable=not sys.stderr.isatty()):
                if (
                    not file.is_file()
                    or file.suffix in (".txt", ".npz", ".json", ".parquet", ".bak")
                ):
                    continue
                for image_spec in process_file(file):
                    image_file = Path(image_spec[1])
                    caption_file = image_file.with_suffix(".txt")
                    if has_captions_json or not caption_file.exists():
                        caption_file = ""
                    else:
                        caption_file = str(caption_file)
                    image_specs.append(image_spec)
                    caption_files.append(caption_file)
                    if image_file.stem in mask_stems:
                        mask_files.append(str(mask_stems[image_file.stem]))
                    elif self.default_mask_file is not None:
                        mask_files.append(str(self.default_mask_file))
                    else:
                        if self.mask_path is not None:
                            logger.warning(
                                "No mask file for %s, not using mask.",
                                image_file,
                            )
                        mask_files.append(None)
                    if self.control_path is not None:
                        if image_file.stem not in control_stems:
                            raise RuntimeError(
                                f"No control file for image {image_file}"
                            )
                        control_files.append(
                            str(control_stems[image_file.stem])
                        )

            if len(image_specs) == 0:
                raise RuntimeError(
                    f"Directory {self.path} had no images/videos!"
                )

            d = {
                "image_spec": image_specs,
                "caption_file": caption_files,
                "mask_file": mask_files,
            }
            if self.control_path:
                d["control_file"] = control_files
            metadata_dataset = datasets.Dataset.from_dict(d)

            if captions_json.exists():
                print("Loading captions JSON")
                with open(captions_json) as f:
                    caption_data = json.load(f)

                def add_captions(example):
                    tar_file, image_file = example["image_spec"]
                    if tar_file is None:
                        image_file = image_file.split("/")[-1]
                    captions = caption_data.get(image_file)
                    if captions is None:
                        logger.warning(
                            "Image %s not in captions.json",
                            image_file,
                        )
                    else:
                        assert isinstance(
                            captions, list
                        ), "captions.json must contain lists of captions"
                    return {"caption": captions if captions is not None else [""]}

                metadata_dataset = metadata_dataset.map(
                    add_captions,
                    cache_file_name=str(
                        self.cache_dir
                        / "metadata/metadata_with_captions.arrow"
                    ),
                    load_from_cache_file=(not regenerate_cache and trust_cache),
                    desc="Adding captions",
                )

            seed = seed_from_hash(self.path)
            if self.shuffle_metadata:
                metadata_dataset = metadata_dataset.shuffle(seed=seed)
            print("Saving intermediate metadata dataset.")
            metadata_dataset.save_to_disk(str(metadata_cache_1))
            del metadata_dataset

        print("Loading intermediate metadata dataset.")
        metadata_dataset = datasets.load_from_disk(str(metadata_cache_1))
        metadata_map_fn, tarfile_map = self._metadata_map_fn()
        print("Caching ungrouped metadata.")
        try:
            metadata_dataset = metadata_dataset.map(
                metadata_map_fn,
                cache_file_name=str(metadata_cache_2),
                load_from_cache_file=(not regenerate_cache and trust_cache),
                batched=True,
                batch_size=1,
                num_proc=metadata_num_proc,
                remove_columns=metadata_dataset.column_names,
            )
        finally:
            # Close any tar handles opened during the in-process map so we don't leak FDs
            # across cache builds. With num_proc > 1 the map runs in forked workers whose
            # handles are reclaimed on process exit, so this covers the in-process path.
            for tar_f in tarfile_map.values():
                tar_f.close()
            tarfile_map.clear()
        return metadata_dataset

    def _metadata_map_fn(self):
        tarfile_map = {}

        def fn(example):
            caption_file = example["caption_file"][0]
            image_spec = example["image_spec"][0]
            image_file = Path(image_spec[1])
            captions = None
            if "caption" in example:
                captions = example["caption"][0]
            if captions is None and caption_file:
                # rengu-flow: .txt = one caption per line
                captions = _read_captions_from_txt_per_line(caption_file)
            if captions is None:
                # Fallback: directory_caption or empty
                directory_caption = self.directory_config.get(
                    "directory_caption"
                )
                if directory_caption is not None:
                    captions = [directory_caption]
                else:
                    captions = [""]
                    logger.warning(
                        "No caption for %s; using empty caption.",
                        image_file,
                    )
            prefix = self.directory_config.get("directory_caption", "")
            captions = [prefix + c for c in captions]
            empty_return = {
                "image_spec": [],
                "mask_file": [],
                "caption": [],
                "ar_bucket": [],
                "size_bucket": [],
                "is_video": [],
            }
            if self.control_path:
                empty_return["control_file"] = []

            if image_spec[0] is None:
                filepath_or_file = str(image_file)
            else:
                tar_filename = image_spec[0]
                if tar_filename not in tarfile_map:
                    tarfile_map[tar_filename] = tarfile.open(tar_filename)
                tar_f = tarfile_map[tar_filename]
                filepath_or_file = tar_f.extractfile(str(image_file))

            if image_file.suffix == ".webp":
                reader = imageio.get_reader(filepath_or_file)
                if reader.get_length() > 1:
                    raise NotImplementedError(
                        "WebP videos are not supported."
                    )
            try:
                if image_file.suffix in VIDEO_EXTENSIONS:
                    meta = imageio.v3.immeta(filepath_or_file)
                    first_frame = next(imageio.v3.imiter(filepath_or_file))
                    height, width = first_frame.shape[:2]
                    if self.framerate is None:
                        raise RuntimeError(
                            "Model framerate required for video."
                        )
                    frames = int(self.framerate * meta["duration"])
                else:
                    pil_img = Image.open(filepath_or_file)
                    width, height = pil_img.size
                    frames = 1
            except Exception as e:
                logger.warning(
                    "Could not open %s: %s. Skipping.", image_file, e
                )
                return empty_return
            finally:
                if hasattr(filepath_or_file, "close"):
                    filepath_or_file.close()

            is_video = frames > 1
            log_ar = np.log(width / height)

            if self.use_size_buckets:
                size_bucket = self._find_closest_size_bucket(
                    log_ar, frames, is_video
                )
                if size_bucket is None:
                    return empty_return
                ar_bucket = None
            else:
                ar_bucket = self._find_closest_ar_bucket(
                    log_ar, frames, is_video
                )
                if ar_bucket is None:
                    return empty_return
                size_bucket = None

            if is_video and self._aug_enabled:
                raise RuntimeError(
                    f"Augmentation is enabled for {self.path} but {image_file} is video; "
                    "not supported in this release."
                )

            variant_keys = self._variant_keys
            ret = {
                "image_spec": [],
                "mask_file": [],
                "caption": [],
                "ar_bucket": [],
                "size_bucket": [],
                "is_video": [],
            }
            if self.control_path:
                ret["control_file"] = []
            for vk in variant_keys:
                ret["image_spec"].append(with_variant_key(image_spec, vk))
                ret["mask_file"].append(example["mask_file"][0])
                ret["caption"].append(captions)
                ret["ar_bucket"].append(ar_bucket)
                ret["size_bucket"].append(size_bucket)
                ret["is_video"].append(is_video)
                if self.control_path:
                    ret["control_file"].append(example["control_file"][0])
            return ret

        return fn, tarfile_map

    def _find_closest_ar_bucket(self, log_ar, frames, is_video):
        i = np.argmin(np.abs(log_ar - self.log_ars))
        diffs = frames - self.frame_buckets
        positive_diffs = diffs[diffs >= 0]
        if len(positive_diffs) == 0:
            return None
        j = np.argmin(positive_diffs)
        if is_video and self.frame_buckets[j] == 1:
            return None
        return (self.ars[i], self.frame_buckets[j])

    def _find_closest_size_bucket(self, log_ar, frames, is_video):
        ar_diffs = np.abs(log_ar - self.log_ars)
        candidate = self.size_buckets_config[
            np.argsort(ar_diffs, kind="stable")
        ]
        for size_bucket in candidate:
            if is_video and size_bucket[-1] == 1:
                continue
            if frames >= size_bucket[-1]:
                return tuple(size_bucket)
        return None

    def _process_user_provided_ars(self, ars) -> np.ndarray:
        out = []
        for ar in ars:
            if isinstance(ar, (tuple, list)):
                ar = ar[0] / ar[1]
            out.append(ar)
        return np.array(out)

    def _process_user_provided_resolutions(self, resolutions) -> list:
        out = []
        for res in resolutions:
            if isinstance(res, (tuple, list)):
                res = math.sqrt(res[0] * res[1])
            out.append(res)
        return out

    def get_size_bucket_datasets(self) -> list:
        if self.use_size_buckets:
            return self.size_bucket_datasets
        result = []
        for ar_ds in self.ar_bucket_datasets:
            result.extend(ar_ds.get_size_bucket_datasets())
        return result

    def cache_latents(
        self,
        map_fn,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        print(f"caching latents: {self.path}")
        datasets_list = (
            self.size_bucket_datasets
            if self.use_size_buckets
            else self.ar_bucket_datasets
        )
        for ds in datasets_list:
            ds.cache_latents(
                map_fn,
                regenerate_cache=regenerate_cache,
                trust_cache=trust_cache,
                caching_batch_size=caching_batch_size,
                cache_num_proc=cache_num_proc,
                cache_keep_in_memory=cache_keep_in_memory,
                cache_format=cache_format,
            )

    def cache_text_embeddings(
        self,
        map_fn,
        i: int,
        regenerate_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        print(f"caching text embeddings: {self.path}")
        datasets_list = (
            self.size_bucket_datasets
            if self.use_size_buckets
            else self.ar_bucket_datasets
        )
        for ds in datasets_list:
            ds.cache_text_embeddings(
                map_fn, i,
                regenerate_cache=regenerate_cache,
                caching_batch_size=caching_batch_size,
                cache_num_proc=cache_num_proc,
                cache_keep_in_memory=cache_keep_in_memory,
                cache_format=cache_format,
            )
        empty_ds = datasets.Dataset.from_dict(
            {
                "caption": [""],
                "is_video": [False],
                "image_spec": [(None, None)],
            }
        )
        uncond_ds = _map_and_cache(
            empty_ds,
            map_fn,
            cache_dir=self.cache_dir,
            cache_file_prefix=f"uncond_text_embeddings_{i}_",
            regenerate_cache=regenerate_cache,
            num_proc=cache_num_proc,
            keep_in_memory=cache_keep_in_memory,
            cache_format=cache_format,
        )
        for sb in self.get_size_bucket_datasets():
            sb.uncond_text_embeddings.append(uncond_ds)


def parse_resolution_schedule(dataset_config: dict):
    """Parse the optional ``[resolution_schedule]`` section.

    Returns ``(active, stages, cum_frac)`` where ``stages`` is a list of
    ``(frozenset_of_resolutions, normalized_fraction)`` in temporal order and
    ``cum_frac`` is the running sum of the normalized fractions (last ~= 1.0).
    When the schedule is absent or disabled, returns ``(False, [], [])``.
    """
    sched = dataset_config.get("resolution_schedule")
    if not isinstance(sched, dict) or not sched.get("enabled", False):
        return False, [], []
    raw_stages = sched.get("stage", sched.get("stages", [])) or []
    stages = []
    for st in raw_stages:
        res = st.get("resolutions", st.get("resolution"))
        if res is None:
            continue
        if not isinstance(res, (list, tuple)):
            res = [res]
        res_set = frozenset(int(r) for r in res)
        if not res_set:
            continue
        frac = float(st.get("fraction", 0.0))
        if frac <= 0.0:
            continue
        stages.append((res_set, frac))
    if not stages:
        return False, [], []
    total = sum(f for _, f in stages)
    stages = [(rs, f / total) for rs, f in stages]
    cum_frac = []
    running = 0.0
    for _, f in stages:
        running += f
        cum_frac.append(running)
    return True, stages, cum_frac


class Dataset:
    """Top-level dataset: multiple DirectoryDatasets, post_init for DP rank and batch sizes."""

    def __init__(
        self,
        dataset_config: dict,
        model,
        skip_dataset_validation: bool = False,
    ) -> None:
        self.dataset_config = dataset_config
        self.model = model
        self.model_name = getattr(self.model, "name", "model")
        self.post_init_called = False
        self.eval_quantile = None
        if not skip_dataset_validation:
            if hasattr(
                model, "model_specific_dataset_config_validation"
            ) and callable(model.model_specific_dataset_config_validation):
                model.model_specific_dataset_config_validation(
                    dataset_config
                )
        try:
            cache_text_embeddings = bool(model.get_text_encoders())
        except Exception:
            cache_text_embeddings = False
        self.directory_datasets = []
        for directory_config in dataset_config["directory"]:
            dir_dataset = DirectoryDataset(
                directory_config,
                dataset_config,
                self.model_name,
                framerate=getattr(model, "framerate", None),
                round_to_multiple=getattr(
                    model, "pixels_round_to_multiple", 32
                ),
                skip_dataset_validation=skip_dataset_validation,
                cache_text_embeddings=cache_text_embeddings,
            )
            self.directory_datasets.append(dir_dataset)
        # Tag dropout + cached text embeddings is no longer refused: with the cache on, the
        # dropout distribution is pre-baked into the embedding cache via cached_caption_variants
        # (K = 1 bakes a single fixed variant for the whole dataset — diffusion-pipe's default;
        # K >= 2 bakes rotating variants), and with the cache off it is applied live per sample.
        # Either way the dropout reaches the model, so there is nothing to reject here.
        # Rotation is active when at least one directory limits images (max_images or
        # subsample_ratio < 1) and is not static; the loader uses this to keep workers in sync
        # with the current epoch (see loader.py).
        self.rotation_active = any(
            effective_sample_cap(
                1,
                directory_max_images(d.directory_config),
                directory_subsample_ratio(d.directory_config),
            )
            is not None
            and directory_subsample_shuffle(d.directory_config)
            for d in self.directory_datasets
        )

        # Staged multi-resolution schedule (optional). When active, the set of
        # resolutions sampled changes with training progress; the loader treats
        # schedule_active like rotation_active and re-creates the dataloader each
        # epoch so a new stage takes effect (see loader.py and set_epoch below).
        (
            self.schedule_active,
            self._schedule_stages,
            self._schedule_cum_frac,
        ) = parse_resolution_schedule(dataset_config)
        self.current_step = 1
        self._schedule_target = None
        self._active_stage = None
        self.full_epoch_len = 0

    @property
    def caption_variants(self) -> int:
        """Uniform captions-per-image across every bucket (1 when mixed).

        Multi-line .txt captions multiply the iteration order (each variant is
        its own example). The training loop divides steps_per_epoch by this so
        an "epoch" still means one pass over the images — variants rotate
        across the epochs instead of inflating them.
        """
        values = set()
        for dir_ds in self.directory_datasets:
            for sb in getattr(dir_ds, "size_bucket_datasets", []):
                values.add(sb.caption_variants)
            for ar in getattr(dir_ds, "ar_bucket_datasets", []):
                for sb in getattr(ar, "size_buckets", []):
                    values.add(sb.caption_variants)
        return values.pop() if len(values) == 1 else 1

    def set_epoch(self, epoch: int) -> None:
        """Propagate the current epoch to every size bucket so rotation advances.

        When a resolution schedule is active, also re-evaluate which stage the
        current step falls in and rebuild the iteration order if it changed. This
        runs during the loader's epoch rollover, right before the dataloader is
        re-created, so the new stage's resolutions take effect immediately.
        """
        for bucket in getattr(self, "buckets", []):
            bucket.set_epoch(epoch)
        if self.schedule_active and self.post_init_called:
            stage = self._stage_for_step(self.current_step)
            if stage != self._active_stage:
                self._active_stage = stage
                self.iteration_order = self._build_iteration_order(
                    self._active_resolutions_for_stage(stage)
                )

    def set_schedule_target(self, target_steps: int) -> None:
        """Set the total step budget the resolution schedule is measured against."""
        self._schedule_target = int(target_steps) if target_steps else None

    def update_active_stage(self, step: int) -> bool:
        """Update progress and switch the active stage if ``step`` crossed a boundary.

        Returns True when the active resolution set changed (so the caller can restart
        iteration mid-epoch). This is the step-accurate driver of the schedule; it
        works even when a single-resolution epoch is longer than a stage's step span
        (large datasets / few epochs), which epoch-boundary switching cannot handle.
        """
        if not (self.schedule_active and self.post_init_called):
            return False
        self.current_step = int(step)
        stage = self._stage_for_step(self.current_step)
        if stage == self._active_stage:
            return False
        self._active_stage = stage
        self.iteration_order = self._build_iteration_order(
            self._active_resolutions_for_stage(stage)
        )
        return True

    def _stage_for_step(self, step: int) -> int:
        """Return the stage index whose step range contains ``step`` (1-based)."""
        if not self.schedule_active:
            return 0
        target = self._schedule_target
        if not target or target <= 0:
            return 0
        progress = (max(1, int(step)) - 1) / target
        for i, cum in enumerate(self._schedule_cum_frac):
            if progress < cum:
                return i
        return len(self._schedule_cum_frac) - 1

    def _active_resolutions_for_stage(self, stage: int) -> frozenset:
        """Resolutions sampled during ``stage`` (empty stage list => all resolutions)."""
        if not self._schedule_stages:
            return frozenset()
        stage = max(0, min(stage, len(self._schedule_stages) - 1))
        return self._schedule_stages[stage][0]

    def distinct_size_buckets(self) -> set[tuple]:
        """Distinct (width, height, frames) pixel buckets with at least one sample.

        Each bucket is one latent shape the model will see, so the count sizes
        torch.compile's per-shape recompile budget (see training/compile_plan.py).
        Valid after the dataset manager's cache() pass, before post_init.
        """
        buckets = set()
        for dir_ds in self.directory_datasets:
            for sb in dir_ds.get_size_bucket_datasets():
                if len(sb.metadata_dataset) > 0:
                    buckets.add(tuple(sb.size_bucket[-3:]))
        return buckets

    def get_augmentation_resolver(self):
        """Return callable(image_spec) -> (resolved_config, fingerprint) or None."""
        roots: list[tuple[str, dict, str]] = []
        for dir_ds in self.directory_datasets:
            if not dir_ds._aug_enabled:
                continue
            root = str(Path(dir_ds.directory_config["path"]).resolve())
            roots.append((root, dir_ds._resolved_augmentation, dir_ds._aug_fingerprint))
        if not roots:
            return None

        def resolve(spec):
            path = str(spec[1])
            for root, aug_resolved, aug_fp in roots:
                if path_is_under(path, root):
                    return aug_resolved, aug_fp
            return None

        return resolve

    def post_init(
        self,
        data_parallel_rank: int,
        data_parallel_world_size: int,
        per_device_batch_size: dict,
        gradient_accumulation_steps: int,
        per_device_batch_size_image: dict,
    ) -> None:
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        global_batch_size = {
            s: bs * gradient_accumulation_steps * self.data_parallel_world_size
            for s, bs in per_device_batch_size.items()
        }
        global_batch_size_image = {
            s: bs * gradient_accumulation_steps * self.data_parallel_world_size
            for s, bs in per_device_batch_size_image.items()
        }
        datasets_by_size_bucket = defaultdict(list)
        for dir_ds in self.directory_datasets:
            for sb in dir_ds.get_size_bucket_datasets():
                datasets_by_size_bucket[sb.size_bucket].append(sb)
        self.buckets = []
        for datalist in datasets_by_size_bucket.values():
            self.buckets.append(ConcatenatedBatchedDataset(datalist))
        for bucket in self.buckets:
            bucket.post_init(
                global_batch_size,
                global_batch_size_image,
                data_parallel_rank,
                data_parallel_world_size,
            )
        # full_epoch_len reflects all resolutions (used for total_steps / progress),
        # independent of which schedule stage is currently active.
        full_order = self._build_iteration_order(active_resolutions=None)
        self.full_epoch_len = len(full_order)
        if self.schedule_active:
            self._active_stage = self._stage_for_step(self.current_step)
            self.iteration_order = self._build_iteration_order(
                self._active_resolutions_for_stage(self._active_stage)
            )
        else:
            self.iteration_order = full_order
        self.post_init_called = True

    def _build_iteration_order(self, active_resolutions: frozenset | None = None):
        """Build the (bucket_idx, item_idx) order, optionally limited to a resolution set.

        ``active_resolutions=None`` includes every bucket (default behavior). When a
        stage filters down to no rows (e.g. a configured resolution produced no
        images), fall back to all buckets so a run never stalls on an empty epoch.
        """
        iteration_order = []
        for i, bucket in enumerate(self.buckets):
            if active_resolutions is not None and bucket.resolution not in active_resolutions:
                continue
            iteration_order.extend([i] * len(bucket))
        if active_resolutions is not None and not iteration_order:
            logger.warning(
                "resolution_schedule stage %s matched no cached buckets; "
                "falling back to all resolutions for this epoch.",
                sorted(active_resolutions),
            )
            for i, bucket in enumerate(self.buckets):
                iteration_order.extend([i] * len(bucket))
        shuffle_with_seed(iteration_order, 0)
        cumulative = [0] * len(self.buckets)
        for k, idx in enumerate(iteration_order):
            iteration_order[k] = (idx, cumulative[idx])
            cumulative[idx] += 1
        if subsample_ratio := self.dataset_config.get("subsample_ratio"):
            new_len = max(1, int(len(iteration_order) * subsample_ratio))
            iteration_order = iteration_order[:new_len]
        return iteration_order

    def __len__(self) -> int:
        assert self.post_init_called
        return len(self.iteration_order)

    def avg_examples_per_step(self) -> float:
        """Mean images consumed per optimizer step across the full epoch.

        With a per-resolution ``micro_batch_size_per_gpu`` dict the per-step
        batch varies by bucket, so example accounting (x-axis examples,
        eval/save_every_n_examples) needs the weighted average: total images
        per epoch over total steps per epoch. Equals the uniform global batch
        when the config is a plain integer.
        """
        assert self.post_init_called
        total_images = sum(len(b.iteration_order) for b in self.buckets)
        total_steps = sum(len(b) for b in self.buckets)
        return total_images / max(1, total_steps)

    def __getitem__(self, idx):
        assert self.post_init_called
        i, j = self.iteration_order[idx]
        examples = self.buckets[i][j]
        return self._collate(examples)

    def _collate(self, examples: list) -> dict:
        ret = {}
        for key in examples[0]:
            if key == "mask":
                continue
            features = [ex[key] for ex in examples]
            if torch.is_tensor(features[0]):
                shape = features[0].shape
                if all(f.shape == shape for f in features):
                    ret[key] = torch.stack(features)
                else:
                    ret[key] = features
            else:
                ret[key] = features
        masks = [ex["mask"] for ex in examples]
        shape = None
        for m in masks:
            if m is not None:
                assert shape is None or m.shape == shape
                shape = m.shape
        if shape is not None:
            for i, m in enumerate(masks):
                if m is None:
                    masks[i] = torch.ones(shape, dtype=torch.float16)
            ret["mask"] = torch.stack(masks)
        else:
            ret["mask"] = None
        return ret

    def cache_metadata(
        self,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        cache_num_proc: int | None = None,
    ) -> None:
        for ds in self.directory_datasets:
            ds.cache_metadata(
                regenerate_cache=regenerate_cache,
                trust_cache=trust_cache,
                cache_num_proc=cache_num_proc,
            )

    def cache_latents(
        self,
        map_fn,
        regenerate_cache: bool = False,
        trust_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        for ds in self.directory_datasets:
            ds.cache_latents(
                map_fn,
                regenerate_cache=regenerate_cache,
                trust_cache=trust_cache,
                caching_batch_size=caching_batch_size,
                cache_num_proc=cache_num_proc,
                cache_keep_in_memory=cache_keep_in_memory,
                cache_format=cache_format,
            )

    def cache_text_embeddings(
        self,
        map_fn,
        i: int,
        regenerate_cache: bool = False,
        caching_batch_size: int = 1,
        cache_num_proc: int | None = None,
        cache_keep_in_memory: bool = False,
        cache_format: str = "v2",
    ) -> None:
        for ds in self.directory_datasets:
            ds.cache_text_embeddings(
                map_fn, i,
                regenerate_cache=regenerate_cache,
                caching_batch_size=caching_batch_size,
                cache_num_proc=cache_num_proc,
                cache_keep_in_memory=cache_keep_in_memory,
                cache_format=cache_format,
            )
