"""Directory-based dataset with buckets, cache, and multi-caption (from diffusion-pipe)."""

from __future__ import annotations

import json
import logging
import math
import os
import random
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
    dedup_and_sort,
    resolve_cache_num_proc,
    seed_from_hash,
)
from rengu_flow.utils.common import is_main_process, round_to_nearest_multiple
from rengu_flow.utils.paths import path_is_under

logger = logging.getLogger(__name__)

CAPTIONS_JSON_FILE = "captions.json"
UNCOND_FRACTION = 0.0

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


def shuffle_captions(
    captions: list[str],
    count: int = 0,
    delimiter: str = ", ",
    caption_prefix: str = "",
) -> list[str]:
    """Apply prefix and optionally shuffle comma-separated tags and repeat."""
    if count == 0:
        return [caption_prefix + c for c in captions]

    def shuffle_caption(caption: str, delim: str = ", ") -> str:
        parts = caption.split(delim)
        random.shuffle(parts)
        return delim.join(parts)

    return [
        caption_prefix + shuffle_caption(caption, delimiter)
        for caption in captions
        for _ in range(count)
    ]


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
    from rengu_flow.data.cache_utils import _map_and_cache

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
    te_dataset = _map_and_cache(
        flattened,
        map_fn,
        cache_dir,
        cache_file_prefix=f"text_embeddings_{i}_",
        new_fingerprint_args=[i],
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


class SizeBucketDataset:
    """Single size bucket from one directory: latents + text embeddings cache, iteration order."""

    def __init__(
        self,
        metadata_dataset,
        directory_config: dict,
        size_bucket: tuple,
        cache_base: Path,
        directory_dataset=None,
    ) -> None:
        # Per-bucket shuffle mixes multi-resolution training better (diffusion-pipe).
        metadata_dataset = metadata_dataset.shuffle(seed=seed_from_hash(size_bucket))
        self.metadata_dataset = metadata_dataset
        self.directory_config = directory_config
        self.size_bucket = size_bucket
        self.path = Path(directory_config["path"])
        self.cache_dir = cache_base / f"cache_{bucket_suffix(size_bucket)}"
        self.captions_dict = (
            directory_dataset.captions_dict if directory_dataset is not None else None
        )

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
        self._aug_fingerprint = getattr(directory_dataset, "_aug_fingerprint", "")

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
        if map_fn is None:
            self.latent_dataset = _map_and_cache(
                self.metadata_dataset,
                None,
                self.cache_dir,
                cache_file_prefix="latents_",
                new_fingerprint_args=latent_fp_args,
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
            regenerate_cache=regenerate_cache,
            caching_batch_size=caching_batch_size,
            num_proc=cache_num_proc,
            keep_in_memory=cache_keep_in_memory,
            cache_format=cache_format,
        )
        assert len(self.latent_dataset) == len(self.metadata_dataset)

        if (
            regenerate_cache
            or not iteration_order_cache_dir.exists()
            or not trust_cache
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
            iteration_order = trim_iteration_order_by_subsample_ratio(
                iteration_order, directory_subsample_ratio(self.directory_config)
            )
            iteration_order.save_to_disk(str(iteration_order_cache_dir))

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
            UNCOND_FRACTION > 0 and random.random() < UNCOND_FRACTION
        )
        if use_uncond:
            caption = ""
        elif self.captions_dict:
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
            caption = entry["caption"]
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

    def get_items_batch(self, idx_list: list[int]) -> list[dict]:
        """Load multiple training samples; batches latent cache reads per shard."""
        entries = []
        for idx in idx_list:
            idx = idx % len(self.iteration_order)
            entries.append(self.iteration_order[idx])
        latent_idxs = [e["latents_idx"] for e in entries]
        latent_dicts = self.latent_dataset.get_many(latent_idxs)
        return [
            self._sample_from_entry(entry, latent_dicts[i])
            for i, entry in enumerate(entries)
        ]

    def __getitem__(self, idx):
        idx = idx % len(self.iteration_order)
        entry = self.iteration_order[idx]
        return self._sample_from_entry(entry)

    def __len__(self) -> int:
        return int(len(self.iteration_order) * self.num_repeats)


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
    ) -> None:
        self._set_defaults(directory_config, dataset_config)
        self.directory_config = directory_config
        self.dataset_config = dataset_config
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

        shuffle_tags = directory_config.get(
            "shuffle_tags", dataset_config.get("shuffle_tags", False)
        )
        if shuffle_tags:
            # When tag shuffling is on, cache_shuffle_num is the per-image shuffle/repeat
            # count; an unset or 0 value means "shuffle once" rather than "off".
            cache_shuffle_num = directory_config.get(
                "cache_shuffle_num", dataset_config.get("cache_shuffle_num", 0)
            )
            self.shuffle = cache_shuffle_num or 1
        else:
            # Tag shuffling off → no cache shuffle, regardless of cache_shuffle_num.
            self.shuffle = 0
        self.shuffle_metadata = directory_config["shuffle_metadata"]
        self.directory_config["cache_shuffle_num"] = self.shuffle
        self.shuffle_delimiter = directory_config.get(
            "cache_shuffle_delimiter",
            dataset_config.get("cache_shuffle_delimiter", ", "),
        )
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
            min_ar = directory_config.get(
                "min_ar", dataset_config["min_ar"]
            )
            max_ar = directory_config.get(
                "max_ar", dataset_config["max_ar"]
            )
            num_ar = directory_config.get(
                "num_ar_buckets", dataset_config["num_ar_buckets"]
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

    def _set_defaults(
        self, directory_config: dict, dataset_config: dict
    ) -> None:
        directory_config.setdefault(
            "enable_ar_bucket",
            dataset_config.get("enable_ar_bucket", False),
        )
        directory_config.setdefault(
            "shuffle_tags", dataset_config.get("shuffle_tags", False)
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
            for file in tqdm(files):
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
        metadata_map_fn = self._metadata_map_fn()
        print("Caching ungrouped metadata.")
        metadata_dataset = metadata_dataset.map(
            metadata_map_fn,
            cache_file_name=str(metadata_cache_2),
            load_from_cache_file=(not regenerate_cache and trust_cache),
            batched=True,
            batch_size=1,
            num_proc=metadata_num_proc,
            remove_columns=metadata_dataset.column_names,
        )
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
            if self.directory_config.get("shuffle_tags") and self.shuffle == 0:
                self.shuffle = 1
            captions = shuffle_captions(
                captions,
                self.shuffle,
                self.shuffle_delimiter,
                self.directory_config.get("directory_caption", ""),
            )
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

        return fn

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
            )
            self.directory_datasets.append(dir_dataset)

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
        iteration_order = []
        for i, bucket in enumerate(self.buckets):
            iteration_order.extend([i] * len(bucket))
        shuffle_with_seed(iteration_order, 0)
        cumulative = [0] * len(self.buckets)
        for k, idx in enumerate(iteration_order):
            iteration_order[k] = (idx, cumulative[idx])
            cumulative[idx] += 1
        self.iteration_order = iteration_order
        if subsample_ratio := self.dataset_config.get("subsample_ratio"):
            new_len = max(1, int(len(self.iteration_order) * subsample_ratio))
            self.iteration_order = self.iteration_order[:new_len]
        self.post_init_called = True

    def __len__(self) -> int:
        assert self.post_init_called
        return len(self.iteration_order)

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
