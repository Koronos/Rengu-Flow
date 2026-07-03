"""Cache reuse must be reliable without --trust-cache.

Regression guards for "the cache regenerates although nothing changed":
  * a second run over an unchanged dataset re-encodes NOTHING (latents and TE),
  * every cache-key ingredient is identical across processes with different
    PYTHONHASHSEED (randomized str hashing is the classic source of per-run
    fingerprint churn: any set/dict-order or builtin-hash leak shows up here).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import torch

from rengu_flow.data.dataset import SizeBucketDataset
from rengu_flow.data.tag_dropout import TagDropoutConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

DROP = TagDropoutConfig(enabled=True, default_probability=0.5)


def _mock_dir_dataset(dataset_config, tag_dropout=DROP):
    return SimpleNamespace(
        captions_dict=None,
        uncond_fraction=0.0,
        tag_dropout=tag_dropout,
        dataset_config=dataset_config,
        caches_text_embeddings=True,
        _aug_fingerprint="",
    )


def _metadata(n=3):
    return datasets.Dataset.from_dict(
        {
            "image_spec": [[None, f"img{i}.jpg"] for i in range(n)],
            "caption": [["red, hair, smile, outdoors"] for _ in range(n)],
        }
    )


def _build(tmp_path, counter):
    def latent_map(example, rank):
        counter["latents"] += len(example["image_spec"])
        return {"latents": torch.zeros(len(example["image_spec"]), 4)}

    sb = SizeBucketDataset(
        _metadata(3),
        {"path": str(tmp_path), "num_repeats": 1},
        (512, 512, 1),
        tmp_path / "cache",
        _mock_dir_dataset({"cached_caption_variants": 2}),
    )
    sb.cache_latents(latent_map, regenerate_cache=False, trust_cache=False)

    def te_map(example, rank):
        counter["te"] += len(example["caption"])
        return {"prompt_embeds": torch.zeros(len(example["caption"]), 3, 8)}

    sb.cache_text_embeddings(te_map, 0, regenerate_cache=False)
    return sb


def test_second_run_reencodes_nothing(tmp_path):
    """Fresh objects, unchanged inputs, trust_cache=False: run 2 must be pure cache load."""
    first = {"latents": 0, "te": 0}
    _build(tmp_path, first)
    assert first["latents"] > 0 and first["te"] > 0  # run 1 actually encoded

    second = {"latents": 0, "te": 0}
    _build(tmp_path, second)
    assert second == {"latents": 0, "te": 0}, (
        f"unchanged dataset re-encoded {second} on the second run — a cache-key "
        "ingredient is unstable (see test_fingerprints_stable_across_hash_seeds)"
    )


_PROBE = r"""
import sys
from types import SimpleNamespace
import datasets, torch
from datasets.fingerprint import Hasher
from rengu_flow.data.dataset import SizeBucketDataset, AUG_MVP_VERSION, expand_caption_variants
from rengu_flow.data.cache_utils import content_fingerprint
from rengu_flow.data.tag_dropout import TagDropoutConfig
from rengu_flow.data.cache_paths import dataset_cache_id

meta = datasets.Dataset.from_dict({
    "image_spec": [[None, f"img{i}.jpg"] for i in range(3)],
    "caption": [["red, hair, smile, outdoors"] for _ in range(3)],
})
sb = SizeBucketDataset(
    meta,
    {"path": sys.argv[1], "num_repeats": 1},
    (512, 512, 1),
    __import__("pathlib").Path(sys.argv[1]) / "cache",
    SimpleNamespace(
        captions_dict=None, uncond_fraction=0.0,
        tag_dropout=TagDropoutConfig(enabled=True, default_probability=0.5),
        dataset_config={"cached_caption_variants": 3},
        caches_text_embeddings=True, _aug_fingerprint="",
    ),
)
sb.cache_latents(lambda ex, rank: {"latents": torch.zeros(len(ex["image_spec"]), 4)},
                 regenerate_cache=False, trust_cache=False)
cols = sb.metadata_dataset.column_names
latent_cols = [c for c in ("image_spec", "mask_file", "size_bucket", "is_video", "control_file") if c in cols]
lfp = content_fingerprint(sb.metadata_dataset, latent_cols)
cfp = content_fingerprint(sb.metadata_dataset, [c for c in ("caption", "image_spec") if c in cols])
key = Hasher.hash([AUG_MVP_VERSION, sb._aug_fingerprint, lfp, "cache_format=v2"])
variants = expand_caption_variants(
    ["one, two, three, four"], 4,
    TagDropoutConfig(enabled=True, default_probability=0.5), True, seed_key="img.jpg",
)
# A fixed path: the id must depend only on the value, never on the process.
print(dataset_cache_id({"directory": [{"path": "/fixed/dataset/dir"}]}))
print(lfp); print(cfp); print(key); print("|".join(variants))
"""


def test_fingerprints_stable_across_hash_seeds(tmp_path):
    """Every cache-key ingredient must be byte-identical under different PYTHONHASHSEED.

    Python randomizes str hashing per process; anything deriving cache keys from set/dict
    iteration order or builtin hash() produces a new fingerprint every run — the trainer
    then silently regenerates every cache on every launch.
    """
    outs = []
    for seed in ("1", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(REPO_ROOT))
        work = tmp_path / f"seed{seed}"
        work.mkdir()
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE, str(work)],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        # Keep only the 5 probe lines; earlier stdout is caching progress whose
        # line count is time-throttled (not a determinism signal).
        outs.append("\n".join(proc.stdout.strip().splitlines()[-5:]))
    assert outs[0] == outs[1], (
        "cache-key ingredients changed with PYTHONHASHSEED:\n"
        f"--- seed 1 ---\n{outs[0]}\n--- seed 424242 ---\n{outs[1]}"
    )
