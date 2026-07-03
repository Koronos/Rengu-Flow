"""Unified caching progress: one log format, one monotonic bar (regression guards)."""

from __future__ import annotations

from types import SimpleNamespace

import datasets
import torch

from rengu_flow.control.progress_stream import ProgressEmitter
from rengu_flow.data import caching_progress
from rengu_flow.data.caching_progress import CachingProgress
from rengu_flow.data.dataset import SizeBucketDataset
from rengu_flow.data.tag_dropout import TagDropoutConfig


class _CaptureEmitter(ProgressEmitter):
    def __init__(self):
        super().__init__(min_interval_sec=0.0)
        self.payloads = []
        self._write = lambda line: None

    def emit(self, payload, *, force=False):
        self.payloads.append(dict(payload))
        return True


def test_percent_is_monotonic_across_stages_and_units():
    em = _CaptureEmitter()
    p = CachingProgress(emitter=em, quiet=True)
    p.plan(["metadata", "latents", "text embeddings 1"])
    with p.stage("metadata", units=2):
        with p.unit("dir a"):
            pass
        with p.unit("dir b"):
            pass
    with p.stage("latents", units=3):
        for i in range(3):
            with p.unit(f"bucket {i}"):
                p.unit_progress(1, 4)
                p.unit_progress(3, 4)
                p.unit_progress(4, 4)
    with p.stage("text embeddings 1", units=1):
        with p.unit("bucket 0"):
            p.unit_progress(2, 2)

    percents = [pl["percent"] for pl in em.payloads]
    assert percents == sorted(percents), f"bar went backwards: {percents}"
    assert percents[-1] == 100.0
    # Stage labels ride along so the UI can say WHAT is progressing.
    stages_seen = {(pl["stage"], pl["stage_name"]) for pl in em.payloads}
    assert (2, "latents") in stages_seen and (3, "text embeddings 1") in stages_seen


def test_stage_lines_report_encoded_vs_reused(capsys):
    p = CachingProgress(emitter=None)
    p.plan(["latents"])
    with p.stage("latents", units=1):
        with p.unit("latents 512x512x1"):
            p.add_encoded(240)
            p.add_reused(15)
            p.note("latents: 240 to encode, 15 cached")
    out = capsys.readouterr().out
    assert "[cache] stage 1/1: latents" in out
    assert "240 encoded, 15 reused" in out
    assert "[cache]   latents: 240 to encode, 15 cached" in out


def test_note_falls_back_to_plain_print_when_inactive(capsys):
    assert caching_progress.get_active() is None
    caching_progress.note("building iteration order")
    assert "[cache]   building iteration order" in capsys.readouterr().out


def _bucket(tmp_path, counter):
    def latent_map(example, rank):
        counter["n"] += len(example["image_spec"])
        return {"latents": torch.zeros(len(example["image_spec"]), 4)}

    sb = SizeBucketDataset(
        datasets.Dataset.from_dict(
            {
                "image_spec": [[None, f"img{i}.jpg"] for i in range(3)],
                "caption": [["red, hair"] for _ in range(3)],
            }
        ),
        {"path": str(tmp_path), "num_repeats": 1},
        (512, 512, 1),
        tmp_path / "cache",
        SimpleNamespace(
            captions_dict=None,
            uncond_fraction=0.0,
            tag_dropout=TagDropoutConfig(),
            dataset_config={},
            caches_text_embeddings=True,
            _aug_fingerprint="",
        ),
    )
    sb.cache_latents(latent_map, regenerate_cache=False, trust_cache=False)


def test_cache_flow_reports_reuse_through_coordinator(tmp_path, capsys):
    """End to end through _map_and_cache: run 1 says 'N to encode', run 2 says '0 to encode'."""
    em = _CaptureEmitter()
    p = CachingProgress(emitter=em)
    p.plan(["latents"])
    with caching_progress.activate(p):
        with p.stage("latents", units=2):
            with p.unit("latents 512x512x1"):
                _bucket(tmp_path, {"n": 0})
            with p.unit("latents 512x512x1 (rerun)"):
                _bucket(tmp_path, {"n": 0})
    out = capsys.readouterr().out
    assert "latents: 3 to encode, 0 cached" in out
    assert "latents: 0 to encode, 3 cached" in out  # second run = pure reuse, visible in the log
    percents = [pl["percent"] for pl in em.payloads]
    assert percents == sorted(percents)
