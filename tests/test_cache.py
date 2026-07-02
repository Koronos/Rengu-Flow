"""Unit tests for the disk cache (mmap tensor stacks + SQLite metadata)."""

from __future__ import annotations

import json

import pytest
import torch

from rengu_flow.utils.cache import (
    MANIFEST_NAME,
    Cache,
    open_disk_cache,
    reject_legacy_v1,
)


def _latents_item(*, mask=None, scale: float = 1.0) -> dict:
    return {
        "latents": torch.randn(4, 2, 3, 3, dtype=torch.bfloat16) * scale,
        "mask": mask,
        "caption": "a photo",
        "image_spec": ("img.png", None),
    }


def test_cache_roundtrip_bf16_and_meta(tmp_path):
    cache = Cache(tmp_path / "latents", "fp-test")
    items = [_latents_item(mask=None), _latents_item(mask=torch.ones(2, 3))]
    for it in items:
        cache.add(it)
    cache.finalize_current_shard()

    assert len(cache) == 2
    for i, expected in enumerate(items):
        got = cache[i]
        assert got["caption"] == expected["caption"]
        assert tuple(got["image_spec"]) == expected["image_spec"]
        if expected["mask"] is None:
            assert got["mask"] is None
        else:
            assert torch.equal(got["mask"], expected["mask"])
        assert got["latents"].shape == expected["latents"].shape
        assert got["latents"].dtype == expected["latents"].dtype
        assert torch.allclose(got["latents"].float(), expected["latents"].float())


def test_cache_read_from_other_thread(tmp_path):
    """Reading from a DataLoader prefetch/worker thread must not trip SQLite's thread guard."""
    import threading

    cache = Cache(tmp_path / "latents", "fp-thread")
    for _ in range(3):
        cache.add(_latents_item())
    cache.finalize_current_shard()

    # Open the read path on this (main) thread first, so the meta connection is created here.
    assert cache[0]["caption"] == "a photo"

    errors: list[Exception] = []

    def read_in_thread() -> None:
        try:
            for i in range(3):
                assert cache[i]["caption"] == "a photo"
        except Exception as e:  # noqa: BLE001 - record any cross-thread failure
            errors.append(e)

    t = threading.Thread(target=read_in_thread)
    t.start()
    t.join()
    assert not errors, f"cross-thread cache read failed: {errors!r}"


def test_cache_get_many(tmp_path):
    cache = Cache(tmp_path / "latents", "fp-many")
    for _ in range(4):
        cache.add(_latents_item())
    cache.finalize_current_shard()
    batch = cache.get_many([3, 1, 0])
    assert len(batch) == 3
    assert batch[0]["caption"] == "a photo"


def test_cache_resume_after_finalize(tmp_path):
    cache_dir = tmp_path / "latents"
    c1 = Cache(cache_dir, "fp-resume")
    c1.add(_latents_item(scale=1.0))
    c1.add(_latents_item(scale=2.0))
    c1.finalize_current_shard()

    c2 = Cache(cache_dir, "fp-resume")
    assert len(c2) == 2
    item3 = _latents_item(scale=3.0)
    c2.add(item3)
    c2.finalize_current_shard()
    assert len(c2) == 3
    assert torch.allclose(c2[2]["latents"].float(), item3["latents"].float())


def test_cache_fingerprint_mismatch_clears(tmp_path):
    cache_dir = tmp_path / "latents"
    c1 = Cache(cache_dir, "fp-a")
    c1.add(_latents_item())
    c1.finalize_current_shard()
    assert (cache_dir / MANIFEST_NAME).is_file()
    c1.close()  # release mmap/db handles so the stale-fingerprint clear can unlink on Windows

    c2 = Cache(cache_dir, "fp-b")
    assert len(c2) == 0
    manifest = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["fingerprint"] == "fp-b"
    assert manifest["count"] == 0


def test_cache_shape_mismatch_raises(tmp_path):
    cache = Cache(tmp_path / "latents", "fp-shape")
    cache.add(_latents_item())
    bad = _latents_item()
    bad["latents"] = torch.randn(8, 2, 3, 3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="incompatible"):
        cache.add(bad)


def test_cache_int64_tensor(tmp_path):
    cache = Cache(tmp_path / "te", "fp-int")
    item = {
        "prompt_embeds": torch.randn(8, 4, dtype=torch.float32),
        "attn_mask": torch.ones(8, dtype=torch.int64),
        "caption": "x",
    }
    cache.add(item)
    cache.finalize_current_shard()
    got = cache[0]
    assert got["attn_mask"].dtype == torch.int64
    assert torch.equal(got["attn_mask"], item["attn_mask"])


def test_cache_variable_dim0_prompt_embeds(tmp_path):
    cache = Cache(tmp_path / "te", "fp-var")
    shapes = [(7, 768), (8, 768), (6, 768), (8, 768)]
    for i, shape in enumerate(shapes):
        cache.add(
            {
                "prompt_embeds": torch.randn(*shape, dtype=torch.float32),
                "caption": f"c{i}",
            }
        )
    cache.finalize_current_shard()
    for i, shape in enumerate(shapes):
        got = cache[i]["prompt_embeds"]
        assert tuple(got.shape) == shape
    assert cache.tensor_specs["prompt_embeds"]["shape"] == [8, 768]


def test_mmaps_open_lazily_per_key(tmp_path):
    """An opened-but-unread cache holds no mmaps; reading maps only the keys touched.

    Keeps the fd count bounded so many bucket caches can coexist (was: every cache
    eagerly mmap'd all tensors on open -> 'too many open files' with many buckets).
    """
    cache_dir = tmp_path / "latents"
    c1 = Cache(cache_dir, "fp-lazy")
    c1.add(_latents_item())  # latents + mask tensors
    c1.finalize_current_shard()
    assert c1._mmaps == {}, "finalize must not eager-open mmaps"
    c1.close()

    assert c1._meta_con is None, "finalize must release the SQLite connection too"

    c2 = Cache(cache_dir, "fp-lazy")
    assert c2._mmaps == {}, "opening a cache must not mmap anything"
    assert c2._meta_con is None, "opening a cache must not open SQLite"
    item = c2[0]  # first read maps + opens meta on demand
    assert item["latents"] is not None
    assert "latents" in c2._mmaps  # the touched key is now mapped
    assert c2._meta_con is not None  # meta opened on first read
    c2.close()


def test_corrupt_manifest_regenerates_instead_of_crashing(tmp_path):
    """A manifest truncated by a crash mid-write must regenerate, not raise on resume."""
    cache_dir = tmp_path / "latents"
    c1 = Cache(cache_dir, "fp-corrupt")
    c1.add(_latents_item())
    c1.finalize_current_shard()
    assert len(c1) == 1
    c1.close()

    # Simulate a crash mid-write: manifest.json left as invalid JSON.
    (cache_dir / MANIFEST_NAME).write_text('{"format_version": 2, "coun', encoding="utf-8")

    c2 = Cache(cache_dir, "fp-corrupt")  # must not raise
    assert len(c2) == 0
    manifest = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["count"] == 0


def test_manifest_missing_required_key_regenerates(tmp_path):
    """A structurally-wrong manifest (missing keys) regenerates rather than KeyError."""
    cache_dir = tmp_path / "latents"
    c1 = Cache(cache_dir, "fp-key")
    c1.add(_latents_item())
    c1.finalize_current_shard()
    c1.close()

    (cache_dir / MANIFEST_NAME).write_text('{"format_version": 2}', encoding="utf-8")  # no count/tensors

    c2 = Cache(cache_dir, "fp-key")  # must not raise
    assert len(c2) == 0


def test_reject_legacy_v1_and_open(tmp_path):
    v2_dir = tmp_path / "current"
    v2_dir.mkdir()
    c = Cache(v2_dir, "fp")
    c.add(_latents_item())
    c.finalize_current_shard()
    reject_legacy_v1(v2_dir)  # no metadata.db -> no raise

    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    # A legacy v1 cache is identified by its metadata.db; the v1 writer is gone,
    # so drop the marker file directly to exercise the rejection path.
    (v1_dir / "metadata.db").write_bytes(b"")
    with pytest.raises(ValueError, match="Legacy cache v1"):
        reject_legacy_v1(v1_dir)

    opened = open_disk_cache(v2_dir, "fp")
    assert isinstance(opened, Cache)

    with pytest.raises(ValueError, match="Legacy cache v1"):
        open_disk_cache(v1_dir, "fp1")

    with pytest.raises(ValueError, match="Legacy cache v1"):
        open_disk_cache(v1_dir, "fp1")


def test_cache_resume_after_checkpoint(tmp_path):
    """An interrupted run resumes from the last checkpoint, dropping any tensor/meta
    rows written past it, and finishes consistently."""
    from rengu_flow.utils.cache import TENSORS_DIR

    fp = "fp-resume"
    first = [_latents_item(scale=float(i + 1)) for i in range(5)]
    cache = Cache(tmp_path / "latents", fp)
    for it in first:
        cache.add(it)
    cache._checkpoint()  # durable resume point at 5
    assert cache.count == 5

    # Simulate a crash mid-bucket: a tensor row + a committed meta row written past
    # the checkpoint that the manifest never recorded.
    key = next(iter(cache.tensor_specs))
    tbin = tmp_path / "latents" / TENSORS_DIR / f"{key}.bin"
    with open(tbin, "ab") as f:
        f.write(b"\x00" * cache._row_size(key))  # junk row at idx 5
    cache._meta_con.execute("INSERT INTO item_meta(idx, payload) VALUES(5, '{}')")
    cache._meta_con.commit()
    del cache  # drop the writer without finalizing

    # Reopen: count is the manifest's (5), not the stray tail.
    cache2 = open_disk_cache(tmp_path / "latents", fp)
    assert len(cache2) == 5
    rest = [_latents_item(scale=float(i + 10)) for i in range(5)]  # idx 5..9
    for it in rest:
        cache2.add(it)
    cache2.finalize_current_shard()

    assert len(cache2) == 10
    expected = first + rest
    for i, exp in enumerate(expected):
        assert torch.allclose(cache2[i]["latents"].float(), exp["latents"].float()), i


def test_cache_valid_flags(tmp_path):
    cache = Cache(tmp_path / "latents", "fp-valid")
    cache.add({**_latents_item(), "valid": True})
    cache.add({**_latents_item(), "valid": False})   # tombstone
    cache.add(_latents_item())                        # no key -> defaults valid
    cache.finalize_current_shard()
    assert cache.valid_flags() == [True, False, True]


def test_cache_variable_dim0_3d_and_1d(tmp_path):
    """krea2-shaped rows: (L, layers, D) stacks and (L,) bool masks grow on dim 0 too."""
    cache = Cache(tmp_path / "te3d", "fp-var3d")
    lengths = [50, 55, 48]
    for i, n in enumerate(lengths):
        cache.add(
            {
                "prompt_embeds": torch.randn(n, 12, 32, dtype=torch.bfloat16),
                "text_mask": torch.ones(n, dtype=torch.bool),
                "caption": f"c{i}",
            }
        )
    cache.finalize_current_shard()
    for i, n in enumerate(lengths):
        assert tuple(cache[i]["prompt_embeds"].shape) == (n, 12, 32)
        assert tuple(cache[i]["text_mask"].shape) == (n,)
        assert cache[i]["text_mask"].all()
    assert cache.tensor_specs["prompt_embeds"]["shape"] == [55, 12, 32]
