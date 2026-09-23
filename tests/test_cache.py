"""Unit tests for the disk cache (mmap tensor stacks + SQLite metadata)."""

from __future__ import annotations

import json

import pytest
import torch

from rengu_flow.utils.cache import (
    MANIFEST_NAME,
    TENSORS_DIR,
    Cache,
    StaleCacheLayoutError,
    open_disk_cache,
    reject_legacy_v1,
    reject_stale_sequence_layout,
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
    # Ragged storage: each row kept at its own length, no dim-0 padding to a shared max.
    spec = cache.tensor_specs["prompt_embeds"]
    assert spec["ragged"] is True
    assert spec["trailing_shape"] == [768]
    # The .bin holds exactly the sum of real row bytes (bf16 storage = 2 B/elt) — zero padding.
    bin_size = (tmp_path / "te" / TENSORS_DIR / "prompt_embeds.bin").stat().st_size
    assert bin_size == sum(n * 768 * 2 for n, _d in shapes)


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
    """krea2-shaped rows: (L, layers, D) stacks and (L,) bool masks stored ragged per row."""
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
    # Both sequence keys are ragged (no shared dim-0 width to pad short rows up to).
    assert cache.tensor_specs["prompt_embeds"]["ragged"] is True
    assert cache.tensor_specs["prompt_embeds"]["trailing_shape"] == [12, 32]
    assert cache.tensor_specs["text_mask"]["ragged"] is True
    emb_bytes = (tmp_path / "te3d" / TENSORS_DIR / "prompt_embeds.bin").stat().st_size
    assert emb_bytes == sum(n * 12 * 32 * 2 for n in lengths)


def test_cache_image_pad_mask_is_ragged_like_the_embeddings(tmp_path):
    """qwen_image21 edit rows: the (L,) image_pad_mask has the embeddings' per-row length. A
    fixed-width stack refused a row longer than the first and padded a shorter one past its
    embeddings."""
    cache = Cache(tmp_path / "te_edit", "fp-pad-mask")
    lengths = [10, 14, 6]
    for i, n in enumerate(lengths):
        pad = torch.zeros(n, dtype=torch.bool)
        pad[2:6] = True
        cache.add(
            {
                "prompt_embeds": torch.randn(n, 8),
                "text_mask": torch.ones(n, dtype=torch.bool),
                "image_pad_mask": pad,
                "caption": f"c{i}",
            }
        )
    cache.finalize_current_shard()
    for i, n in enumerate(lengths):
        assert tuple(cache[i]["image_pad_mask"].shape) == (n,)
        assert cache[i]["image_pad_mask"].tolist() == [False] * 2 + [True] * 4 + [False] * (n - 6)
    assert cache.tensor_specs["image_pad_mask"]["ragged"] is True


def test_cache_ragged_resume_appends_correctly(tmp_path):
    """Reopen a ragged cache and append more rows: offsets/truncate must keep every row readable."""
    d = tmp_path / "te_resume"
    lengths_a, lengths_b = [40, 90, 55], [70, 30]
    c1 = Cache(d, "fp-resume")
    for i, n in enumerate(lengths_a):
        c1.add({"prompt_embeds": torch.full((n, 8), float(i), dtype=torch.bfloat16), "caption": f"a{i}"})
    c1.finalize_current_shard()
    c1.close()

    c2 = Cache(d, "fp-resume")
    assert len(c2) == len(lengths_a)  # recovered committed rows on reopen
    for j, n in enumerate(lengths_b):
        c2.add({"prompt_embeds": torch.full((n, 8), float(100 + j), dtype=torch.bfloat16), "caption": f"b{j}"})
    c2.finalize_current_shard()

    all_lengths = lengths_a + lengths_b
    for i, (n, v) in enumerate(zip(all_lengths, [0.0, 1.0, 2.0, 100.0, 101.0])):
        row = c2[i]["prompt_embeds"]
        assert tuple(row.shape) == (n, 8)
        assert row.float().eq(v).all()  # right bytes at the right offset across the resume seam
    emb = (d / TENSORS_DIR / "prompt_embeds.bin").stat().st_size
    assert emb == sum(n * 8 * 2 for n in all_lengths)  # still zero padding after resume


def test_cache_ragged_null_row_costs_zero_bytes(tmp_path):
    """A ragged key absent on a row consumes NO bytes (no zero placeholder) and reads back None;
    the following real row's offset is still correct — the null gap doesn't corrupt anything."""
    d = tmp_path / "te_null"
    c = Cache(d, "fp-null")
    c.add({"prompt_embeds": torch.full((5, 8), 1.0, dtype=torch.bfloat16), "caption": "a"})
    c.add({"caption": "b"})  # no prompt_embeds -> null row
    c.add({"prompt_embeds": torch.full((9, 8), 2.0, dtype=torch.bfloat16), "caption": "c"})
    c.finalize_current_shard()

    assert tuple(c[0]["prompt_embeds"].shape) == (5, 8)
    assert c[0]["prompt_embeds"].float().eq(1.0).all()
    assert c[1]["prompt_embeds"] is None  # null row -> None, exactly as the fixed-stack path did
    assert tuple(c[2]["prompt_embeds"].shape) == (9, 8)  # offset survived the null gap
    assert c[2]["prompt_embeds"].float().eq(2.0).all()
    assert len(c) == 3
    # .bin holds only the two real rows — the null contributed zero bytes.
    emb = (d / TENSORS_DIR / "prompt_embeds.bin").stat().st_size
    assert emb == (5 + 9) * 8 * 2  # bf16 = 2 B/elt, zero padding


def test_cache_refresh_reads_interleaved_add_read(tmp_path):
    """Live read-back store pattern (TE dedup spill): rows added after a read become
    visible after refresh_reads(), including across a dim-0 growth."""
    cache = Cache(tmp_path / "spill", "fp-spill")
    cache.add({"prompt_embeds": torch.randn(5, 12, 8, dtype=torch.bfloat16), "caption": "a"})
    cache.refresh_reads()
    first = cache[0]["prompt_embeds"]
    assert tuple(first.shape) == (5, 12, 8)

    cache.add({"prompt_embeds": torch.randn(9, 12, 8, dtype=torch.bfloat16), "caption": "b"})  # grows bucket
    cache.add({"prompt_embeds": torch.randn(3, 12, 8, dtype=torch.bfloat16), "caption": "c"})
    cache.refresh_reads()
    assert tuple(cache[0]["prompt_embeds"].shape) == (5, 12, 8)
    assert torch.equal(cache[0]["prompt_embeds"], first)
    assert tuple(cache[1]["prompt_embeds"].shape) == (9, 12, 8)
    assert tuple(cache[2]["prompt_embeds"].shape) == (3, 12, 8)


def test_cache_dim0_growth_is_amortized(tmp_path, monkeypatch):
    """Regression: growing to each new max exactly rewrote the whole stack per new
    longest sequence — O(N^2) I/O that stalled TE caching for minutes (idle GPU) once
    the dedup spill reached GBs. Slack growth keeps full-file rewrites logarithmic."""
    from rengu_flow.utils import cache as cache_mod

    cache = Cache(tmp_path / "c", "fp-growth")
    rewrites = {"n": 0}
    orig = Cache._grow_tensor_dim0

    def counting_grow(self, key, new_d0):
        before = tuple(self.tensor_specs[key]["shape"]) if key in self.tensor_specs else None
        orig(self, key, new_d0)
        after = tuple(self.tensor_specs[key]["shape"])
        if before != after:
            rewrites["n"] += 1

    monkeypatch.setattr(cache_mod.Cache, "_grow_tensor_dim0", counting_grow)
    # Strictly increasing lengths 4..200: worst case for exact growth (one rewrite per add).
    lengths = list(range(4, 201))
    for n in lengths:
        cache.add({"prompt_embeds": torch.randn(n, 4, dtype=torch.bfloat16), "caption": str(n)})
    assert rewrites["n"] <= 12, f"{rewrites['n']} full-file rewrites for {len(lengths)} adds"
    # Rows survive every growth intact (true shapes come from per-item meta).
    cache.finalize_current_shard()
    for i, n in enumerate(lengths):
        assert tuple(cache[i]["prompt_embeds"].shape) == (n, 4)


def test_cache_manifest_ahead_of_meta_self_heals(tmp_path):
    """Regression (prod IndexError at train step): a kill between the manifest write and
    the SQLite commit left the manifest claiming rows the meta rolled back — the cache
    then read as complete, nothing re-encoded, and training crashed with
    'Cache index N out of range'. On open, the committed row count wins."""
    cache = Cache(tmp_path / "c", "fp-heal")
    for i in range(36):
        cache.add({"prompt_embeds": torch.randn(6, 4, dtype=torch.bfloat16), "caption": str(i)})
    # Simulate the kill window: manifest records count=36, then the process dies before
    # the SQLite commit (rollback loses everything after the last 128-item checkpoint).
    cache._write_manifest()
    cache._meta_con.rollback()
    cache._meta_con.close()
    cache._meta_con = None

    reopened = Cache(tmp_path / "c", "fp-heal")
    assert reopened.count == 0  # nothing was committed — heal to the durable truth
    # The tail simply re-encodes (resume path), ending complete and readable.
    for i in range(36):
        reopened.add({"prompt_embeds": torch.randn(6, 4, dtype=torch.bfloat16), "caption": str(i)})
    reopened.finalize_current_shard()
    final = Cache(tmp_path / "c", "fp-heal")
    assert final.count == 36
    assert final[35]["caption"] == "35"  # the exact read that crashed in prod


def test_cache_finalize_then_kill_keeps_all_rows(tmp_path):
    """finalize commits the meta before the manifest, so a finalized cache reopened
    after any crash still serves every row."""
    cache = Cache(tmp_path / "c", "fp-final")
    for i in range(5):
        cache.add({"prompt_embeds": torch.randn(3, 4, dtype=torch.bfloat16), "caption": str(i)})
    cache.finalize_current_shard()
    reopened = Cache(tmp_path / "c", "fp-final")
    assert reopened.count == 5
    assert reopened[4]["caption"] == "4"


def _write_manifest_by_hand(cache_dir, tensors: dict) -> None:
    """A manifest as an earlier Rengu wrote it (same FORMAT_VERSION; only the layout differs)."""
    from rengu_flow.utils.cache import FORMAT_VERSION

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / TENSORS_DIR).mkdir(exist_ok=True)
    (cache_dir / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "fingerprint": "fp-old",
                "reuse_key": None,
                "count": 3,
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )


_RAGGED_EMBEDS = {"ragged": True, "trailing_shape": [8], "dtype": "float32", "storage_dtype": "bfloat16"}
# Before fdad555 the (L,) image_pad_mask was a fixed-width stack, not a ragged row.
_FIXED_PAD_MASK = {"shape": [10], "dtype": "bool", "storage_dtype": "bool", "item_nbytes": 10}


def test_open_rejects_text_cache_with_fixed_width_sequence_key(tmp_path):
    cache_dir = tmp_path / "text_embeddings_1"
    _write_manifest_by_hand(
        cache_dir,
        {"prompt_embeds": _RAGGED_EMBEDS, "text_mask": dict(_RAGGED_EMBEDS, trailing_shape=[]),
         "image_pad_mask": _FIXED_PAD_MASK},
    )
    with pytest.raises(StaleCacheLayoutError) as exc:
        open_disk_cache(cache_dir, "fp-old")
    msg = str(exc.value)
    assert str(cache_dir) in msg and "'image_pad_mask'" in msg
    assert "earlier version of Rengu" in msg and "--regenerate_text_cache" in msg
    # Nothing was invalidated behind the user's back.
    assert json.loads((cache_dir / MANIFEST_NAME).read_text())["count"] == 3
    # Regenerating is the escape hatch: it opens the cache only to clear it.
    rebuilt = open_disk_cache(cache_dir, "fp-old", regenerate=True)
    rebuilt.clear()
    assert len(rebuilt) == 0
    open_disk_cache(cache_dir, "fp-old")  # the fresh (empty) cache no longer trips the guard


def _valid_text_rows(model: str, n: int) -> dict:
    """One text-embedding row as each model's text-encoder fn caches it."""
    if model == "sdxl":  # encoder 1 and 2 share a cache dir per index; both key sets here
        return {"prompt_embeds": torch.randn(77, 16), "prompt_embeds_2": torch.randn(77, 24),
                "pooled_prompt_embeds": torch.randn(24)}
    if model == "cosmos":  # token ids / masks are padded to the tokenizer's max length
        return {"prompt_embeds": torch.randn(n, 16), "attn_mask": torch.ones(12, dtype=torch.int64),
                "t5_input_ids": torch.ones(12, dtype=torch.int64), "t5_attn_mask": torch.ones(12, dtype=torch.int64)}
    if model == "krea2":  # compacted (L, layers, D) stack + (L,) mask
        return {"prompt_embeds": torch.randn(n, 3, 16), "text_mask": torch.ones(n, dtype=torch.bool)}
    if model == "qwen_t2i":
        return {"prompt_embeds": torch.randn(n, 16), "text_mask": torch.ones(n, dtype=torch.bool)}
    raise AssertionError(model)


@pytest.mark.parametrize("model", ["sdxl", "cosmos", "krea2", "qwen_t2i"])
def test_valid_text_caches_do_not_trip_the_layout_guard(tmp_path, model):
    cache_dir = tmp_path / model
    cache = open_disk_cache(cache_dir, "fp")
    for i, n in enumerate([5, 9, 7]):
        cache.add({**_valid_text_rows(model, n), "caption": f"c{i}"})
    cache.finalize_current_shard()
    cache.close()
    reopened = open_disk_cache(cache_dir, "fp")  # must not raise
    assert len(reopened) == 3
    reject_stale_sequence_layout(cache_dir)


def test_map_and_cache_guard_blocks_load_but_not_regenerate(tmp_path):
    """The caching path: a stale text cache stops the run on open/load; --regenerate_text_cache
    (regenerate_cache=True here) rebuilds it in the current layout."""
    import datasets

    from rengu_flow.data.cache_utils import _map_and_cache

    ds = datasets.Dataset.from_dict({"caption": ["a", "bb", "ccc"]})

    def map_fn(batch, rank):
        rows = [len(c) + 2 for c in batch["caption"]]
        return {
            "prompt_embeds": [torch.randn(n, 8) for n in rows],
            "image_pad_mask": [torch.ones(n, dtype=torch.bool) for n in rows],
            "caption": list(batch["caption"]),
        }

    _write_manifest_by_hand(
        tmp_path / "text_embeddings_1", {"prompt_embeds": _RAGGED_EMBEDS, "image_pad_mask": _FIXED_PAD_MASK}
    )
    with pytest.raises(StaleCacheLayoutError):
        _map_and_cache(ds, map_fn, tmp_path, cache_file_prefix="text_embeddings_1_")
    with pytest.raises(StaleCacheLayoutError):  # the trusted load after caching
        _map_and_cache(ds, None, tmp_path, cache_file_prefix="text_embeddings_1_")
    cache = _map_and_cache(ds, map_fn, tmp_path, cache_file_prefix="text_embeddings_1_", regenerate_cache=True)
    assert len(cache) == 3
    assert cache.tensor_specs["image_pad_mask"]["ragged"] is True
    cache.close()
    reject_stale_sequence_layout(tmp_path / "text_embeddings_1")
