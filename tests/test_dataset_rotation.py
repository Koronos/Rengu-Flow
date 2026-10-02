"""Unit tests for the pure max_images rotation index mapping."""

from __future__ import annotations

import math

from rengu_flow.data.dataset import (
    FolderSubsampler,
    SizeBucketDataset,
    effective_sample_cap,
    rotation_window_index,
)


def _window(epoch, pool_len, cap, static):
    """The pool indices served in a given epoch for slots 0..cap-1."""
    return [
        rotation_window_index(pos, epoch, pool_len, cap, static)
        for pos in range(cap)
    ]


class _FakeBucket:
    """Minimal stand-in for the uncapped per-epoch reshuffle path of SizeBucketDataset._pool_index
    (no directory => no folder cap), without building a real cached dataset."""

    def __init__(self, pool_len, static=False):
        self._len = pool_len
        self.directory_dataset = None
        self.subsample_shuffle = not static
        self._epoch = 1
        self._epoch_order_seed = 4242
        self._epoch_order_cache = None
        self._epoch_order_for = None
        self._served_cache = None
        self._served_for = None

    @property
    def _pool_len(self):
        return self._len

    _served_rows = SizeBucketDataset._served_rows
    _effective_len = SizeBucketDataset._effective_len

    def _epoch_pool_order(self):
        return SizeBucketDataset._epoch_pool_order(self)

    def pool_index(self, idx):
        return SizeBucketDataset._pool_index(self, idx)


def test_uncapped_pool_index_covers_everything_each_epoch():
    b = _FakeBucket(pool_len=8)  # no cap
    for epoch in (1, 2, 5):
        b._epoch = epoch
        served = sorted(b.pool_index(i) for i in range(8))
        assert served == list(range(8))  # full coverage, nothing left out


def test_uncapped_pool_index_varies_per_epoch():
    b = _FakeBucket(pool_len=12)
    b._epoch = 1
    e1 = [b.pool_index(i) for i in range(12)]
    b._epoch = 2
    e2 = [b.pool_index(i) for i in range(12)]
    assert e1 != e2  # the slice dropped on a partial pass differs between epochs


# --- FolderSubsampler: the per-folder base-image cap (max_images / subsample_ratio) ---------


def test_folder_subsampler_caps_and_rotates_to_cover():
    keys = [f"img{i}" for i in range(10)]
    sub = FolderSubsampler(keys, cap=4, static=False, seed=0)
    seen: set = set()
    for epoch in range(1, math.ceil(10 / 4) + 1):
        sel = sub.selected(epoch)
        assert len(sel) == 4  # the folder contributes exactly `cap` base images per epoch
        seen.update(sel)
    assert seen == set(keys)  # rotation covers the whole folder


def test_folder_subsampler_frozen_when_static():
    keys = [f"img{i}" for i in range(10)]
    sub = FolderSubsampler(keys, cap=4, static=True, seed=0)
    assert sub.selected(1) == sub.selected(2) == sub.selected(5)


def test_folder_subsampler_small_folder_repeats_to_cap():
    sub = FolderSubsampler(["a", "b", "c"], cap=8, static=False, seed=0)
    sel = sub.selected(1)
    assert len(sel) == 8  # fills its quota by repeating
    assert set(sel) == {"a", "b", "c"}


def test_folder_subsampler_uncapped_returns_whole_folder():
    sub = FolderSubsampler(["a", "b", "c"], cap=None, static=False, seed=0)
    assert sorted(sub.selected(1)) == ["a", "b", "c"]


def test_no_cap_is_identity_modulo_pool():
    # cap=None -> behave as before: just wrap the position into the pool.
    assert [rotation_window_index(p, 3, 5, None, False) for p in range(7)] == [
        0, 1, 2, 3, 4, 0, 1
    ]


def test_static_uses_same_window_every_epoch():
    first = _window(1, 10, 4, static=True)
    assert first == [0, 1, 2, 3]
    for epoch in range(1, 6):
        assert _window(epoch, 10, 4, static=True) == first


def test_rotating_advances_window_each_epoch():
    assert _window(1, 10, 4, static=False) == [0, 1, 2, 3]
    assert _window(2, 10, 4, static=False) == [4, 5, 6, 7]
    # Wraps around when it runs off the end of the pool.
    assert _window(3, 10, 4, static=False) == [8, 9, 0, 1]


def test_rotating_covers_whole_pool():
    pool_len, cap = 10, 4
    epochs_needed = math.ceil(pool_len / cap)
    seen: set[int] = set()
    for epoch in range(1, epochs_needed + 1):
        seen.update(_window(epoch, pool_len, cap, static=False))
    assert seen == set(range(pool_len))


def test_repeat_to_n_when_pool_smaller_than_cap():
    # 3 images, cap 8 -> repeat the pool up to the cap.
    assert _window(1, 3, 8, static=False) == [0, 1, 2, 0, 1, 2, 0, 1]
    # Static behaves the same (offset 0) when repeating.
    assert _window(1, 3, 8, static=True) == [0, 1, 2, 0, 1, 2, 0, 1]


def test_empty_pool_is_safe():
    assert rotation_window_index(5, 3, 0, 4, False) == 0


def test_deterministic_for_same_inputs():
    a = rotation_window_index(2, 7, 13, 5, False)
    b = rotation_window_index(2, 7, 13, 5, False)
    assert a == b


def test_effective_sample_cap_max_images_wins():
    assert effective_sample_cap(100, 10, 1.0) == 10
    # Absolute cap takes precedence even if a ratio is also present.
    assert effective_sample_cap(100, 10, 0.5) == 10


def test_effective_sample_cap_from_ratio():
    assert effective_sample_cap(100, None, 0.25) == 25
    # Always at least 1 row when a ratio < 1 is set.
    assert effective_sample_cap(3, None, 0.1) == 1


def test_effective_sample_cap_none_when_unlimited():
    assert effective_sample_cap(100, None, 1.0) is None
    assert effective_sample_cap(100, None, 1.5) is None


# --- stratified quota: per-bucket length constant across epochs ------------------------------


def test_stratified_quotas_sum_to_cap_and_keep_every_stratum():
    from rengu_flow.data.dataset import _stratified_quotas

    assert sum(_stratified_quotas([60, 30, 10], 20)) == 20
    assert all(q >= 1 for q in _stratified_quotas([60, 30, 10], 20))
    assert _stratified_quotas([60, 30, 10], 100) == [60, 30, 10]
    assert sum(_stratified_quotas([5, 5, 5], 2)) == 2


def test_multi_bucket_cap_serves_constant_length_without_duplicates():
    """max_images over several AR buckets: each bucket's served length is frozen at post_init, so
    it must not move with the epoch (it did -> duplicates, skipped images, empty bucket on row 0)."""
    import collections

    buckets = {
        "sq": [f"a{i}" for i in range(60)],
        "port": [f"b{i}" for i in range(30)],
        "land": [f"c{i}" for i in range(10)],
    }
    groups = {(None, k): name for name, v in buckets.items() for k in v}
    sub = FolderSubsampler(list(groups), cap=20, static=False, seed=123, groups=groups)

    class DD:
        def folder_subsampler(self):
            return sub

    sbs = {}
    for name, keys in buckets.items():
        sb = object.__new__(SizeBucketDataset)
        sb.iteration_order = {"image_spec": [(None, k) for k in keys]}
        sb.directory_dataset = DD()
        sb._epoch = 1
        sb._served_cache = None
        sb._served_for = None
        sb._rows_by_base_cache = None
        sb._base_keys_cache = None
        sb.subsample_shuffle = True
        sb.num_repeats = 1
        sbs[name] = sb
    slots = {n: len(sb) for n, sb in sbs.items()}
    assert sum(slots.values()) == 20
    for epoch in range(1, 8):
        seen = collections.Counter()
        for n, sb in sbs.items():
            sb.set_epoch(epoch)
            assert len(sb) == slots[n], (epoch, n)
            for j in range(slots[n]):
                seen[sb.iteration_order["image_spec"][sb._pool_index(j)][1]] += 1
        assert max(seen.values()) == 1, epoch  # no duplicate within an epoch
        assert set(seen) == {k for _, k in sub.selected(epoch)}  # nothing selected is skipped


def test_stratified_rotation_covers_every_image():
    keys = [(None, f"x{i}") for i in range(12)]
    groups = {k: ("a" if i < 8 else "b") for i, k in enumerate(keys)}
    sub = FolderSubsampler(keys, cap=6, static=False, seed=1, groups=groups)
    seen = set()
    for epoch in range(1, 5):
        seen.update(sub.selected(epoch))
    assert seen == set(keys)


def test_order_depends_on_epoch_even_without_a_cap():
    """Persistent workers need re-forking whenever the order moves with the epoch -- the default
    subsample_shuffle reshuffles the whole pool per epoch even with no max_images."""
    from rengu_flow.data.dataset import order_depends_on_epoch

    assert order_depends_on_epoch({"path": "x"})
    assert order_depends_on_epoch({"path": "x", "max_images": 10})
    assert not order_depends_on_epoch({"path": "x", "subsample_shuffle": False})
