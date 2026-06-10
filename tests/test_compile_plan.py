"""Tests for shape-aware torch.compile planning (training/compile_plan.py)."""

from __future__ import annotations

from rengu_flow.data.dataset import Dataset
from rengu_flow.training.compile_plan import (
    DEFAULT_ACCUMULATED_LIMIT,
    DEFAULT_CACHE_SIZE_LIMIT,
    CompilePlan,
    apply_dynamo_limits,
    plan_compile,
)

# --- plan_compile ---------------------------------------------------------------


def test_unknown_shape_count_keeps_legacy_kwargs():
    plan = plan_compile({}, None)
    assert plan.kwargs == {}
    assert plan.cache_size_limit is None
    assert plan.accumulated_cache_size_limit is None


def test_single_shape_forces_static_without_raising_limits():
    plan = plan_compile({}, 1)
    assert plan.kwargs == {"dynamic": False}
    assert plan.cache_size_limit is None
    assert plan.accumulated_cache_size_limit is None


def test_multi_shape_static_specialization_raises_limits():
    plan = plan_compile({}, 12)
    assert plan.kwargs == {"dynamic": False}
    assert plan.cache_size_limit == 14  # shapes + margin
    assert plan.accumulated_cache_size_limit > DEFAULT_ACCUMULATED_LIMIT
    assert any("static" in n for n in plan.notes)
    assert any("cache_size_limit" in n for n in plan.notes)


def test_few_shapes_fit_default_cache_limit():
    plan = plan_compile({}, 3)
    assert plan.kwargs == {"dynamic": False}
    # 3 + margin <= 8: defaults suffice, no limit override.
    assert plan.cache_size_limit is None
    assert plan.accumulated_cache_size_limit is None


def test_explicit_dynamic_is_respected_and_still_sized():
    plan = plan_compile({"compile_dynamic": True}, 12)
    assert plan.kwargs == {"dynamic": True}
    # Dynamic mode still recompiles per resolution bucket; budget for it.
    assert plan.cache_size_limit == 14
    assert not any("static" in n for n in plan.notes)


def test_compile_mode_is_passed_through():
    plan = plan_compile({"compile_mode": "reduce-overhead"}, 2)
    assert plan.kwargs == {"mode": "reduce-overhead", "dynamic": False}


# --- apply_dynamo_limits ----------------------------------------------------------


def test_apply_dynamo_limits_noop_without_overrides():
    # Must not import/alter torch._dynamo state when defaults suffice.
    apply_dynamo_limits(CompilePlan())


def test_apply_dynamo_limits_raises_but_never_lowers():
    import torch._dynamo

    cfg = torch._dynamo.config
    before_cache, before_acc = cfg.cache_size_limit, cfg.accumulated_cache_size_limit
    try:
        plan = plan_compile({}, 12)
        apply_dynamo_limits(plan)
        assert cfg.cache_size_limit == 14
        assert cfg.accumulated_cache_size_limit == 14 * 24

        # A second, smaller plan must not lower the limits already in place.
        apply_dynamo_limits(CompilePlan(cache_size_limit=10, accumulated_cache_size_limit=256))
        assert cfg.cache_size_limit == 14
        assert cfg.accumulated_cache_size_limit == 14 * 24
    finally:
        cfg.cache_size_limit = before_cache
        cfg.accumulated_cache_size_limit = before_acc


def test_default_constants_match_torch():
    import torch._dynamo  # noqa: F401  (import resets nothing; reads live defaults)

    # Guard against torch changing its defaults under us silently.
    assert DEFAULT_CACHE_SIZE_LIMIT == 8
    assert DEFAULT_ACCUMULATED_LIMIT == 256


# --- Dataset.distinct_size_buckets ------------------------------------------------


class _FakeMetadata:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class _FakeSizeBucket:
    def __init__(self, size_bucket, n):
        self.size_bucket = size_bucket
        self.metadata_dataset = _FakeMetadata(n)


class _FakeDirectory:
    def __init__(self, buckets):
        self._buckets = buckets

    def get_size_bucket_datasets(self):
        return self._buckets


def test_distinct_size_buckets_dedups_and_skips_empty():
    ds = object.__new__(Dataset)
    ds.directory_datasets = [
        _FakeDirectory(
            [
                # 4-tuple naming buckets (ar, w, h, frames) from AR bucketing.
                _FakeSizeBucket((1.0, 512, 512, 1), 5),
                _FakeSizeBucket((1.0, 1024, 1024, 1), 5),
                _FakeSizeBucket((2.0, 1440, 736, 1), 0),  # empty -> ignored
            ]
        ),
        _FakeDirectory(
            [
                # Same pixel bucket from another directory -> deduped.
                _FakeSizeBucket((1.0, 512, 512, 1), 2),
                # 3-tuple bucket (explicit size_buckets path).
                _FakeSizeBucket((768, 768, 1), 1),
            ]
        ),
    ]
    assert ds.distinct_size_buckets() == {
        (512, 512, 1),
        (1024, 1024, 1),
        (768, 768, 1),
    }
