"""Unit tests for the pure max_images rotation index mapping."""

from __future__ import annotations

import math

from rengu_flow.data.dataset import effective_sample_cap, rotation_window_index


def _window(epoch, pool_len, cap, static):
    """The pool indices served in a given epoch for slots 0..cap-1."""
    return [
        rotation_window_index(pos, epoch, pool_len, cap, static)
        for pos in range(cap)
    ]


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
