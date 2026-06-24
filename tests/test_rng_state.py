"""Capture/restore of global RNG state for bit-identical resume."""

from __future__ import annotations

import random

import numpy as np
import torch

from rengu_flow.utils.rng_state import capture_rng_state, restore_rng_state


def _draw():
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_rng_round_trip_reproduces_stream():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    [random.random() for _ in range(5)]
    np.random.rand(5)
    torch.rand(5)

    state = capture_rng_state()
    expected = _draw()

    # Diverge the RNG as a fresh (resumed) process would.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    _draw()

    restore_rng_state(state)
    assert _draw() == expected


def test_restore_none_is_noop():
    random.seed(7)
    before = random.random()
    restore_rng_state(None)
    random.seed(7)
    assert random.random() == before
