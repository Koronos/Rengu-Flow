"""Tests for isolate_rng (RNG save/restore for reproducible eval). No GPU."""

import random

import numpy as np
import torch

from rengu_flow.utils.isolate_rng import isolate_rng


def test_isolate_rng_restores_torch_state():
    """After leaving isolate_rng, torch RNG continues from the state at entry (not from inside)."""
    torch.manual_seed(42)
    stream = [torch.rand(1).item() for _ in range(6)]
    torch.manual_seed(42)
    before = [torch.rand(1).item() for _ in range(3)]
    with isolate_rng(include_cuda=False):
        _ = [torch.rand(1).item() for _ in range(5)]
    after = [torch.rand(1).item() for _ in range(3)]
    assert before == stream[:3]
    assert after == stream[3:6]


def test_isolate_rng_restores_numpy_state():
    """After leaving isolate_rng, numpy RNG continues from the state at entry."""
    np.random.seed(123)
    stream = [np.random.rand() for _ in range(6)]
    np.random.seed(123)
    before = [np.random.rand() for _ in range(3)]
    with isolate_rng(include_cuda=False):
        _ = [np.random.rand() for _ in range(5)]
    after = [np.random.rand() for _ in range(3)]
    assert before == stream[:3]
    assert after == stream[3:6]


def test_isolate_rng_restores_python_random_state():
    """After leaving isolate_rng, random module continues from the state at entry."""
    random.seed(99)
    stream = [random.random() for _ in range(6)]
    random.seed(99)
    before = [random.random() for _ in range(3)]
    with isolate_rng(include_cuda=False):
        _ = [random.random() for _ in range(5)]
    after = [random.random() for _ in range(3)]
    assert before == stream[:3]
    assert after == stream[3:6]


def test_isolate_rng_isolates_inner_seed():
    """Seeds set inside isolate_rng do not affect RNG after exit."""
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    with isolate_rng(include_cuda=False):
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
    # Outer state should be unchanged (still at seed 1)
    torch.manual_seed(1)
    first_after = torch.rand(1).item()
    torch.manual_seed(1)
    second_after = torch.rand(1).item()
    assert first_after == second_after
