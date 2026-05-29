# Isolated RNG context for reproducible eval (save/restore state on exit).
# Logic aligned with diffusion-pipe utils/isolate_rng (PyTorch Lightning-style).

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import torch
import numpy as np
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state


def _collect_rng_states(include_cuda: bool = True) -> dict[str, Any]:
    """Collect global RNG state for torch, numpy, and Python."""
    states = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": python_get_rng_state(),
    }
    if include_cuda:
        try:
            states["torch.cuda"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            pass
    return states


def _set_rng_states(rng_state_dict: dict[str, Any]) -> None:
    """Restore global RNG state from a previously collected dict."""
    torch.set_rng_state(rng_state_dict["torch"])
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))


@contextmanager
def isolate_rng(include_cuda: bool = True) -> Generator[None, None, None]:
    """Context manager that restores RNG state on exit. Use with fixed seeds inside for reproducible eval."""
    states = _collect_rng_states(include_cuda)
    try:
        yield
    finally:
        _set_rng_states(states)
