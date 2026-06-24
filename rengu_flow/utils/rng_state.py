"""Capture/restore global RNG state across a resume.

So a resumed run reproduces the same stochastic data stream (augmentation, tag dropout,
uncond dropout, live shuffling) as the uninterrupted run instead of diverging from a fresh
RNG. Best-effort: with ``dataloader_num_workers > 0`` each worker reseeds at fork, so this is
exact only for ``num_workers == 0`` (the single-process data path); it still restores the
main-process RNG either way. CUDA RNG is skipped silently if the device count changed.
"""

from __future__ import annotations

from typing import Any


def capture_rng_state() -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte(t: Any) -> Any:
    import torch

    if isinstance(t, torch.Tensor):
        return t.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(t, dtype=torch.uint8)


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    import random

    import numpy as np
    import torch

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_byte(state["torch"]))
    cuda = state.get("torch_cuda")
    if cuda is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_cpu_byte(s) for s in cuda])
        except Exception:
            pass  # different device count / layout on resume -> leave CUDA RNG as-is
