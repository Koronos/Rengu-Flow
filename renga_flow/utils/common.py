"""Common utilities (distributed rank, main process). Used by models and training loop."""

import gc

try:
    import torch
except ImportError:
    torch = None

try:
    import deepspeed.comm.comm as dist

    def get_rank() -> int:
        if dist.is_initialized():
            return dist.get_rank()
        return 0
except Exception:

    def get_rank() -> int:
        return 0


def is_main_process() -> bool:
    return get_rank() == 0


def empty_cuda_cache() -> None:
    """Run gc.collect() and torch.cuda.empty_cache() to free GPU memory (e.g. before eval)."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# For SDXL autocast; None = use default (no override).
AUTOCAST_DTYPE = None


def round_to_nearest_multiple(x: float, multiple: int) -> float:
    """Round x to the nearest multiple of multiple (e.g. for pixel dimensions)."""
    return round(x / multiple) * multiple


def round_down_to_multiple(x: float, multiple: int) -> float:
    """Round x down to the nearest multiple."""
    return (x // multiple) * multiple


def load_state_dict(path):
    """Load a checkpoint from .safetensors or .pt/.pth."""
    from pathlib import Path

    from safetensors import safe_open

    path = Path(path)
    path_str = str(path)
    if path_str.endswith(".safetensors"):
        tensors = {}
        with safe_open(path_str, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    return torch.load(path_str, weights_only=True)


def iterate_safetensors(path):
    """Yield (key, tensor) from a safetensors file or directory of shards."""
    from pathlib import Path

    from safetensors import safe_open

    path = Path(path)
    if path.is_dir():
        safetensors_files = list(path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files in directory {path}")
    else:
        if path.suffix != ".safetensors":
            raise ValueError(f"Expected {path} to be a safetensors file")
        safetensors_files = [path]
    for filename in safetensors_files:
        with safe_open(str(filename), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)
