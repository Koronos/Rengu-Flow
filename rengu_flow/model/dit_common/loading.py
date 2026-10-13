"""Local-file loading helpers shared by the DiT pipelines (krea2, qwen_image21, nanosaur2, ...).

rengu never downloads models: every component path is a file or folder the user already has.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from rengu_flow.config.validation import ConfigValidationError

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_COMFY_PREFIX = re.compile(r"^(model\.)?diffusion_model\.")


def require_exists(path: str | Path, what: str) -> Path:
    """``path`` as a ``Path``; a missing path is a config error naming ``model.<what>``."""
    p = Path(path)
    if not p.exists():
        raise ConfigValidationError(
            f"model.{what}: {p} does not exist. Point it at a local .safetensors file or "
            "folder you already downloaded — rengu never downloads models or resolves repo ids."
        )
    return p


def strip_comfy_prefix(state_dict: dict) -> dict:
    """Drop the ``diffusion_model.`` / ``model.diffusion_model.`` prefix some ComfyUI re-exports
    wrap DiT keys in."""
    return {_COMFY_PREFIX.sub("", k): v for k, v in state_dict.items()}


def read_safetensors(path: str | Path, dtype: torch.dtype | None = None, keep_dtypes=FP8_DTYPES) -> dict:
    """Read a ``.safetensors`` file tensor by tensor, casting each to ``dtype`` as it is read, so
    peak host RAM is one copy of the weights in ``dtype`` (not the file plus a cast copy). Tensors
    whose dtype is in ``keep_dtypes`` (fp8 by default, for pre-quantized guards) keep it."""
    from safetensors import safe_open

    state_dict = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if dtype is not None and tensor.dtype not in keep_dtypes:
                tensor = tensor.to(dtype)
            state_dict[key] = tensor
    return state_dict


def guard_not_prequantized(
    state_dict: dict,
    what: str,
    *,
    kind: str,
    advice: str,
    dtypes=FP8_DTYPES,
    key_suffixes: tuple[str, ...] = ("scale_weight",),
) -> None:
    """Refuse a pre-quantized checkpoint (it cannot be trained): any tensor whose dtype is in
    ``dtypes`` or any key ending in ``key_suffixes``. The error reads ``model.<what> points at a
    pre-quantized (<kind>) file, which cannot be trained<advice>`` — ``advice`` carries its own
    leading punctuation (". Use the bf16 file ..." / "; use ...")."""
    quantized = set(dtypes) & {v.dtype for v in state_dict.values()}
    marked = any(k.endswith(tuple(key_suffixes)) for k in state_dict)
    if quantized or marked:
        raise ConfigValidationError(
            f"model.{what} points at a pre-quantized ({kind}) file, which cannot be trained{advice}"
        )


def config_dir_or_bundled(path: str | Path, bundled: Path) -> Path:
    """Where a component's ``config.json`` lives: ``path`` when it is a folder that has one, else
    the ``bundled`` copy (single files carry no config)."""
    path = Path(path)
    return path if (not path.is_file() and (path / "config.json").exists()) else bundled


def load_file_or_folder(path: str | Path, what: str, *, folder, file):
    """Load a component the user points at with ``model.<what>``: ``folder(path)`` for a folder
    (diffusers/transformers layout), ``file(path)`` for a single file. A missing path is a
    config error (``require_exists``)."""
    path = require_exists(path, what)
    return file(path) if path.is_file() else folder(path)
