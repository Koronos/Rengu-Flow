"""Qwen3-VL text-decoder loading shared by the DiT pipelines that condition on it (krea2,
qwen_image21).

Text-to-image conditioning only runs the *text* decoder (``Qwen3VLTextModel``) — the vision
tower and the LM head are never used — so single files and sharded folders are reduced to the
text-decoder keys and loaded into a ``Qwen3VLTextModel`` built on ``meta``. Image-conditioned
(edit) encoding additionally needs the vision tower (``Qwen3VLVisionModel``, with its deepstack
mergers), which :func:`load_qwen3vl_vision_model` reads the same way from the ``visual.*`` keys.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _text_key(key: str) -> str | None:
    """Checkpoint key -> ``Qwen3VLTextModel`` key, or ``None`` when it is not a text-decoder key.

    Accepts the transformers layout (``model.language_model.layers.N...``), the ComfyUI /
    official single-file layout (``model.layers.N...``) and bare keys (``layers.N...``)."""
    key = re.sub(r"^model\.", "", key)
    if key.startswith(("visual.", "lm_head.")) or key.endswith((".weight_scale", ".comfy_quant")):
        return None
    if key.startswith("language_model."):
        key = key[len("language_model.") :]
    return key


def remap_qwen3vl_text_state_dict(state_dict: dict, dtype: torch.dtype) -> dict:
    """Reduce a Qwen3-VL checkpoint to ``Qwen3VLTextModel`` keys in ``dtype``.

    ComfyUI "scaled fp8" files store each quantized Linear as an fp8 ``.weight`` plus a scalar
    ``.weight_scale`` and a ``.comfy_quant`` marker: those are dequantized to ``dtype``; vision,
    LM-head and marker entries are dropped."""
    scales = {
        k[: -len(".weight_scale")]: v.float()
        for k, v in state_dict.items()
        if k.endswith(".weight_scale")
    }
    remapped = {}
    for k, v in state_dict.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        new = _text_key(k)
        if new is None:
            continue
        if v.dtype in _FP8_DTYPES:
            v = v.float() * scales.get(base, torch.tensor(1.0))
        remapped[new] = v.to(dtype)
    return remapped


def checkpoint_files(path: str | Path) -> list[Path]:
    """The ``.safetensors`` file(s) of a checkpoint: the file itself, the shards named by a
    ``*.safetensors.index.json`` in the folder, or every ``*.safetensors`` in the folder."""
    path = Path(path)
    if path.is_file():
        return [path]
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if indexes:
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        return [path / name for name in sorted(set(weight_map.values()))]
    return sorted(path.glob("*.safetensors"))


def load_qwen3vl_text_model(files: list[Path], text_config, dtype: torch.dtype):
    """Build a ``Qwen3VLTextModel`` from ``text_config`` and fill it from ``files`` shard by
    shard (only the text-decoder tensors are read). Returns it in eval mode, frozen."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import Qwen3VLTextModel

    state_dict = {}
    for file in files:
        with safe_open(str(file), framework="pt") as handle:
            keys = list(handle.keys())
            # A scaled-fp8 file needs its scale siblings to dequantize; read those too.
            wanted = [k for k in keys if _text_key(k) is not None or k.endswith(".weight_scale")]
            state_dict.update({k: handle.get_tensor(k) for k in wanted})
    remapped = remap_qwen3vl_text_state_dict(state_dict, dtype)
    del state_dict
    # include_buffers=False: params land on meta (replaced below by assign) but the
    # non-persistent rotary inv_freq buffer (absent from checkpoints) is computed for real.
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLTextModel._from_config(text_config)
    model.load_state_dict(remapped, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model


def _vision_key(key: str) -> str | None:
    """Checkpoint key -> ``Qwen3VLVisionModel`` key, or ``None`` when it is not a vision-tower key
    (``model.visual.X`` in the transformers layout, ``visual.X`` in single files)."""
    key = re.sub(r"^model\.", "", key)
    if not key.startswith("visual.") or key.endswith((".weight_scale", ".comfy_quant")):
        return None
    return key[len("visual.") :]


def vision_checkpoint_files(path: str | Path) -> list[Path]:
    """``checkpoint_files`` narrowed to the shards that hold ``visual.*`` tensors when a
    ``*.safetensors.index.json`` says which (the vision tower sits in one shard of four; opening a
    shard maps the whole file, which Windows charges against the commit limit)."""
    path = Path(path)
    files = checkpoint_files(path)
    indexes = sorted(path.glob("*.safetensors.index.json")) if path.is_dir() else []
    if not indexes:
        return files
    weight_map = json.loads(indexes[0].read_text())["weight_map"]
    wanted = {name for key, name in weight_map.items() if _vision_key(key) is not None}
    return [f for f in files if f.name in wanted] or files


def load_qwen3vl_vision_model(files: list[Path], vision_config, dtype: torch.dtype):
    """Build a ``Qwen3VLVisionModel`` (patch embed, blocks, merger and the deepstack mergers of
    ``vision_config.deepstack_visual_indexes``) from ``vision_config`` and fill it from the
    ``visual.*`` tensors of ``files``. Returns it in eval mode, frozen. Raises ``ValueError``
    when the checkpoint carries no vision tower (e.g. a text-only re-export)."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    state_dict = {}
    for file in files:
        with safe_open(str(file), framework="pt") as handle:
            for k in handle.keys():
                new = _vision_key(k)
                if new is not None:
                    state_dict[new] = handle.get_tensor(k).to(dtype)
    if not state_dict:
        raise ValueError(
            "the text-encoder checkpoint has no Qwen3-VL vision tower (no visual.* tensors); "
            "image-conditioned encoding needs the full Qwen3-VL checkpoint."
        )
    # include_buffers=False: the rotary inv_freq buffer is not in the checkpoint and stays real.
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLVisionModel._from_config(vision_config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model
