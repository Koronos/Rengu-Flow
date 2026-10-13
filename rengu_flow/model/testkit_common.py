"""Shared pieces of the per-model tiny test kits (``rengu_flow/model/<m>/testkit.py``).

A test kit builds a *tiny random* version of a model through its real loaders (files written to a
temp folder in the formats the loaders accept) so the contract suite (``tests/contract/``) and the
mini smokes can exercise every registered model on CPU without real weights. Production code, kept
light: no pytest, torch only inside functions that need it.

A kit module exposes:

- ``tiny_config(tmp_path, **over) -> dict`` — writes the tiny weights under ``tmp_path`` (reused
  when already there) and returns a training config (``[model]`` plus top-level keys);
- ``tiny_pipeline(config)`` — the real pipeline with its diffusion model loaded (and the
  ``[adapter]`` of ``config`` configured, when present);
- ``monolithic_forward(pipeline, features)`` — the model's own forward on ``prepare_inputs``
  features, shaped like the output of ``to_layers()``;
- ``cases()`` — the :class:`ModelCase` s the model trains on (t2i / edit / video);
- ``SMOKE_FIXTURE_EXEMPT`` — optional reason string: the model has no real-weights smoke fixture
  ``tests/fixtures/smoke/train_<type>.toml`` (otherwise one is required);
- ``CAPABILITIES`` — which contract checks apply (see :data:`ALL_CAPABILITIES`), and
  ``NOT_APPLICABLE`` — ``{capability: reason}`` for every one it leaves out (the skip says why;
  ``CAPABILITIES == ALL_CAPABILITIES - NOT_APPLICABLE.keys()`` is checked).

Optional, with DiT defaults in the contract suite (``tests/contract/conftest.py``):

- ``denoiser(pipeline)`` — the trained network (default ``pipeline.transformer``);
- ``adapter_networks(pipeline)`` — ``{original_name prefix: module}`` the adapters attach to
  (default ``{"": pipeline.transformer}``);
- ``adapter_scope(pipeline, adapter)`` — the module names an adapter export must cover, in export
  naming without ``spec.export_prefix`` (default: the linears of ``adapter_target_modules``
  minus ``ADAPTER_SCOPE_EXCLUDES[adapter]``);
- ``ADAPTER_SCOPE_EXCLUDES`` — ``{adapter: (module prefix, ...)}`` a family legitimately leaves
  out of the default scope; the other families must keep those modules frozen by default;
- ``reload_export(config, save_dir)`` — a pipeline loaded from ``save_model``'s output (default:
  ``tiny_pipeline`` with ``transformer_path = export_transformer_path(save_dir)``);
- ``RAGGED_CAPTIONS`` — captions whose text rows have different lengths (default
  :data:`SEAM_CAPTIONS` ``[:4]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Contract checks a kit can opt out of (a model that cannot honor one declares it here; the
# contract suite never branches on the model type).
#   disk_loaders     tiny weights go through the real loaders (tiny_config writes real files)
#   lazy_text_encoder  __init__ does not read the text encoder (warm text caches never load it)
#   ragged_text      the text-encoder fn caches one row per caption at its own length
#   flow_target      prepare_inputs: noisy = x + t * target (rectified flow, velocity target)
#   dit_preview      generate_preview_image(cfg, prompt, step, seed) + memoized prompt embeds
#   preview          previews render on CPU through utils.preview.run_previews (the spec's
#                    runner); the kit's preview_config(cfg) gives the [preview] sampler keys
#   save_round_trip  save_model's output loads back through the model's loader (identical weights)
#   adapters         lora / lokr / lycoris_locon attach through configure_adapter, export the
#                    declared scope and reload
ALL_CAPABILITIES = frozenset(
    {
        "disk_loaders",
        "lazy_text_encoder",
        "ragged_text",
        "flow_target",
        "dit_preview",
        "preview",
        "save_round_trip",
        "adapters",
    }
)

# Captions of very different token lengths (incl. an empty one): ragged text rows.
SEAM_CAPTIONS = (
    "a red fox sits on the mossy rock at dusk under a cloudy sky",
    "girl with blue ears",
    "",
    "forest sunlight",
    "a cat and a dog under the blue sky at dusk near a wooden house by the river",
    "red fox",
)

# Adapter configs the contract attaches (rengu config shape; dtype added by the kit).
ADAPTER_CONFIGS: dict[str, dict[str, Any]] = {
    "lora": {"type": "lora", "rank": 4, "alpha": 4, "dropout": 0.0},
    "lokr": {"type": "lokr", "rank": 4, "alpha": 4, "factor": -1, "decompose_both": False, "full_matrix": False},
    "lycoris_locon": {"type": "lycoris_locon", "rank": 4, "alpha": 4},
}

# Top-level training-config keys ``tiny_config(**over)`` routes outside ``[model]``.
TOP_LEVEL_KEYS = frozenset({"adapter", "tread", "activation_checkpointing", "pipeline_stages", "optimizer"})


@dataclass(frozen=True)
class ModelCase:
    """One kind of training sample a model takes.

    ``write_dataset(root)`` writes real media (+ captions) under ``root`` and returns the dataset
    config (``resolutions``, buckets, ``[[directory]]``) the data path caches through the tiny
    VAE / text encoder. ``sample_batch(pipeline, seed)`` returns a synthetic batch in the cached
    layout (what ``Dataset.__getitem__`` yields) for the per-pipeline checks; its rows have
    different caption lengths. ``expected_grids`` = target latent grids (h, w) the dataset must
    produce; ``controls`` = condition images per sample (edit). ``write_dataset=None``: the case has
    no CPU-encodable media (video), so only the per-pipeline checks run it.
    """

    name: str
    write_dataset: Callable[[Path], dict] | None
    sample_batch: Callable[[Any, int], dict]
    expected_grids: frozenset = field(default_factory=frozenset)
    controls: int = 0


def split_config(over: dict) -> tuple[dict, dict]:
    """``(model_overrides, top_level_overrides)`` of ``tiny_config``'s ``**over``."""
    model = {k: v for k, v in over.items() if k not in TOP_LEVEL_KEYS}
    top = {k: v for k, v in over.items() if k in TOP_LEVEL_KEYS}
    return model, top


def training_config(model_section: dict, over: dict) -> dict:
    """A minimal training config around ``model_section`` with ``over`` applied."""
    model_over, top = split_config(over)
    config = {
        "model": {**model_section, **model_over},
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "pipeline_stages": 1,
    }
    config.update(top)
    return config


def prepare_adapter_config(adapter: dict, dtype) -> dict:
    """The ``[adapter]`` section as ``configure_adapter`` receives it after config loading
    (torch dtype, LyCORIS defaults)."""
    adapter = dict(adapter)
    adapter.setdefault("dtype", dtype)
    if str(adapter.get("type", "")).startswith("lycoris_"):
        from rengu_flow.networks.lycoris_meta import apply_lycoris_defaults

        apply_lycoris_defaults(adapter)
    return adapter


def randomize_(module, std: float = 0.1, seed: int = 0) -> None:
    """Checkpoint-like weights: every floating parameter random (norm scales, modulation tables and
    zero-initialized projections too), so no path of the tiny model is an identity or a zero."""
    import torch

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * std)


def tiny_qwen3vl(hidden: int = 32, layers: int = 3, seed: int = 0):
    """A tiny ``Qwen3VLForConditionalGeneration`` (krea2 / qwen_image21 text encoder) with the
    real vocabulary (the bundled tokenizer), mRoPE, and a vision tower with the release's
    16-px patch / 2x2 merge geometry and deepstack (so the real Qwen3-VL processor fits)."""
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(seed)
    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=64,
            vocab_size=151936,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            rope_theta=5000000,
        ),
        vision_config=dict(
            depth=3,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=hidden,
            deepstack_visual_indexes=[0, 2],
        ),
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    with torch.no_grad():  # exercise the (default zero/one-initialized) norms too
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(0.05 * torch.randn_like(p))
    return model


def write_qwen3vl_folder(folder: Path, model) -> None:
    """A transformers-layout text-encoder folder (config.json + model.safetensors, no LM head)."""
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items() if not k.startswith("lm_head.")}
    save_file(state, str(folder / "model.safetensors"))
    model.config.save_pretrained(folder)


def write_image(path: Path, size: tuple[int, int], seed: int) -> None:
    """A random RGB PNG of ``size`` (w, h)."""
    import torch
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    w, h = size
    Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()).save(path)


def write_t2i_tree(root: Path, base: int) -> dict:
    """Square, landscape (2:1) and portrait (1:2) images at ``base`` px with captions of mixed
    lengths (one empty), two per bucket; returns the dataset config (3 AR buckets)."""
    sizes = {"sq": (base, base), "wide": (base * 2, base), "tall": (base, base * 2)}
    for i, (stem, caption) in enumerate(zip(("sq_a", "sq_b", "wide_a", "wide_b", "tall_a", "tall_b"), SEAM_CAPTIONS)):
        write_image(root / f"{stem}.png", sizes[stem.split("_")[0]], i)
        (root / f"{stem}.txt").write_text(caption, encoding="utf-8")
    return {
        "resolutions": [base],
        "enable_ar_bucket": True,
        "min_ar": 0.5,
        "max_ar": 2.0,
        "num_ar_buckets": 3,
        "directory": [{"path": str(root), "num_repeats": 1}],
    }


def run_layers(layers, features, on_output: Callable[[int, Any], None] | None = None):
    """Run ``features`` through the pipeline layers in order (``on_output(i, x)`` sees every
    inter-layer value)."""
    x = features
    for i, layer in enumerate(layers):
        x = layer(x)
        if on_output is not None:
            on_output(i, x)
    return x
