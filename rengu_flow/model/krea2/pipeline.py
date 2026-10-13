"""Krea 2 training pipeline (Qwen3-VL text taps + Qwen-Image VAE + vendored Krea2 DiT).

Trains the open-weights Krea 2 checkpoints (``krea/Krea-2-Raw`` diffusers layout): full
finetune or any rengu adapter (LoRA / LoKr / LyCORIS catalog) on the DiT blocks. The VAE
and text encoder are always frozen; text embeddings must be cached (the tapped Qwen3-VL
stack is far too heavy to keep in the training graph).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rengu_flow.model import dit_common
from rengu_flow.model.dit_common.streaming import LazyStreamedEncoder
from rengu_flow.model.krea2.layers import FinalLayer, InitialLayer, TransformerLayer
from rengu_flow.model.krea2.text import (
    DEFAULT_SELECT_LAYERS,
    encode_prompts,
    pad_text_embeddings,
)
from rengu_flow.model.krea2 import loading
from rengu_flow.registry.models import register_model

# The model authors' recommended LoRA scope is every Linear in the DiT (per-block
# attention/MLP, text fusion, img_in/txt_in/time projections, final linear) — targeting
# the root class makes the adapter walkers collect them all. Narrow with
# adapter.target_include/exclude when needed.
ADAPTER_TARGET_MODULES = ("Krea2Transformer2DModel",)

# Adapter/full-model exports use the official Krea 2 LoRA convention ("transformer." +
# diffusers module names), which ComfyUI and diffusers both load.
EXPORT_PREFIX = "transformer."

# Named layer groups for adapter.layer_groups: globs over the DiT's dotted module
# paths (see networks/adapter_targets.py). "text_fusion" is the conditioning stack —
# Krea2TextFusion (2 layerwise blocks over the 12 tapped layers + 2 refiner blocks over
# the tokens) plus the txt_in projection (canonical checkpoint/diffusers name). The
# union of every group is the default all-linears scope.
ADAPTER_LAYER_GROUPS = {
    "text_fusion": ("text_fusion.*", "txt_in.*"),
    "attention": ("transformer_blocks.*.attn.*",),
    "feedforward": ("transformer_blocks.*.ff.*",),
    "time_modulation": ("time_embed.*", "time_mod_proj"),
    "image_in_out": ("img_in", "final_layer.*"),
}

# Quantization scope for the frozen base: the per-block attention/SwiGLU linears only.
# The text-fusion stack is small and delicate and the shared projections are tiny —
# keep them in compute dtype (same split musubi-tuner uses for its fp8 path).
QUANT_LEAF_NAMES = frozenset({"to_q", "to_k", "to_v", "to_gate", "0", "gate", "up", "down"})
QUANT_SKIP_SUBSTRINGS = (
    "text_fusion",
    "txt_in",
    "time_embed",
    "time_mod_proj",
    "img_in",
    "final_layer",
)

# Krea 2 resolution-aware timestep shift (matches the reference scheduler config:
# base_image_seq_len=256, max_image_seq_len=6400, base_shift=0.5, max_shift=1.15).
SHIFT_BASE_SEQ_LEN = 256
SHIFT_MAX_SEQ_LEN = 6400
SHIFT_BASE = 0.5
SHIFT_MAX = 1.15


def calculate_shift(image_seq_len: int) -> float:
    return dit_common.calculate_shift(
        image_seq_len, SHIFT_BASE_SEQ_LEN, SHIFT_MAX_SEQ_LEN, SHIFT_BASE, SHIFT_MAX
    )


def time_shift(mu: float, t: torch.Tensor) -> torch.Tensor:
    return dit_common.time_shift(mu, 1.0, t)


@register_model("krea2")
class Krea2Pipeline(dit_common.DiTPipeline):
    name = "krea2"
    checkpointable_layers = ["TransformerLayer"]
    adapter_target_modules = list(ADAPTER_TARGET_MODULES)
    adapter_layer_groups = ADAPTER_LAYER_GROUPS
    adapter_export_prefix = EXPORT_PREFIX
    pixels_round_to_multiple = 16  # VAE f8 x patch_size 2
    # The VAE is small: keep a reloaded one in host RAM between previews. The ~8 GB text
    # encoder caching freed is unloaded again (preview prompts are memoized).
    preview_keep_reloaded_vae = True
    preview_unload_freed_text_encoder = True

    # The adapter-composable frozen-base quantization (model.transformer_fp8_matmul / _4bit).
    quant_leaf_names = QUANT_LEAF_NAMES
    quant_skip_substrings = QUANT_SKIP_SUBSTRINGS
    quant_label = "Krea2"
    text_cache_required_reason = (
        ": the tapped Qwen3-VL hidden-state stack cannot run inside the training graph."
    )
    missing_component_message = (
        "model.{component}_path is required for {name}: point it at the .safetensors file (or "
        "folder) you downloaded for the {component}. Alternatively set model.{bundle} to a full "
        "diffusers-layout folder."
    )
    # GAS splits of mixed caption lengths keep the unmasked (flash) attention path.
    trim_micro_batch = dit_common.text_padding_trimmer()

    def __init__(self, config):
        self._init_pipeline(config)
        dtype = self.model_config["dtype"]
        self.max_sequence_length = int(self.model_config.get("max_sequence_length", 512))
        self.select_layers = tuple(self._pipeline_index().get("text_encoder_select_layers", DEFAULT_SELECT_LAYERS))

        self.vae = self._load_vae()
        self.tokenizer = loading.load_tokenizer(self.model_config.get("tokenizer_path"))
        text_encoder_path = self._component_path("text_encoder")
        # Lazy: read from disk only when captions (or new preview prompts) need encoding — a
        # run with a warm text cache never loads it. Streams its decoder layers from host RAM
        # when the ~8 GB encoder does not fit in free VRAM.
        self.text_encoder = LazyStreamedEncoder(
            lambda: loading.load_text_encoder(text_encoder_path, dtype),
            layers_of=lambda module: module.layers,
            name="Qwen3-VL text encoder",
        )

    def _pipeline_index(self) -> dict:
        checkpoint = self.model_config.get("checkpoint_path")
        if not checkpoint:
            return {}
        index = Path(checkpoint) / "model_index.json"
        if not index.exists():
            return {}
        try:
            return json.loads(index.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    # ---- DiTPipeline hooks ---------------------------------------------------------------

    def _load_transformer(self, dtype):
        return loading.load_transformer(self._component_path("transformer"), dtype)

    def _load_vae(self):
        return loading.load_vae(self._component_path("vae"), self.model_config["dtype"])

    def _encode_text(self, text_encoder, prompts, device):
        return encode_prompts(
            text_encoder,
            self.tokenizer,
            prompts,
            select_layers=self.select_layers,
            max_sequence_length=self.max_sequence_length,
            device=device,
        )

    # ---- training ------------------------------------------------------------------------

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs["latents"].float()
        prompt_embeds, text_mask = pad_text_embeddings(inputs["prompt_embeds"], inputs["text_mask"])
        h, w = latents.shape[-2:]
        # Krea 2 trains with a resolution-aware exponential time shift; a fixed model.shift
        # overrides the dynamic default.
        noisy_latents, t, target, mask = dit_common.flow_matching_batch(
            self.model_config, latents, inputs["mask"], timestep_quantile, mu=calculate_shift((h // 2) * (w // 2))
        )
        return (noisy_latents, t, prompt_embeds, text_mask), (target, mask)

    def to_layers(self):
        from rengu_flow.model.krea2.layers import RouteEndLayer, RouteStartLayer
        from rengu_flow.training.token_routing import parse_tread

        tread = parse_tread(self.config, len(self.transformer.transformer_blocks))
        route = tread.route if tread else None
        # Manual AC (true / an interval) also checkpoints the text branch; "auto" leaves the
        # save/recompute split to compile's partitioner.
        ac = self.config.get("activation_checkpointing", False)
        layers = [
            InitialLayer(
                self.transformer,
                checkpoint_text=bool(ac) and ac != "auto",
                pipe_parallel=int(self.config.get("pipeline_stages", 1)) > 1,
            )
        ]
        for i, block in enumerate(self.transformer.transformer_blocks):
            if route and i == route[0]:
                layers.append(RouteStartLayer(tread.drop_ratio, tread.disable_after_frac))
            layers.append(TransformerLayer(block, i, self.offloader))
            if route and i == route[1]:
                layers.append(RouteEndLayer())
        layers.append(FinalLayer(self.transformer))
        return layers

    # ---- previews (VAE/TE lifecycle, memoized prompts, memory: DiTPipeline) ---------------

    def generate_preview_image(self, preview_cfg: dict, prompt: str, step: int, seed: int):
        from rengu_flow.model.krea2.preview_sampling import generate_preview_image as _gen

        return _gen(self, preview_cfg, prompt, step, seed)
