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
import torch.nn.functional as F
from torch import nn

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.data.preprocess_media import PreprocessMediaFile
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
from rengu_flow.utils.save_io import atomic_save_safetensors

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

    def __init__(self, config):
        self.config = config
        self.model_config = config["model"]
        self._init_block_swap_state()
        dtype = self.model_config["dtype"]

        if not self.model_config.get("cache_text_embeddings", True):
            raise ConfigValidationError(
                "krea2 requires cache_text_embeddings = true: the tapped Qwen3-VL hidden-state "
                "stack cannot run inside the training graph."
            )
        self.cache_text_embeddings = True
        self.max_sequence_length = int(self.model_config.get("max_sequence_length", 512))
        self.select_layers = tuple(self._pipeline_index().get("text_encoder_select_layers", DEFAULT_SELECT_LAYERS))

        self.vae = loading.load_vae(self._component_path("vae"), dtype)
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
        self.transformer = None
        self._preview_embed_cache: dict = {}

    def _component_path(self, component: str) -> str:
        """Resolve a component to what the user assigned: ``model.<component>_path`` (a local
        .safetensors file or folder), or the ``<component>`` subfolder of an optional
        ``model.checkpoint_path`` diffusers folder. Never a repo id, never downloaded."""
        override = self.model_config.get(f"{component}_path")
        if override:
            return str(override)
        checkpoint = self.model_config.get("checkpoint_path")
        if not checkpoint:
            raise ConfigValidationError(
                f"model.{component}_path is required for krea2: point it at the .safetensors "
                f"file (or folder) you downloaded for the {component}. Alternatively set "
                "model.checkpoint_path to a full diffusers-layout folder."
            )
        return str(Path(checkpoint) / component)

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

    def load_diffusion_model(self, *, force: bool = False) -> None:
        if self.transformer is not None and not force:
            return
        dtype = self.model_config["dtype"]
        transformer_dtype = self.model_config.get("transformer_dtype", dtype)
        self.transformer = loading.load_transformer(
            self._component_path("transformer"), transformer_dtype
        )
        self._maybe_quantize_frozen_dit()
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if "adapter" not in self.config:
                p.requires_grad_(True)

    def _maybe_quantize_frozen_dit(self) -> None:
        """Optionally quantize the frozen DiT's matmul linears (same knobs as cosmos):
        ``model.transformer_fp8_matmul`` / ``model.transformer_4bit`` (mutually exclusive).
        The base stays frozen; the quantization-aware ``lokr`` adapter composes on top."""
        dit_common.quantize_frozen_dit(
            self.transformer,
            self.model_config,
            leaf_names=QUANT_LEAF_NAMES,
            skip_substrings=QUANT_SKIP_SUBSTRINGS,
            label="Krea2",
        )

    # ---- caching hooks -------------------------------------------------------------------

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def get_preprocess_media_file_fn(self, augmentation_resolver=None):
        return PreprocessMediaFile(
            self.config, support_video=False, augmentation_resolver=augmentation_resolver
        )

    def _latent_stats(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1)
        return mean, std

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            # Qwen-Image VAE is video-shaped: (B, C, T=1, H, W) in, (B, 16, 1, h, w) out.
            # The distribution's mode: a cached latent is reused every epoch, so one frozen
            # random draw would bake that noise in for the whole run.
            latents = vae.encode(tensor.unsqueeze(2)).latent_dist.mode().squeeze(2)
            mean, std = self._latent_stats(latents.device, latents.dtype)
            return {"latents": (latents - mean) / std}

        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            device = next(text_encoder.parameters()).device
            embeds, mask = encode_prompts(
                text_encoder,
                self.tokenizer,
                captions,
                select_layers=self.select_layers,
                max_sequence_length=self.max_sequence_length,
                device=device,
            )
            # One row per caption at its own valid length: a padded batch tensor would store
            # every row at the caching batch's longest caption (False tails), which then
            # stack into padded — masked-attention — training batches.
            return {
                "prompt_embeds": [e[m] for e, m in zip(embeds, mask)],
                "text_mask": [m[m] for m in mask],
            }

        return fn

    # ---- training ------------------------------------------------------------------------

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs["latents"].float()
        mask = inputs["mask"]
        prompt_embeds, text_mask = pad_text_embeddings(inputs["prompt_embeds"], inputs["text_mask"])

        bs, _channels, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode="nearest-exact")

        t = dit_common.sample_timesteps(self.model_config, bs, latents.device, timestep_quantile)
        # Krea 2 trains with a resolution-aware exponential time shift; a fixed model.shift
        # overrides the dynamic default.
        t = dit_common.shift_timesteps(
            t, self.model_config.get("shift", None), calculate_shift((h // 2) * (w // 2))
        )
        noisy_latents, target, t = dit_common.add_flow_noise(latents, t)

        return (noisy_latents, t, prompt_embeds, text_mask), (target, mask)

    def trim_micro_batch(self, features):
        """PipelineDataLoader hook: drop the text lanes this micro-batch does not use, so a
        GAS split of mixed caption lengths still gets the unmasked (flash) attention path."""
        return dit_common.trim_text_padding(features)

    def to_layers(self):
        from rengu_flow.model.krea2.layers import RouteEndLayer, RouteStartLayer
        from rengu_flow.training.token_routing import resolve_route

        route = None
        if tread := self.config.get("tread"):
            num_blocks = len(self.transformer.transformer_blocks)
            route = resolve_route(
                num_blocks, int(tread.get("start_block", 2)), int(tread.get("end_block", -3))
            )
            drop_ratio = float(tread["drop_ratio"])
            if not 0.0 < drop_ratio < 1.0:
                raise ConfigValidationError(
                    f"tread.drop_ratio must be in (0, 1), got {drop_ratio}."
                )
            disable_after_frac = float(tread.get("disable_after_frac", 1.0))
            if not 0.0 < disable_after_frac <= 1.0:
                raise ConfigValidationError(
                    f"tread.disable_after_frac must be in (0, 1], got {disable_after_frac}."
                )
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
                layers.append(RouteStartLayer(drop_ratio, disable_after_frac))
            layers.append(TransformerLayer(block, i, self.offloader))
            if route and i == route[1]:
                layers.append(RouteEndLayer())
        layers.append(FinalLayer(self.transformer))
        return layers

    def freeze_text_encoders(self):
        pass

    # ---- block swap ------------------------------------------------------------------------

    def get_block_swap_modules(self) -> list[nn.Module]:
        if self.transformer is None:
            return []
        return list(self.transformer.transformer_blocks)

    def _block_swap_root_modules(self) -> list:
        return [self.transformer]

    # ---- adapters (configure/save/load come from DiTPipeline) ------------------------------

    def load_and_fuse_adapter(self, path):
        raise NotImplementedError("load_and_fuse_adapter is not implemented for krea2")

    # ---- export --------------------------------------------------------------------------

    def save_model(self, save_dir, state_dict):
        """Write a diffusers-layout transformer folder (config.json + weights), loadable by
        ``Krea2Transformer2DModel.from_pretrained`` and by diffusers' ``Krea2Pipeline`` as the
        ``transformer`` component."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.transformer.save_config(save_dir)
        atomic_save_safetensors(save_dir / "diffusion_pytorch_model.safetensors", state_dict)

    # ---- previews --------------------------------------------------------------------------

    def _reload_vae_for_preview(self) -> None:
        self.vae = loading.load_vae(self._component_path("vae"), self.model_config["dtype"])

    def _reload_text_encoder_for_preview(self) -> nn.Module:
        # The lazy wrapper reloads itself on the next .to(<cuda>) / forward.
        return self.text_encoder

    def preview_prompt_embeds(self, prompts: list[str], preview_cfg: dict, device) -> list[torch.Tensor]:
        """``(L, layers, D)`` embeddings on CPU per prompt (valid tokens only). Preview prompts
        repeat every preview, so they are memoized: the text encoder is loaded only for
        prompts never seen."""
        missing = [p for p in dict.fromkeys(prompts) if p not in self._preview_embed_cache]
        if missing:
            self.ensure_text_encoder_for_preview(device)
            # No autocast: encode exactly like the training-caption cache does.
            with torch.autocast(torch.device(device).type, enabled=False):
                embeds, mask = encode_prompts(
                    self.text_encoder,
                    self.tokenizer,
                    missing,
                    select_layers=self.select_layers,
                    max_sequence_length=self.max_sequence_length,
                    device=device,
                )
            for i, p in enumerate(missing):
                self._preview_embed_cache[p] = embeds[i][mask[i]].cpu()
            self.offload_text_encoder_after_encode(preview_cfg)
        return [self._preview_embed_cache[p] for p in prompts]

    def prepare_preview_memory(self, preview_cfg: dict) -> None:
        self._prepare_blocks_preview_memory(preview_cfg)

    def restore_after_preview(self) -> None:
        state = getattr(self, "_preview_restore_state", None) or {}
        # The VAE is small: keep a reloaded one in host RAM between previews instead of
        # parking it on meta and re-reading it from disk every preview.
        state.pop("vae_was_meta", None)
        super().restore_after_preview()
        if next(self.vae.parameters()).device.type != "meta":
            self.vae.to("cpu")
        # Preview prompts are memoized, so an encoder that caching had freed is not needed
        # again: unload it instead of parking ~8 GB of weights in host RAM between previews.
        if getattr(self, "_preview_te_rest_device", None) is not None and self._preview_te_rest_device.type == "cpu":
            self.text_encoder.to("meta")
            self._preview_te_rest_device = None

    def generate_preview_image(self, preview_cfg: dict, prompt: str, step: int, seed: int):
        from rengu_flow.model.krea2.preview_sampling import generate_preview_image as _gen

        return _gen(self, preview_cfg, prompt, step, seed)
