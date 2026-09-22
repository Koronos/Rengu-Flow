"""Qwen-Image 2.1 training pipeline (Qwen3-VL-8B text encoder + RGBA 16x VAE + vendored
single-stream block-causal DiT).

Trains ``Qwen/Qwen-Image-2.1`` text-to-image: full finetune or any rengu adapter (LoRA / LoKr /
LyCORIS catalog) on the DiT. The VAE and text encoder are always frozen; text embeddings must be
cached (the 8B encoder cannot sit in the training graph). The encoder is loaded lazily (only
when captions still need encoding) and streams its decoder layers from pinned host RAM when it
does not fit in VRAM (``model.text_encoder_offload``). Image-conditioned (edit) training is not
supported: every sample is ``[prompt | target image]``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.data.preprocess_media import PreprocessMediaFile
from rengu_flow.model import dit_common
from rengu_flow.model.dit_common.streaming import OFFLOAD_MODES, LazyStreamedEncoder
from rengu_flow.model.qwen_image21 import loading
from rengu_flow.model.qwen_image21.layers import FinalLayer, InitialLayer, TransformerLayer
from rengu_flow.model.qwen_image21.text import drop_index, encode_prompts, text_model_of
from rengu_flow.registry.models import register_model
from rengu_flow.utils.save_io import atomic_save_safetensors

# Every Linear in the DiT (per-block attention/SwiGLU, the shared modulation, text/image
# projections, timestep MLP, output norm/projection) — the same all-linears scope krea2 uses.
# Narrow with adapter.layer_groups / target_include / target_exclude.
ADAPTER_TARGET_MODULES = ("QwenImage21Transformer2DModel",)

# "transformer." + diffusers module names: the key form ComfyUI's Qwen-Image LoRA loader maps
# for 2.1 (including the gate_layer/proj halves of its fused img_mlp.gate_up) and diffusers'
# QwenImageLoraLoaderMixin reads. A "diffusion_model." prefix would silently drop every MLP
# adapter in ComfyUI (its model only has the fused gate_up key).
EXPORT_PREFIX = "transformer."

# Named layer groups for adapter.layer_groups: globs over the DiT's dotted module paths.
ADAPTER_LAYER_GROUPS = {
    "attention": ("transformer_blocks.*.attn.*",),
    "feedforward": ("transformer_blocks.*.img_mlp.*",),
    # The timestep path: one modulation projection shared by every block, the timestep MLP and
    # the final adaptive norm's scale.
    "modulation": ("modulation.*", "time_text_embed.*", "norm_out.*"),
    "text_projection": ("txt_in.*",),
    "image_in_out": ("img_in", "proj_out"),
}

# Quantization scope for the frozen base: the per-block attention/SwiGLU linears only (~98% of
# the weights). The shared modulation, projections and timestep MLP stay in compute dtype.
QUANT_LEAF_NAMES = frozenset({"to_q", "to_k", "to_v", "0", "proj", "gate_layer", "out"})
QUANT_SKIP_SUBSTRINGS = (
    "time_text_embed",
    "txt_in",
    "img_in",
    "modulation",
    "norm_out",
    "proj_out",
)

# Resolution-aware timestep shift of the reference scheduler (scheduler_config.json:
# base_image_seq_len=256, max_image_seq_len=8192, base_shift=0.5, max_shift=0.9, exponential).
SHIFT_BASE_SEQ_LEN = 256
SHIFT_MAX_SEQ_LEN = 8192
SHIFT_BASE = 0.5
SHIFT_MAX = 0.9


def calculate_shift(image_seq_len: int) -> float:
    return dit_common.calculate_shift(
        image_seq_len, SHIFT_BASE_SEQ_LEN, SHIFT_MAX_SEQ_LEN, SHIFT_BASE, SHIFT_MAX
    )


@register_model("qwen_image21")
class QwenImage21Pipeline(dit_common.DiTPipeline):
    name = "qwen_image21"
    checkpointable_layers = ["TransformerLayer"]
    adapter_target_modules = list(ADAPTER_TARGET_MODULES)
    adapter_layer_groups = ADAPTER_LAYER_GROUPS
    adapter_export_prefix = EXPORT_PREFIX
    # VAE f16, unpatched latents, and 2x2 latent groups per vision slot -> even latent grid.
    pixels_round_to_multiple = 32
    vae_spatial_compression = 16

    def __init__(self, config):
        self.config = config
        self.model_config = config["model"]
        self._init_block_swap_state()
        dtype = self.model_config["dtype"]

        if not self.model_config.get("cache_text_embeddings", True):
            raise ConfigValidationError(
                "qwen_image21 requires cache_text_embeddings = true: the 8B Qwen3-VL text "
                "encoder cannot run inside the training graph."
            )
        self.cache_text_embeddings = True
        offload = str(self.model_config.get("text_encoder_offload", "auto"))
        if offload not in OFFLOAD_MODES:
            raise ConfigValidationError(
                f"model.text_encoder_offload must be one of {', '.join(OFFLOAD_MODES)}; got {offload!r}."
            )

        self.vae = loading.load_vae(self._component_path("vae"), dtype)
        self.tokenizer = loading.load_tokenizer(self._processor_path())
        self.drop_idx = drop_index(self.tokenizer)
        text_encoder_path = self._component_path("text_encoder")
        self.text_encoder = LazyStreamedEncoder(
            lambda: loading.load_text_encoder(text_encoder_path, dtype),
            layers_of=lambda module: module.layers,
            offload=offload,
            name="Qwen3-VL text encoder",
        )
        self.transformer = None
        self._preview_embed_cache: dict[str, torch.Tensor] = {}

    # ---- component paths ------------------------------------------------------------------

    def _component_path(self, component: str) -> str:
        """``model.<component>_path`` (a local file or folder) when set, else the
        ``<component>`` subfolder of ``model.diffusers_path`` (the Qwen/Qwen-Image-2.1
        download). Never a repo id, never downloaded."""
        override = self.model_config.get(f"{component}_path")
        if override:
            return str(override)
        root = self.model_config.get("diffusers_path")
        if not root:
            raise ConfigValidationError(
                f"model.{component}_path is required for qwen_image21 unless model.diffusers_path "
                "points at the Qwen-Image-2.1 diffusers folder (transformer/, vae/, text_encoder/, "
                "processor/)."
            )
        return str(Path(root) / component)

    def _processor_path(self) -> str | None:
        if override := self.model_config.get("processor_path"):
            return str(override)
        root = self.model_config.get("diffusers_path")
        if root and (Path(root) / "processor").is_dir():
            return str(Path(root) / "processor")
        return None  # bundled Qwen3-VL tokenizer

    # ---- DiT -------------------------------------------------------------------------------

    def load_diffusion_model(self, *, force: bool = False) -> None:
        if self.transformer is not None and not force:
            return
        dtype = self.model_config["dtype"]
        transformer_dtype = self.model_config.get("transformer_dtype", dtype)
        self.transformer = loading.load_transformer(
            self._component_path("transformer"), transformer_dtype
        )
        dit_common.quantize_frozen_dit(
            self.transformer,
            self.model_config,
            leaf_names=QUANT_LEAF_NAMES,
            skip_substrings=QUANT_SKIP_SUBSTRINGS,
            label="Qwen-Image 2.1",
        )
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if "adapter" not in self.config:
                p.requires_grad_(True)

    # ---- caching hooks ---------------------------------------------------------------------

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
            if tensor.shape[1] == 3:
                # The VAE reads RGBA; dataset images are RGB -> fully opaque alpha (1 in [-1, 1]).
                tensor = torch.cat([tensor, torch.ones_like(tensor[:, :1])], dim=1)
            # (B, 4, T=1, H, W) in -> (B, 64, 1, H/16, W/16) out.
            latents = vae.encode(tensor.unsqueeze(2)).latent_dist.sample().squeeze(2)
            mean, std = self._latent_stats(latents.device, latents.dtype)
            return {"latents": (latents - mean) / std}

        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            # Device of the embedding table (resident on the GPU when streaming; loads a lazy
            # encoder that was never placed).
            device = next(text_model_of(text_encoder).parameters()).device
            embeds, mask = encode_prompts(
                text_encoder, self.tokenizer, captions, device=device, drop_idx=self.drop_idx
            )
            return {"prompt_embeds": embeds, "text_mask": mask}

        return fn

    # ---- training --------------------------------------------------------------------------

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs["latents"].float()
        mask = inputs["mask"]
        prompt_embeds, text_mask = dit_common.pad_text_embeddings(inputs["prompt_embeds"], inputs["text_mask"])

        bs, _channels, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode="nearest-exact")

        t = dit_common.sample_timesteps(self.model_config, bs, latents.device, timestep_quantile)
        # Resolution-aware exponential shift of the reference scheduler (the image sequence is
        # the unpatched latent grid); a fixed model.shift overrides it.
        t = dit_common.shift_timesteps(t, self.model_config.get("shift", None), calculate_shift(h * w))
        noisy_latents, target, t = dit_common.add_flow_noise(latents, t)

        return (noisy_latents, t, prompt_embeds, text_mask), (target, mask)

    def to_layers(self):
        if self.config.get("tread"):
            raise ConfigValidationError("[tread] token routing is not supported for qwen_image21.")
        layers = [InitialLayer(self.transformer)]
        for i, block in enumerate(self.transformer.transformer_blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
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
        raise NotImplementedError("load_and_fuse_adapter is not implemented for qwen_image21")

    # ---- export ----------------------------------------------------------------------------

    def save_model(self, save_dir, state_dict):
        """Write a diffusers-layout transformer folder (config.json + weights), loadable by
        ``QwenImage21Transformer2DModel.from_pretrained`` and as the ``transformer`` component
        of diffusers' ``QwenImage21Pipeline``."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.transformer.save_config(save_dir)
        atomic_save_safetensors(save_dir / "diffusion_pytorch_model.safetensors", state_dict)

    # ---- previews --------------------------------------------------------------------------

    def _reload_vae_for_preview(self) -> None:
        self.vae = loading.load_vae(self._component_path("vae"), self.model_config["dtype"])

    def _reload_text_encoder_for_preview(self) -> nn.Module:
        # The lazy wrapper reloads itself on the next .to(<cuda>).
        return self.text_encoder

    def preview_prompt_embeds(self, prompts: list[str], preview_cfg: dict, device) -> list[tuple[torch.Tensor, int]]:
        """``(embeds (L, D) on CPU, L)`` per prompt. Preview prompts repeat every preview, so
        their embeddings are memoized: the 8B encoder is loaded only for prompts never seen."""
        missing = [p for p in dict.fromkeys(prompts) if p not in self._preview_embed_cache]
        if missing:
            self.ensure_text_encoder_for_preview(device)
            # No autocast: encode exactly like the training-caption cache does.
            embeds, mask = encode_prompts(
                self.text_encoder, self.tokenizer, missing, device=device, drop_idx=self.drop_idx
            )
            for i, p in enumerate(missing):
                self._preview_embed_cache[p] = embeds[i][mask[i]].cpu()
            self.offload_text_encoder_after_encode(preview_cfg)
        return [(self._preview_embed_cache[p], self._preview_embed_cache[p].shape[0]) for p in prompts]

    def prepare_preview_memory(self, preview_cfg: dict) -> None:
        self._prepare_blocks_preview_memory(preview_cfg)

    def restore_after_preview(self) -> None:
        super().restore_after_preview()
        # Preview prompts are memoized, so an encoder that caching had freed is not needed
        # again: unload it instead of parking ~16 GB of weights in host RAM between previews.
        if getattr(self, "_preview_te_rest_device", None) is not None and self._preview_te_rest_device.type == "cpu":
            self.text_encoder.to("meta")
            self._preview_te_rest_device = None

    def generate_preview_image(self, preview_cfg: dict, prompt: str, step: int, seed: int):
        from rengu_flow.model.qwen_image21.preview_sampling import generate_preview_image as _gen

        return _gen(self, preview_cfg, prompt, step, seed)
