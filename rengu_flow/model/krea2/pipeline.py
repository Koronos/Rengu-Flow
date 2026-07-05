"""Krea 2 training pipeline (Qwen3-VL text taps + Qwen-Image VAE + vendored Krea2 DiT).

Trains the open-weights Krea 2 checkpoints (``krea/Krea-2-Raw`` diffusers layout): full
finetune or any rengu adapter (LoRA / LoKr / LyCORIS catalog) on the DiT blocks. The VAE
and text encoder are always frozen; text embeddings must be cached (the tapped Qwen3-VL
stack is far too heavy to keep in the training graph).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.data.preprocess_media import PreprocessMediaFile
from rengu_flow.model.base import BasePipeline
from rengu_flow.model.krea2.layers import FinalLayer, InitialLayer, TransformerLayer
from rengu_flow.model.krea2.text import (
    DEFAULT_SELECT_LAYERS,
    compact_text_embeddings,
    encode_prompts,
    pad_text_embeddings,
)
from rengu_flow.model.krea2 import loading
from rengu_flow.networks import adapter_dit
from rengu_flow.registry.models import register_model
from rengu_flow.utils.common import is_main_process
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
# the 12-layer Krea2TextFusion refiner plus the txt_in projection (canonical
# checkpoint/diffusers name).
ADAPTER_LAYER_GROUPS = {
    "text_fusion": ("text_fusion.*", "txt_in.*"),
    "attention": ("transformer_blocks.*.attn.*",),
    "feedforward": ("transformer_blocks.*.ff.*",),
    "time_modulation": ("time_mod_proj",),
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
    m = (SHIFT_MAX - SHIFT_BASE) / (SHIFT_MAX_SEQ_LEN - SHIFT_BASE_SEQ_LEN)
    b = SHIFT_BASE - m * SHIFT_BASE_SEQ_LEN
    return image_seq_len * m + b


def time_shift(mu: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1))


@register_model("krea2")
class Krea2Pipeline(BasePipeline):
    name = "krea2"
    checkpointable_layers = ["TransformerLayer"]
    adapter_target_modules = list(ADAPTER_TARGET_MODULES)
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
        self.text_encoder = loading.load_text_encoder(self._component_path("text_encoder"), dtype)
        self.transformer = None

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
        fp8_matmul = bool(self.model_config.get("transformer_fp8_matmul", False))
        four_bit = bool(self.model_config.get("transformer_4bit", False))
        if not fp8_matmul and not four_bit:
            return

        from rengu_flow.training import quantize_dit

        scope = {"leaf_names": QUANT_LEAF_NAMES, "skip_substrings": QUANT_SKIP_SUBSTRINGS}
        if four_bit:
            n = quantize_dit.convert_dit_to_4bit(
                self.transformer, compute_dtype=torch.bfloat16, **scope
            )
            if is_main_process():
                print(f"rengu_flow: quantized {n} frozen Krea2 DiT linears to 4-bit NF4 (bnb).")
        else:
            # Tensorwise e4m3 (the sm89-viable scheme): 2x GEMM throughput under
            # block-scope compile AND 1 byte/param storage (no hi-precision copy).
            grad_mode = str(self.model_config.get("fp8_grad_mode", "bf16"))
            n = quantize_dit.convert_dit_to_fp8_tensorwise(
                self.transformer, grad_mode=grad_mode, **scope
            )
            if is_main_process():
                print(
                    f"rengu_flow: converted {n} frozen Krea2 DiT linears to fp8 tensorwise "
                    f"matmul (grad_mode={grad_mode})."
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
            latents = vae.encode(tensor.unsqueeze(2)).latent_dist.sample().squeeze(2)
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
            embeds, mask = compact_text_embeddings(embeds, mask)
            return {"prompt_embeds": embeds, "text_mask": mask}

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

        timestep_sample_method = self.model_config.get("timestep_sample_method", "logit_normal")
        if timestep_sample_method == "logit_normal":
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == "uniform":
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == "logit_normal":
            sigmoid_scale = self.model_config.get("sigmoid_scale", 1.0)
            t = torch.sigmoid(t * sigmoid_scale)

        # Krea 2 trains with a resolution-aware exponential time shift; a fixed model.shift
        # overrides the dynamic default.
        if shift := self.model_config.get("shift", None):
            t = (t * shift) / (1 + (shift - 1) * t)
        else:
            mu = calculate_shift((h // 2) * (w // 2))
            t = time_shift(mu, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents
        t = t.view(-1, 1)

        return (noisy_latents, t, prompt_embeds, text_mask), (target, mask)

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
        layers = [InitialLayer(self.transformer)]
        for i, block in enumerate(self.transformer.transformer_blocks):
            if route and i == route[0]:
                layers.append(RouteStartLayer(drop_ratio, disable_after_frac))
            layers.append(TransformerLayer(block, i, self.offloader))
            if route and i == route[1]:
                layers.append(RouteEndLayer())
        layers.append(FinalLayer(self.transformer))
        return layers

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast("cuda", enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                from rengu_flow.model.loss_utils import compute_diffusion_loss_per_element

                loss = compute_diffusion_loss_per_element(output, target, self.config)
                if mask is not None and mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss

        return loss_fn

    def freeze_text_encoders(self):
        pass

    # ---- block swap ------------------------------------------------------------------------

    def get_block_swap_modules(self) -> list[nn.Module]:
        if self.transformer is None:
            return []
        return list(self.transformer.transformer_blocks)

    def _block_swap_root_modules(self) -> list:
        return [self.transformer]

    # ---- adapters ----------------------------------------------------------------------------

    def configure_adapter(self, adapter_config):
        self.peft_config, self.adapter_type = adapter_dit.configure(
            self.transformer,
            adapter_config,
            targets=ADAPTER_TARGET_MODULES,
            layer_groups=ADAPTER_LAYER_GROUPS,
        )
        self.adapter_config = adapter_config
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config["dtype"])

    def save_adapter(self, save_dir, state_dict):
        adapter_dit.save(
            save_dir,
            state_dict,
            self.adapter_config,
            getattr(self, "peft_config", None),
            export_prefix=EXPORT_PREFIX,
        )

    def load_adapter_weights(self, adapter_path):
        adapter_type = getattr(self, "adapter_type", None) or (self.config.get("adapter") or {}).get("type")
        if adapter_type and adapter_type.startswith("lycoris_"):
            from rengu_flow.networks import lycoris_dit

            lycoris_dit.load(self.transformer, adapter_path)
        else:
            adapter_dit.load_weights(self.transformer, adapter_path)

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

    def ensure_vae_for_preview(self) -> None:
        """Reload the VAE when dataset caching parked it on ``meta``."""
        try:
            param = next(self.vae.parameters())
        except StopIteration:
            return
        if param.device.type != "meta":
            return
        if is_main_process():
            print("rengu_flow: loading VAE weights for preview...", flush=True)
        self.vae = loading.load_vae(self._component_path("vae"), self.model_config["dtype"])
        state = getattr(self, "_preview_restore_state", None)
        if state is None:
            self._preview_restore_state = {}
            state = self._preview_restore_state
        state["vae_was_meta"] = True

    def ensure_text_encoder_for_preview(self, device: str | torch.device = "cuda") -> None:
        """Make the text encoder available on *device*; reload from disk only the first time
        after caching freed it to ``meta``, then keep it parked on CPU between previews."""
        try:
            param = next(self.text_encoder.parameters())
        except StopIteration:
            return
        if param.device.type == "meta":
            if is_main_process():
                print("rengu_flow: loading text encoder weights for preview...", flush=True)
            self.text_encoder = loading.load_text_encoder(
                self._component_path("text_encoder"), self.model_config["dtype"]
            )
            self._preview_te_rest_device = torch.device("cpu")
        elif getattr(self, "_preview_te_rest_device", None) is None:
            self._preview_te_rest_device = param.device
        self.text_encoder.to(device)

    def offload_text_encoder_after_encode(self, preview_cfg: dict) -> None:
        if not preview_cfg.get("preview_offload_text_encoder", True):
            return
        try:
            param = next(self.text_encoder.parameters())
        except StopIteration:
            return
        if param.device.type == "meta":
            return
        self.text_encoder.to("cpu")

    def prepare_preview_memory(self, preview_cfg: dict) -> None:
        if self.transformer is None:
            self.load_diffusion_model()
        target = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        state: dict = {"transformer_was_training": self.transformer.training}
        # The training offloader's hooks fire on any forward (previews included) and its retained
        # residents/pending buffers hold several GB; park it so the preview offloader manages the
        # blocks alone and the preview text encoder has room. resume() in restore_after_preview.
        train_offloader = getattr(self, "_block_swap_offloader", None)
        if train_offloader is not None and getattr(train_offloader, "enabled", False):
            train_offloader.suspend()
        blocks_swap = int(preview_cfg.get("preview_blocks_to_swap", 0))
        if blocks_swap > 0:
            from rengu_flow.training.block_swap import BlockSwapOffloader

            # Blocks stream from CPU during the Euler loop; only the small shared modules move.
            for name, module in self.transformer.named_children():
                if name != "transformer_blocks":
                    module.to(target)
            # Mirror the training offloader's frozen/trainable split: if it keeps the (small) adapter
            # params GPU-resident (swap_trainable=False), the preview offloader must too, or resume()
            # finds them stranded on CPU. Default True when there is no training offloader.
            swap_trainable = getattr(train_offloader, "_swap_trainable", True)
            self._preview_offloader = BlockSwapOffloader(
                self.transformer.transformer_blocks, blocks_swap, device=target,
                swap_trainable=swap_trainable,
            )
        else:
            self._preview_offloader = None
            param = next(self.transformer.parameters())
            if param.device != target:
                if is_main_process():
                    print(f"rengu_flow: moving DiT to {target} for preview...", flush=True)
                self.transformer.to(target)
        self.transformer.eval()
        self._preview_restore_state = state

    def restore_after_preview(self) -> None:
        state = getattr(self, "_preview_restore_state", None) or {}
        offloader = getattr(self, "_preview_offloader", None)
        train_offloader = getattr(self, "_block_swap_offloader", None)
        if train_offloader is not None and getattr(train_offloader, "enabled", False):
            # Training streams the blocks itself: resume() re-parks them on the CPU masters and
            # re-arms the hooks. The preview offloader's teardown would instead pull ALL blocks
            # onto the GPU — an instant OOM with an unquantized 12B base.
            train_offloader.resume()
        elif offloader is not None:
            offloader.teardown()
        self._preview_offloader = None
        if state.get("vae_was_meta"):
            self.vae.to("meta")
        rest = getattr(self, "_preview_te_rest_device", None) or torch.device("cpu")
        try:
            next(self.text_encoder.parameters())
            if rest.type == "cuda":
                from rengu_flow.utils.common import empty_cuda_cache

                empty_cuda_cache()
            self.text_encoder.to(rest)
        except StopIteration:
            pass
        if state.get("transformer_was_training") and self.transformer is not None:
            self.transformer.train()
        self._preview_restore_state = None

    def generate_preview_image(self, preview_cfg: dict, prompt: str, step: int, seed: int):
        from rengu_flow.model.krea2.preview_sampling import generate_preview_image as _gen

        return _gen(self, preview_cfg, prompt, step, seed)
