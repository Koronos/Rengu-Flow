"""MOSS-Image training pipeline (Qwen3.5-9B text encoder + FLUX.2 VAE + 5B single-stream DiT).

Trains ``OpenMOSS-Team/MOSS-Image-Base`` (``-SFT`` loads the same way) text-to-image: full finetune
or any rengu adapter (LoRA / LoKr / LyCORIS catalog) on the DiT. The VAE and the text encoder are
always frozen; text embeddings are cached (the 9B encoder never runs in the training graph) and the
encoder streams its decoder layers from host RAM when it does not fit in VRAM
(``model.text_encoder_offload``).

Recipe (MOSS-Image technical report, Table 6 final stage + Eq. 3/4; sampler in ``timesteps.py``):

* sigma = 1 is noise: ``x = (1 - sigma) x0 + sigma eps``, target ``eps - x0``, model time input
  ``1000 sigma``; sigma on the 1000-step grid from the report's truncated logit-normal + 1% uniform,
  no shift (``model.shift`` still applies when set);
* plain velocity MSE over the valid image tokens (no sigma / SNR weighting); text tokens carry no loss;
* latents: FLUX.2 VAE posterior mode -> 2x2 patchify (128 channels at /16) -> ``(z - bn.mean) /
  sqrt(bn.var + eps)`` with the VAE's own ``bn`` statistics (diffusers ``pipeline_flux2`` convention);
* captions: the Qwen3.5 chat template (thinking on, no system prompt), every prompt encoded alone.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model import dit_common
from rengu_flow.model.dit_common.streaming import OFFLOAD_MODES, LazyStreamedEncoder
from rengu_flow.model.dit_common.text import ragged_rows
from rengu_flow.model.moss_image import loading, timesteps
from rengu_flow.model.moss_image.layers import FinalLayer, InitialLayer, TransformerLayer
from rengu_flow.model.moss_image.text import encode_prompts, text_model_of
from rengu_flow.registry.models import register_model
from rengu_flow.utils.save_io import atomic_save_safetensors

# Every Linear of the DiT is reachable (the named layer groups below address any of them); the
# default scope is the blocks' attention + feed-forward (``spec.adapter_default_layer_groups``).
ADAPTER_TARGET_MODULES = ("MOSSImageTransformer",)

# Named layer groups for adapter.layer_groups: globs over the DiT's dotted module paths.
ADAPTER_LAYER_GROUPS = {
    "attention": ("layers.*.proj.[qkvo]",),
    "feedforward": ("layers.*.proj.ffn_*",),
    "text_refiner": ("text_embedder.blocks.*",),
    "text_projection": ("text_embedder.proj",),
    "modulation": ("shared_block_modulation", "video_head_modulation", "time_embedding.*"),
    "image_in_out": ("video_embedder", "final_linear_video"),
}

# Quantization scope for the frozen base: the 42 blocks' big linears only (~98% of the weights).
QUANT_LEAF_NAMES = frozenset({"q", "k", "v", "o", "ffn_up_gate", "ffn_down"})
QUANT_SKIP_SUBSTRINGS = (
    "text_embedder",
    "time_embedding",
    "shared_block_modulation",
    "video_head_modulation",
    "video_embedder",
    "final_linear_video",
)

def patchify(latents: torch.Tensor) -> torch.Tensor:
    """``(B, 32, 2h, 2w)`` VAE latents -> ``(B, 128, h, w)``, channel ``c * 4 + p1 * 2 + p2``
    (diffusers ``Flux2Pipeline._patchify_latents``)."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify(latents: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`patchify` (the reference decode path)."""
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def bn_stats(vae, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mean, std)`` ``(1, 128, 1, 1)`` of the VAE's ``bn``: ``std = sqrt(running_var + eps)``."""
    mean = vae.bn.running_mean.to(device, dtype).view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.to(device, dtype) + vae.config.batch_norm_eps).view(1, -1, 1, 1)
    return mean, std


@register_model("moss_image")
class MOSSImagePipeline(dit_common.DiTPipeline):
    name = "moss_image"
    checkpointable_layers = ["TransformerLayer"]
    adapter_target_modules = list(ADAPTER_TARGET_MODULES)
    adapter_layer_groups = ADAPTER_LAYER_GROUPS
    adapter_export_lora_alpha = True  # ComfyUI: diffusion_model.<path>.lora_A/.lora_B/.alpha
    # VAE f8 x 2x2 patchify = 16.
    pixels_round_to_multiple = 16
    vae_spatial_compression = 16
    blocks_attr = "layers"
    # The small VAE stays in host RAM between previews; the ~17 GB encoder caching freed is unloaded.
    preview_keep_reloaded_vae = True
    preview_unload_freed_text_encoder = True

    quant_leaf_names = QUANT_LEAF_NAMES
    quant_skip_substrings = QUANT_SKIP_SUBSTRINGS
    quant_label = "MOSS-Image"
    text_cache_required_reason = ": the 9B Qwen3.5 text encoder cannot run inside the training graph."
    missing_component_message = (
        "model.{component}_path is required for {name} unless model.{bundle} points at the "
        "MOSS-Image diffusers folder (transformer/, vae/, text_encoder/, tokenizer/)."
    )
    # GAS splits of mixed caption lengths drop the text lanes no row of a micro-batch uses.
    trim_micro_batch = dit_common.text_padding_trimmer()

    def __init__(self, config):
        self._init_pipeline(config)
        dtype = self.model_config["dtype"]
        offload = str(self.model_config.get("text_encoder_offload", "auto"))
        if offload not in OFFLOAD_MODES:
            raise ConfigValidationError(
                f"model.text_encoder_offload must be one of {', '.join(OFFLOAD_MODES)}; got {offload!r}."
            )
        self.vae = self._load_vae()
        self.tokenizer = loading.load_tokenizer(self._tokenizer_path())
        text_encoder_path = self._component_path("text_encoder")
        # Lazy: a run with a warm text cache never reads the encoder.
        self.text_encoder = LazyStreamedEncoder(
            lambda: loading.load_text_encoder(text_encoder_path, dtype),
            layers_of=lambda module: module.layers,
            offload=offload,
            name="Qwen3.5 text encoder",
        )

    def _tokenizer_path(self) -> str | None:
        if override := self.model_config.get("tokenizer_path"):
            return str(override)
        root = self.model_config.get("diffusers_path")
        if root and (Path(root) / "tokenizer").is_dir():
            return str(Path(root) / "tokenizer")
        return None  # bundled Qwen3.5 tokenizer

    # ---- DiTPipeline hooks -----------------------------------------------------------------

    def _load_transformer(self, dtype):
        return loading.load_transformer(self._component_path("transformer"), dtype)

    def _load_vae(self):
        # fp32 as the reference (force_upcast): the bn statistics and the encoder stay exact.
        return loading.load_vae(self._component_path("vae"), torch.float32)

    def _encode_text(self, text_encoder, prompts, device):
        return encode_prompts(text_encoder, self.tokenizer, prompts, device=device)

    # ---- caching hooks ---------------------------------------------------------------------

    def _latent_stats(self, device, dtype):
        return bn_stats(self.vae, device, dtype)

    def encode_latents(self, vae, tensor: torch.Tensor) -> torch.Tensor:
        """Normalized ``(B, 128, h, w)`` latents of pixels ``(B, 3, H, W)`` in ``[-1, 1]``: posterior
        mode, 2x2 patchify, ``(z - bn.mean) / sqrt(bn.var + eps)``."""
        p = next(vae.parameters())
        z = patchify(vae.encode(tensor.to(p.device, p.dtype)).latent_dist.mode())
        mean, std = bn_stats(vae, z.device, z.dtype)
        return (z - mean) / std

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            # Device of the embedding table (loads a lazy encoder that was never placed).
            device = next(text_model_of(text_encoder).parameters()).device
            return ragged_rows(*self._encode_text(text_encoder, list(captions), device))

        return fn

    # ---- training --------------------------------------------------------------------------

    def sample_timesteps(self, batch_size: int, device, timestep_quantile=None) -> torch.Tensor:
        """Sigmas in (0, 1]: the report's sampler (``timestep_sample_method = "moss"``, default) or the
        generic ``logit_normal`` / ``uniform``; a fixed ``model.shift`` applies on top."""
        cfg = self.model_config
        if cfg.get("timestep_sample_method", "moss") == "moss":
            t = timesteps.sample_sigmas(
                batch_size,
                device,
                timestep_quantile,
                mu=float(cfg.get("timestep_mu", timesteps.DEFAULT_MU)),
                std=float(cfg.get("timestep_std", timesteps.DEFAULT_STD)),
                cutoff=float(cfg.get("timestep_cutoff", timesteps.DEFAULT_CUTOFF)),
                uniform_mix=float(cfg.get("timestep_uniform_mix", timesteps.DEFAULT_UNIFORM_MIX)),
            )
        else:
            t = dit_common.sample_timesteps(cfg, batch_size, device, timestep_quantile)
        return dit_common.shift_timesteps(t, cfg.get("shift", None))

    def prepare_inputs(self, inputs, timestep_quantile=None):
        """``(noisy, sigma, prompt_embeds, text_mask)`` and ``(target, mask)``: plain velocity target
        ``eps - x0``, loss mask = the pixel mask on the latent grid (no sigma weighting)."""
        latents = inputs["latents"].float()
        prompt_embeds, text_mask = dit_common.pad_text_embeddings(inputs["prompt_embeds"], inputs["text_mask"])
        mask = inputs["mask"]
        if mask is not None:
            mask = F.interpolate(mask.unsqueeze(1), size=latents.shape[-2:], mode="nearest-exact")
        t = self.sample_timesteps(latents.shape[0], latents.device, timestep_quantile)
        noisy_latents, target, t = dit_common.add_flow_noise(latents, t)
        return (noisy_latents, t, prompt_embeds, text_mask), (target, mask)

    def to_layers(self):
        if self.config.get("tread"):
            raise ConfigValidationError("[tread] token routing is not supported for moss_image.")
        layers = [InitialLayer(self.transformer, pipe_parallel=int(self.config.get("pipeline_stages", 1)) > 1)]
        for i, block in enumerate(self.transformer.layers):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(self.transformer))
        return layers

    # ---- export ----------------------------------------------------------------------------

    def save_model(self, save_dir, state_dict):
        """The official diffusers transformer layout (``config.json`` + weights with the official key
        names, ``rope.bands`` included): reloads in the official pipeline and in this loader."""
        state = dict(state_dict)
        state.setdefault("rope.bands", self.transformer.rope.bands.detach().cpu().float().clone())
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.transformer.save_config(save_dir)
        atomic_save_safetensors(save_dir / "diffusion_pytorch_model.safetensors", state)

    # ---- previews (VAE/TE lifecycle, memory, block swap: DiTPipeline) ------------------------

    def generate_preview_image(self, preview_cfg: dict, prompt: str, step: int, seed: int):
        from rengu_flow.model.moss_image.preview_sampling import generate_preview_image as _gen

        return _gen(self, preview_cfg, prompt, step, seed)
