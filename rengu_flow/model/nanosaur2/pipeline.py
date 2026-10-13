"""Nanosaur2 training pipeline (Gemma3-270M text + semantic DINOv2 VAE + 670M SPRINT DiT).

Trains the ``well9472/Nanosaur2-670M`` release files: full finetune or any rengu adapter (LoRA /
LoKr / LyCORIS catalog). The VAE and the text encoder are frozen and run only while caching
(text embeddings + per-token emphasis weights go through the standard cache).

Default recipe = the reference ``train_lora.py``:

* t = sigmoid(N(ln 2, 1)), no shift; ``z_t = (1 - t) x + t eps``; the model returns the velocity
  ``(z_t - x0_pred) / t`` and the loss is ``mean(((x0_pred - x) / (t + 0.05))^2)``. Here it is
  computed as the velocity MSE weighted by ``(t / (t + 0.05))^2`` — the same quantity, since
  ``x0_pred - x = t (v_target - v)`` — so ``DiTPipeline``'s loss (and the user's loss_type
  choice) applies unchanged; the weight rides in the loss mask.
* caption dropout through the dataset's ``uncond_fraction`` (default 0.1 for this model's
  training dataset, never for eval datasets; the dropped caption is the encoded empty prompt,
  BOS only);
* SPRINT path-drop: each sample skips the sparse middle blocks with probability
  ``model.sparse_skip_prob`` (0.05), or ``model.sparse_skip_prob_uncond`` (0.5) when its caption
  is empty, so the model keeps learning the path-drop unconditional pass its sampler uses.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model import dit_common
from rengu_flow.model.dit_common.streaming import LazyStreamedEncoder
from rengu_flow.model.nanosaur2 import loading
from rengu_flow.model.nanosaur2.dit import SPRINT_NUM_F, SPRINT_NUM_H
from rengu_flow.model.nanosaur2.layers import (
    FinalLayer,
    InitialLayer,
    SparseEndLayer,
    SparseStartLayer,
    TransformerLayer,
)
from rengu_flow.model.nanosaur2.text import encode_prompts
from rengu_flow.registry.models import register_model
from rengu_flow.utils.save_io import atomic_save_safetensors

# The reference LoRA scope (train_lora.py TARGETS = qkv_x, kv_y, proj, qkv, w12, w3 under
# blocks./text_refine_blocks.): every Linear of the image/text attention and feed-forward
# modules. The text-refine adaLN projections and the shared embedders stay out.
ADAPTER_TARGET_MODULES = ("Attention", "TextRefineAttention", "FeedForward")

# Named layer groups for adapter.layer_groups (globs over the DiT's dotted module paths). The
# union of attention + feedforward + text_refine is the default scope.
ADAPTER_LAYER_GROUPS = {
    "attention": ("blocks.*.attn.*",),
    "cross_attention": ("blocks.*.attn.kv_y",),
    "feedforward": ("blocks.*.mlp.*",),
    "text_refine": ("text_refine_blocks.*",),
}

# Reference recipe defaults (train_lora.py / README).
DEFAULT_SPARSE_SKIP_PROB = 0.05
DEFAULT_SPARSE_SKIP_PROB_UNCOND = 0.5
DEFAULT_UNCOND_FRACTION = 0.1
LOGIT_MEAN = math.log(2.0)
LOSS_T_OFFSET = 0.05

# Full-finetune export: the release file name, so the folder drops into ComfyUI as is.
EXPORT_FILE_NAME = "nanosaur2_diffusion_model.safetensors"


def _ragged(rows) -> tuple[list[torch.Tensor], list[int]]:
    if torch.is_tensor(rows):
        rows = list(rows)
    return rows, [int(r.shape[0]) for r in rows]


def pad_text(prompt_embeds, token_weights) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad cached per-caption rows into ``(B, L, D)`` embeddings and ``(B, L)`` weights
    (0 = padding). Also returns each row's token count ``(B,)``."""
    embeds, lengths = _ragged(prompt_embeds)
    weights, _ = _ragged(token_weights)
    masks = [torch.ones(n, dtype=torch.bool) for n in lengths]
    embeds, _ = dit_common.pad_text_embeddings(embeds, masks)
    weights, _ = dit_common.pad_text_embeddings([w.float() for w in weights], masks)
    return embeds, weights, torch.tensor(lengths)


@register_model("nanosaur2")
class Nanosaur2Pipeline(dit_common.DiTPipeline):
    name = "nanosaur2"
    checkpointable_layers = ["TransformerLayer"]
    adapter_target_modules = list(ADAPTER_TARGET_MODULES)
    adapter_layer_groups = ADAPTER_LAYER_GROUPS
    adapter_export_lora_alpha = True  # ComfyUI keys: diffusion_model.<path>.lora_A/.lora_B/.alpha
    pixels_round_to_multiple = 16  # VAE downsample 16, patch 1
    vae_spatial_compression = 16
    preview_keep_reloaded_vae = True
    # The reference recipe's 10% caption dropout (the encoded empty prompt), applied to the
    # training dataset only by data.dataset.apply_model_training_dataset_defaults.
    default_uncond_fraction = DEFAULT_UNCOND_FRACTION
    preview_unload_freed_text_encoder = True
    blocks_attr = "blocks"
    # The spec's text_cache = "required" refuses cache_text_embeddings = false with this reason.
    text_cache_required_reason = (
        " (captions are encoded once while caching; the frozen Gemma3 encoder never runs in the "
        "training graph)."
    )
    # GAS splits of mixed caption lengths keep the unmasked (flash) attention path.
    trim_micro_batch = dit_common.text_padding_trimmer()

    def __init__(self, config):
        self._init_pipeline(config)
        self.sparse_skip_prob = float(self.model_config.get("sparse_skip_prob", DEFAULT_SPARSE_SKIP_PROB))
        self.sparse_skip_prob_uncond = float(
            self.model_config.get("sparse_skip_prob_uncond", DEFAULT_SPARSE_SKIP_PROB_UNCOND)
        )
        for key, value in (("sparse_skip_prob", self.sparse_skip_prob), ("sparse_skip_prob_uncond", self.sparse_skip_prob_uncond)):
            if not 0.0 <= value <= 1.0:
                raise ConfigValidationError(f"model.{key} must be in [0, 1], got {value}.")
        self.vae = self._load_vae()
        text_encoder_path = self._component_path("text_encoder")
        # Lazy: a run with a warm text cache never reads the encoder. Gemma3-270M is tiny, so it
        # runs resident and in float32 (ComfyUI also computes it in float32 over bf16 weights).
        self.text_encoder = LazyStreamedEncoder(
            lambda: loading.load_text_encoder(text_encoder_path, torch.float32),
            layers_of=lambda module: module.layers,
            offload="none",
            name="Gemma3-270M text encoder",
        )
        self._tokenizer = None

    def _component_path(self, component: str) -> str:
        # Overrides the template for its message: the release file names the DiT "diffusion_model".
        if not self.model_config.get(f"{component}_path"):
            raise ConfigValidationError(
                f"model.{component}_path is required for nanosaur2: point it at the "
                f"nanosaur2_{'diffusion_model' if component == 'transformer' else component}.safetensors "
                "file from well9472/Nanosaur2-670M."
            )
        return super()._component_path(component)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = loading.load_tokenizer(self._component_path("text_encoder"))
        return self._tokenizer

    # ---- DiTPipeline hooks ---------------------------------------------------------------

    def _load_transformer(self, dtype):
        return loading.load_transformer(self._component_path("transformer"), dtype)

    def _load_vae(self):
        return loading.load_vae(self._component_path("vae"), self.model_config["dtype"])

    # ---- caching hooks -------------------------------------------------------------------

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            return {"latents": vae.encode(tensor.to(p.device, p.dtype))}

        return fn

    def _encode(self, text_encoder, captions, device):
        module = text_encoder.load() if isinstance(text_encoder, LazyStreamedEncoder) else text_encoder
        # No autocast: the cache and the preview encode identically, in the encoder's dtype.
        with torch.autocast(torch.device(device).type, enabled=False):
            return encode_prompts(module, self.tokenizer, captions, device)

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            device = next(text_encoder.parameters()).device
            if device.type == "meta":  # lazy wrapper not loaded yet (CPU caching)
                device = torch.device("cpu")
            embeds, weights = self._encode(text_encoder, list(captions), device)
            # One row per caption at its own token length (BOS only for an empty caption).
            return {"prompt_embeds": embeds, "token_weights": weights}

        return fn

    # ---- training ------------------------------------------------------------------------

    def sample_timesteps(self, batch_size: int, device, timestep_quantile=None) -> torch.Tensor:
        """Reference logit-normal centered at ln 2 (``model.timestep_sample_method = "uniform"``
        and ``model.sigmoid_scale`` / ``model.shift`` are honored when set)."""
        t = dit_common.sample_timesteps(
            self.model_config, batch_size, device, timestep_quantile, logit_mean=LOGIT_MEAN
        )
        return dit_common.shift_timesteps(t, self.model_config.get("shift", None))

    def sparse_skip_flags(self, lengths: torch.Tensor, timestep_quantile=None) -> torch.Tensor:
        """Per-row path-drop flags: a caption-dropped row (BOS-only text) skips with
        ``sparse_skip_prob_uncond``, the others with ``sparse_skip_prob``. Deterministic probes
        (``timestep_quantile`` set) never skip."""
        if timestep_quantile is not None:
            return torch.zeros(lengths.shape[0], dtype=torch.bool)
        prob = torch.where(lengths <= 1, self.sparse_skip_prob_uncond, self.sparse_skip_prob)
        return torch.rand(lengths.shape[0]) < prob

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs["latents"].float()
        mask = inputs["mask"]
        embeds, weights, lengths = pad_text(inputs["prompt_embeds"], inputs["token_weights"])
        bs, _channels, h, w = latents.shape

        t = self.sample_timesteps(bs, latents.device, timestep_quantile)
        noisy_latents, target, t = dit_common.add_flow_noise(latents, t)
        sparse_skip = self.sparse_skip_flags(lengths, timestep_quantile)

        # The reference x0-space loss ((x0_pred - x) / (t + 0.05))^2 is this weight times the
        # velocity MSE; folded into the mask so the shared loss_fn stays unchanged.
        weight = (t / (t + LOSS_T_OFFSET)).pow(2).view(-1, 1, 1, 1).float()
        if mask is not None:
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode="nearest-exact")
            weight = weight * mask
        return (noisy_latents, t, embeds, weights, sparse_skip), (target, weight)

    def to_layers(self):
        transformer = self.transformer
        pipe_parallel = int(self.config.get("pipeline_stages", 1)) > 1
        n = len(transformer.blocks)
        layers = [InitialLayer(transformer, pipe_parallel=pipe_parallel)]
        for i in range(n):
            if i == SPRINT_NUM_F:
                layers.append(SparseStartLayer(subset=not pipe_parallel))
            layers.append(TransformerLayer(transformer, i))
            if i == n - SPRINT_NUM_H - 1:
                layers.append(SparseEndLayer(transformer, subset=not pipe_parallel))
        layers.append(FinalLayer(transformer))
        return layers

    # ---- export --------------------------------------------------------------------------

    def save_model(self, save_dir, state_dict):
        """One safetensors in the release layout (official key names), loadable wherever
        ``nanosaur2_diffusion_model.safetensors`` is (ComfyUI's Nanosaur2 loader)."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_safetensors(save_dir / EXPORT_FILE_NAME, state_dict)

    # ---- previews (VAE/TE lifecycle, memory: DiTPipeline; the DiT samples resident) -------------

    def preview_prompt_embeds(self, prompts: list[str], preview_cfg: dict, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``(embeds (L, 640), token_weights (L,))`` on CPU per prompt, memoized."""

        def encode(missing):
            embeds, weights = self._encode(self.text_encoder, missing, device)
            return list(zip(embeds, weights))

        return self._memoized_preview_embeds(prompts, preview_cfg, device, encode)

    def generate_preview_image(self, preview_cfg: dict, prompt: str, step: int, seed: int):
        from rengu_flow.model.nanosaur2.preview_sampling import generate_preview_image as _gen

        return _gen(self, preview_cfg, prompt, step, seed)
