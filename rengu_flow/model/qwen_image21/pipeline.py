"""Qwen-Image 2.1 training pipeline (Qwen3-VL-8B text encoder + RGBA 16x VAE + vendored
single-stream block-causal DiT).

Trains ``Qwen/Qwen-Image-2.1`` text-to-image and image-conditioned (edit): full finetune or any
rengu adapter (LoRA / LoKr / LyCORIS catalog) on the DiT. The VAE and text encoder are always
frozen; text embeddings must be cached (the 8B encoder cannot sit in the training graph). The
encoder is loaded lazily (only when captions still need encoding) and streams its decoder
layers from pinned host RAM when it does not fit in VRAM (``model.text_encoder_offload``).

A text-to-image sample is ``[prompt | target image]``. An edit sample (a dataset directory with
``control_path``) is ``[prompt with its N condition images | target]``: the condition images are
read by the Qwen3-VL vision tower (loaded on the first caption that has them) as part of the
prompt, and their clean VAE latents fill the vision slots of the DiT sequence; noise and loss
touch only the target. Both kinds of batch can share one run.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.data.control import summarize_control_problems
from rengu_flow.data.preprocess_media import PreprocessMediaFile
from rengu_flow.model import dit_common
from rengu_flow.model.dit_common.streaming import OFFLOAD_MODES, LazyStreamedEncoderWithCompanion
from rengu_flow.model.qwen_image21 import loading
from rengu_flow.model.qwen_image21.dit import pack_latents
from rengu_flow.model.qwen_image21.layers import MAX_CONDITION_IMAGES, FinalLayer, InitialLayer, TransformerLayer
from rengu_flow.model.qwen_image21.text import (
    drop_index,
    encode_prompts,
    encode_prompts_with_images,
    text_model_of,
    vision_language_model,
)
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

# Smallest condition image the Qwen3-VL processor keeps at its own size: below this area it
# upsamples, and the image no longer maps 1:1 onto the vision slots its VAE latents fill.
MIN_CONTROL_AREA = 256 * 256


def control_row_problems(rows) -> list[str]:
    """One message per target whose control images qwen_image21 cannot train on (first rule it
    breaks): more than ``MAX_CONDITION_IMAGES`` of them, or one below ``MIN_CONTROL_AREA``.
    ``rows`` are :class:`rengu_flow.data.control.ControlRow`."""
    problems: dict[str, str] = {}
    for row in rows:
        if row.target in problems:
            continue
        if row.count > MAX_CONDITION_IMAGES:
            problems[row.target] = (
                f"{row.target}: {row.count} control images, at most {MAX_CONDITION_IMAGES} are supported"
            )
            continue
        small = [(w, h) for w, h in row.sizes if w * h < MIN_CONTROL_AREA]
        if small:
            w, h = small[0]
            problems[row.target] = (
                f"{row.target}: control image resized to {w}x{h}, below the 256x256 minimum area"
            )
    return list(problems.values())


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


def _stack_rows(value) -> torch.Tensor:
    return value if torch.is_tensor(value) else torch.stack(list(value))


def _edit_inputs(inputs: dict, text_len: int, h: int, w: int):
    """``(img_mask, control_latents, control_layout)`` of an edit batch, or ``None`` for t2i.

    - ``img_mask`` ``(B, text_len + h*w/4)`` bool: the cached ``image_pad_mask`` (the condition
      images' vision slots in the prompt sequence, right-padded like the embeddings) followed by
      one slot per 2x2 group of target latents — the reference's ``append_target_slots``.
    - ``control_latents`` ``(B, sum h_i*w_i, 64)``: the N condition latents packed and joined in
      order (they precede the target in the DiT sequence).
    - ``control_layout``: a zero-storage tensor of shape ``(B, h_0, w_0, ..., h_{N-1}, w_{N-1}, 0)``
      carrying the per-image latent grids as host shape metadata (the pipe tuple is tensors-only).
      Batch-major like every other feature: the loader's ``split_batch`` slices dim 0 into
      micro-batches, which must leave the grids intact.
    """
    keys = sorted(
        (k for k in inputs if k.startswith("control_latents_")), key=lambda k: int(k.rsplit("_", 1)[1])
    )
    if not keys:
        return None
    if [int(k.rsplit("_", 1)[1]) for k in keys] != list(range(len(keys))):
        raise ValueError(f"qwen_image21: condition latents must be numbered 0..N-1, got {keys}")
    controls = [_stack_rows(inputs[k]).float() for k in keys]
    bs = controls[0].shape[0]
    if inputs.get("image_pad_mask") is None:
        raise ValueError(
            "qwen_image21: an edit batch (condition latents present) has no cached image_pad_mask — "
            "its text embeddings were encoded without the condition images. Regenerate the text "
            "cache (--regenerate_text_cache)."
        )
    raw = inputs["image_pad_mask"]
    rows = list(raw.unbind(0)) if torch.is_tensor(raw) else list(raw)
    image_pad_mask = torch.zeros((bs, text_len), dtype=torch.bool)
    for i, row in enumerate(rows):
        image_pad_mask[i, : row.shape[0]] = row.bool().cpu()
    grids = [tuple(c.shape[-2:]) for c in controls]
    slots = sum(gh * gw for gh, gw in grids) // 4
    if any(gh % 2 or gw % 2 for gh, gw in grids) or not bool((image_pad_mask.sum(1) == slots).all()):
        raise ValueError(
            f"qwen_image21: the cached image_pad_mask marks {image_pad_mask.sum(1).tolist()} vision "
            f"slots but the condition latents {grids} need {slots} per sample (stale text cache? "
            "regenerate it)."
        )
    if not bool((image_pad_mask == image_pad_mask[:1]).all()):
        raise ValueError("qwen_image21: the samples of an edit batch must share one condition-image layout.")
    img_mask = torch.cat([image_pad_mask, torch.ones((bs, h * w // 4), dtype=torch.bool)], dim=1)
    control_latents = torch.cat([pack_latents(c) for c in controls], dim=1)
    control_layout = control_latents.new_empty((bs, *[d for g in grids for d in g], 0))
    return img_mask, control_latents, control_layout


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
        # The text decoder streams; the vision tower (~1.2 GB) is only read for captions that come
        # with condition images, then stays resident next to it.
        self.text_encoder = LazyStreamedEncoderWithCompanion(
            lambda: loading.load_text_encoder(text_encoder_path, dtype),
            layers_of=lambda module: module.layers,
            companion_loader=lambda: loading.load_vision_encoder(text_encoder_path, dtype),
            offload=offload,
            name="Qwen3-VL text encoder",
            companion_name="Qwen3-VL vision tower",
        )
        self.transformer = None
        self._preview_embed_cache: dict = {}
        self._processor = None
        self._vlm_shell = None

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

    def encode_condition_latents(self, vae, tensor: torch.Tensor) -> torch.Tensor:
        """Normalized VAE latents ``(B, 64, 1, H/16, W/16)`` of a condition-image batch
        ``(B, C, H, W)`` or ``(B, C, 1, H, W)`` in ``[-1, 1]`` (RGB gets an opaque alpha): the
        distribution's mode, as the reference encodes condition images (``sample_mode="argmax"``)."""
        p = next(vae.parameters())
        tensor = tensor.to(p.device, p.dtype)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)
        if tensor.shape[1] == 3:
            tensor = torch.cat([tensor, torch.ones_like(tensor[:, :1])], dim=1)
        latents = vae.encode(tensor).latent_dist.mode()
        mean, std = self._latent_stats(latents.device, latents.dtype)
        return (latents - mean.unsqueeze(2)) / std.unsqueeze(2)

    def get_call_vae_fn(self, vae):
        def fn(tensor, control_tensors=None):
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            if tensor.shape[1] == 3:
                # The VAE reads RGBA; dataset images are RGB -> fully opaque alpha (1 in [-1, 1]).
                tensor = torch.cat([tensor, torch.ones_like(tensor[:, :1])], dim=1)
            # (B, 4, T=1, H, W) in -> (B, 64, 1, H/16, W/16) out.
            latents = vae.encode(tensor.unsqueeze(2)).latent_dist.sample().squeeze(2)
            mean, std = self._latent_stats(latents.device, latents.dtype)
            out = {"latents": (latents - mean) / std}
            if control_tensors is not None:
                if torch.is_tensor(control_tensors):
                    control_tensors = [control_tensors]
                for i, control in enumerate(control_tensors):
                    out[f"control_latents_{i}"] = self.encode_condition_latents(vae, control)
            return out

        return fn

    # ---- image-conditioned text encoding ---------------------------------------------------

    def _get_processor(self):
        if self._processor is None:
            self._processor = loading.load_processor(self._processor_path(), self.tokenizer)
        return self._processor

    def _vision_language_model(self, text_encoder):
        """The full Qwen3-VL (vision tower + deepstack + the streamed text decoder) for prompts
        with condition images. The vision tower loads on first use and stays resident with the
        encoder. The returned shell must be released with ``_release_vision_language_model``."""
        text_model = text_model_of(text_encoder)
        vision = text_encoder.load_companion()
        if self._vlm_shell is None:
            config = loading.load_qwen3vl_config(self._component_path("text_encoder"))
        else:
            config = self._vlm_shell.config
        self._vlm_shell = vision_language_model(text_model, vision, config, self._vlm_shell)
        return self._vlm_shell

    def _release_vision_language_model(self) -> None:
        """Drop the shell's references to the encoder's modules, so unloading the encoder
        (``.to("meta")``) actually frees its ~17.5 GB instead of the shell keeping them alive."""
        if self._vlm_shell is not None:
            self._vlm_shell.language_model = None
            self._vlm_shell.visual = None

    def validate_control_rows(self, rows) -> None:
        """DatasetManager hook, run after the metadata stage and before any encode: fail fast on
        control images the vision tower (area) or the layer layout (count) cannot take, instead
        of mid text-encoder caching or at the first training step."""
        problems = control_row_problems(rows)
        if not problems:
            return
        hints = []
        if any("minimum area" in p for p in problems):
            hints.append("raise control_resolution on the [[directory]] (256 at least; non-square controls need more, each side is floored to 32 px)")
        if any("are supported" in p for p in problems):
            hints.append(f"use fewer control images per target (at most {MAX_CONDITION_IMAGES})")
        raise ValueError(
            f"qwen_image21: {len(problems)} edit row(s) have control images the model cannot train on:\n"
            + summarize_control_problems(problems, "Fix: " + "; or ".join(hints) + ".")
        )

    def encode_edit_prompts(self, text_encoder, captions: list[str], images: list[list], device):
        """``(embeds, mask, image_pad_mask)`` for captions with their condition images, checking
        that every image fills exactly the ``(W/32)*(H/32)`` vision slots its latents need (the
        processor resizes anything else: images must be multiples of 32 px and at least 256x256
        in area)."""
        vlm = self._vision_language_model(text_encoder)
        try:
            embeds, mask, image_pad_mask = encode_prompts_with_images(
                vlm, self._get_processor(), captions, images, device=device, drop_idx=self.drop_idx
            )
        finally:
            self._release_vision_language_model()
        for i, row in enumerate(images):
            expected = sum((img.size[0] // 32) * (img.size[1] // 32) for img in row)
            exact = all(img.size[0] % 32 == 0 and img.size[1] % 32 == 0 for img in row)
            if not exact or int(image_pad_mask[i].sum()) != expected:
                sizes = ", ".join(f"{img.size[0]}x{img.size[1]}" for img in row)
                raise ValueError(
                    f"qwen_image21: condition images ({sizes}) do not map 1:1 onto vision slots "
                    f"({int(image_pad_mask[i].sum())} slots, expected {expected}). Condition images "
                    "must be multiples of 32 px with an area of at least 256x256 "
                    "(raise control_resolution)."
                )
        return embeds, mask, image_pad_mask

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video, control_images=None):
            # Device of the embedding table (resident on the GPU when streaming; loads a lazy
            # encoder that was never placed).
            device = next(text_model_of(text_encoder).parameters()).device
            edit_rows = [i for i, imgs in enumerate(control_images or []) if imgs]
            if not edit_rows:
                embeds, mask = encode_prompts(
                    text_encoder, self.tokenizer, captions, device=device, drop_idx=self.drop_idx
                )
                return {"prompt_embeds": embeds, "text_mask": mask}
            # Captions with condition images go through the full VLM; any without (a mixed batch)
            # through the unchanged text-only path. image_pad_mask is ragged like the embeddings.
            rows = {}
            edit = self.encode_edit_prompts(
                text_encoder, [captions[i] for i in edit_rows], [control_images[i] for i in edit_rows], device
            )
            for j, i in enumerate(edit_rows):
                rows[i] = (edit[0][j], edit[1][j], edit[2][j])
            plain_rows = [i for i in range(len(captions)) if i not in rows]
            if plain_rows:
                embeds, mask = encode_prompts(
                    text_encoder,
                    self.tokenizer,
                    [captions[i] for i in plain_rows],
                    device=device,
                    drop_idx=self.drop_idx,
                )
                for j, i in enumerate(plain_rows):
                    rows[i] = (embeds[j], mask[j], torch.zeros_like(mask[j]))
            max_len = max(r[0].shape[0] for r in rows.values())
            first = rows[edit_rows[0]][0]
            embeds = first.new_zeros((len(captions), max_len, first.shape[-1]))
            mask = torch.zeros((len(captions), max_len), dtype=torch.bool, device=first.device)
            image_pad_mask = torch.zeros_like(mask)
            for i, (e, m, pad) in rows.items():
                embeds[i, : e.shape[0]] = e
                mask[i, : m.shape[0]] = m
                image_pad_mask[i, : pad.shape[0]] = pad
            return {"prompt_embeds": embeds, "text_mask": mask, "image_pad_mask": image_pad_mask}

        return fn

    # ---- training --------------------------------------------------------------------------

    def prepare_inputs(self, inputs, timestep_quantile=None):
        """``(noisy, t, prompt_embeds, text_mask)`` for a text-to-image batch; an edit batch (one
        with ``control_latents_*``) appends ``(img_mask, control_latents, control_layout)`` — see
        ``_edit_inputs``. Noise, the timestep shift and the loss involve the target only."""
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

        edit = _edit_inputs(inputs, prompt_embeds.shape[1], h, w)
        if edit is None:
            return (noisy_latents, t, prompt_embeds, text_mask), (target, mask)
        return (noisy_latents, t, prompt_embeds, text_mask, *edit), (target, mask)

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

    def preview_edit_prompt_embeds(
        self, prompts: list[str], control_paths: list, images: list, resolution: int, preview_cfg: dict, device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``(embeds (L, D), image_pad_mask (L,))`` on CPU per prompt, each encoded with the same
        condition ``images`` (the ones loaded from ``control_paths`` at ``resolution``). Memoized
        like ``preview_prompt_embeds``; the key includes each file's size/mtime stamp."""
        from rengu_flow.data.control import control_stamp

        identity = (tuple((str(path), control_stamp(path)) for path in control_paths), int(resolution))
        keys = [("edit", prompt, identity) for prompt in prompts]
        missing = [k for k in dict.fromkeys(keys) if k not in self._preview_embed_cache]
        if missing:
            self.ensure_text_encoder_for_preview(device)
            embeds, mask, image_pad_mask = self.encode_edit_prompts(
                self.text_encoder, [k[1] for k in missing], [images] * len(missing), device
            )
            for i, k in enumerate(missing):
                self._preview_embed_cache[k] = (embeds[i][mask[i]].cpu(), image_pad_mask[i][mask[i]].cpu())
            self.offload_text_encoder_after_encode(preview_cfg)
        return [self._preview_embed_cache[k] for k in keys]

    def prepare_preview_memory(self, preview_cfg: dict) -> None:
        self._prepare_blocks_preview_memory(preview_cfg)

    def restore_after_preview(self) -> None:
        super().restore_after_preview()
        # Preview prompts are memoized, so an encoder that caching had freed is not needed
        # again: unload it instead of parking ~16 GB of weights in host RAM between previews.
        if getattr(self, "_preview_te_rest_device", None) is not None and self._preview_te_rest_device.type == "cpu":
            self.text_encoder.to("meta")
            self._preview_te_rest_device = None

    def generate_preview_image(
        self, preview_cfg: dict, prompt: str, step: int, seed: int, control_images: list | None = None
    ):
        """One preview; with ``control_images`` (paths) it is an edit of those images."""
        from rengu_flow.model.qwen_image21.preview_sampling import generate_preview_image as _gen

        return _gen(self, preview_cfg, prompt, step, seed, control_images=control_images)
