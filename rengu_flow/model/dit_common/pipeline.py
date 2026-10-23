"""``DiTPipeline``: the pipeline methods the DiT models (krea2, cosmos_predict2, ...) share.

A DiT pipeline subclasses this instead of ``BasePipeline`` and gets, parametrized by class
attributes, the model spec (``self.spec``) and small hooks (``_load_transformer``, ``_load_vae``,
``_encode_text``):

* construction / loading — ``_init_pipeline`` (config, block-swap state, text-cache mode from
  ``spec.text_cache``), ``_component_path`` (``model.<c>_path`` or ``<spec.bundle_key>/<c>``),
  ``load_diffusion_model`` (load, optional frozen-base quantization, trainable params);
* caching — ``get_vae`` / ``get_text_encoders``, ``_latent_stats`` (diffusers VAE config),
  ``get_call_vae_fn`` (posterior mode of a ``T = 1`` image VAE, normalized),
  ``get_call_text_encoder_fn`` (ragged rows), memoized ``preview_prompt_embeds``;
* training / export — ``get_block_swap_modules`` / ``_block_swap_root_modules`` (``blocks_attr``,
  when the spec declares ``block_swap``), ``save_model`` (diffusers transformer folder), and
  ``text_padding_trimmer`` for an opt-in ``trim_micro_batch``;
* adapters — ``configure_adapter`` / ``save_adapter`` / ``load_adapter_weights`` via
  ``networks.adapter_dit`` (``adapter_target_modules``, ``adapter_layer_groups``,
  ``adapter_export_prefix``);
* ``get_loss_fn`` — masked per-element diffusion loss in fp32;
* the preview lifecycle — ``ensure_vae_for_preview``, ``ensure_text_encoder_for_preview``,
  ``offload_text_encoder_after_encode``, ``prepare_preview_memory`` (blocks streamed or resident;
  no preview block swap unless the spec's ``preview_spec.memory_knobs`` offers it),
  ``restore_after_preview`` — plus helpers for a model's own ``prepare_preview_memory``
  (``_suspend_training_block_swap``, ``_make_preview_offloader``).

Models that differ (cosmos_predict2's video VAE and optional text cache, nanosaur2's single-file
release, ...) override the method; every template is a default, never a requirement.

Expected instance attributes: ``config``, ``model_config``, ``transformer``, ``vae``,
``text_encoder`` (and ``cache_text_embeddings``; missing means cached).
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft.tuners.lora import LoraLayer

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model.base import BasePipeline
from rengu_flow.model.dit_common.text import ragged_rows, trim_text_padding
from rengu_flow.networks import adapter_dit
from rengu_flow.utils.common import is_main_process
from rengu_flow.utils.save_io import atomic_save_safetensors


def text_padding_trimmer(embeds_idx: int = 2, mask_idx: int = 3):
    """A ``trim_micro_batch`` method (``trim_micro_batch = text_padding_trimmer()`` in the class
    body): the PipelineDataLoader hook that drops the text lanes no row of a micro-batch uses
    (``trim_text_padding`` over ``features[embeds_idx]`` / ``features[mask_idx]``), so a GAS split
    of mixed caption lengths still gets the unmasked (flash) attention path. Opt-in: a model
    whose features carry no right-padded text lanes must not have the hook."""

    def trim_micro_batch(self, features):
        return trim_text_padding(features, embeds_idx=embeds_idx, mask_idx=mask_idx)

    return trim_micro_batch


class DiTPipeline(BasePipeline):
    # Class names whose nn.Linear children get adapters (adapter_dit.configure ``targets``).
    adapter_target_modules: list[str] = list(adapter_dit.ADAPTER_TARGET_MODULES)
    # Named globs for adapter.layer_groups (keys must match the model capability).
    adapter_layer_groups: dict[str, tuple[str, ...]] | None = None
    # Key prefix of exported adapters; None = adapter_dit.save's default ("diffusion_model.").
    adapter_export_prefix: str | None = None
    # Also write a ``<module>.alpha`` scalar per LoRA module (the kohya / ComfyUI convention).
    adapter_export_lora_alpha: bool = False
    # After a preview, keep a VAE that had to be reloaded from disk in host RAM (small VAEs)
    # instead of parking it back on ``meta`` and re-reading it every preview.
    preview_keep_reloaded_vae: bool = False
    # After a preview, unload a text encoder that caching had freed (see
    # ``_unload_text_encoder_freed_by_caching``): preview prompts are memoized.
    preview_unload_freed_text_encoder: bool = False
    # Attribute of the DiT holding its block list (block swap, preview block streaming).
    blocks_attr: str = "transformer_blocks"
    # Frozen-base quantization scope (``quantize_frozen_dit``); None = no quantization knobs.
    quant_leaf_names: frozenset[str] | None = None
    quant_skip_substrings: tuple[str, ...] = ()
    quant_label: str = ""
    # Why ``cache_text_embeddings`` must stay true (spec ``text_cache = "required"``), appended
    # to "<name> requires cache_text_embeddings = true" (own leading punctuation).
    text_cache_required_reason: str = ""
    # Error for a component path that is unset (and no bundle folder): ``{component}``,
    # ``{name}``, ``{bundle}`` (the spec's ``bundle_key``).
    missing_component_message: str = (
        "model.{component}_path is required for {name}: point it at the local file or folder of "
        "the {component}."
    )

    # ---- hooks (model-specific) -------------------------------------------------------------

    def _load_transformer(self, dtype: torch.dtype) -> torch.nn.Module:
        """The DiT (from ``self._component_path("transformer")``) with weights in ``dtype``."""
        raise NotImplementedError

    def _load_vae(self):
        """The frozen VAE in ``model.dtype`` (``__init__`` and the preview reload)."""
        raise NotImplementedError

    def _encode_text(self, text_encoder, prompts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(embeds (B, L, *feature), mask (B, L) bool)`` of ``prompts`` on ``device`` — what the
        caption cache and the preview prompts both store (valid rows only)."""
        raise NotImplementedError

    def _preview_vae_module(self) -> torch.nn.Module:
        """The nn.Module holding the VAE weights (checked for / parked on ``meta``)."""
        return self.vae

    def _reload_vae_for_preview(self) -> None:
        """Reload the VAE from disk into ``self.vae`` (caching parked it on ``meta``)."""
        self.vae = self._load_vae()

    def _reload_text_encoder_for_preview(self) -> torch.nn.Module:
        """Load the text encoder from disk and return it (caching parked it on ``meta``).

        Default: a ``LazyStreamedEncoder`` reloads itself on its next ``.to(<device>)`` /
        forward, so it is returned as is; eager encoders override this."""
        from rengu_flow.model.dit_common.streaming import LazyStreamedEncoder

        if isinstance(self.text_encoder, LazyStreamedEncoder):
            return self.text_encoder
        raise NotImplementedError

    # ---- construction and loading -----------------------------------------------------------

    @property
    def spec(self):
        """The model's registered ``ModelCapability`` (``None`` for an unregistered class)."""
        from rengu_flow.registry.model_capabilities import get_capability

        return get_capability(getattr(type(self), "name", None))

    def _init_pipeline(self, config) -> None:
        """The common ``__init__`` head: ``config`` / ``model_config``, block-swap state, the
        text-cache mode (``spec.text_cache = "required"`` refuses ``cache_text_embeddings =
        false``), no DiT yet (``load_diffusion_model``) and an empty preview-embedding memo."""
        self.config = config
        self.model_config = config["model"]
        self._init_block_swap_state()
        cached = self.model_config.get("cache_text_embeddings", True)
        spec = self.spec
        if spec is not None and spec.text_cache == "required":
            if not cached:
                raise ConfigValidationError(
                    f"{self.name} requires cache_text_embeddings = true{self.text_cache_required_reason}"
                )
            cached = True
        self.cache_text_embeddings = cached
        if spec is not None and spec.refused_dtypes:
            from rengu_flow.registry.model_config_rules import refused_dtype_problem

            if (problem := refused_dtype_problem(spec, self.model_config)) is not None:
                raise ConfigValidationError(problem)
        self.transformer = None
        self._preview_embed_cache: dict = {}

    def _component_path(self, component: str) -> str:
        """Resolve a component to what the user assigned: ``model.<component>_path`` (a local
        file or folder), else the ``<component>`` subfolder of the spec's bundle folder
        (``model.<bundle_key>``). Never a repo id, never downloaded."""
        override = self.model_config.get(f"{component}_path")
        if override:
            return str(override)
        spec = self.spec
        bundle = spec.bundle_key if spec is not None else None
        root = self.model_config.get(bundle) if bundle else None
        if not root:
            raise ConfigValidationError(
                self.missing_component_message.format(component=component, name=self.name, bundle=bundle)
            )
        return str(Path(root) / component)

    def load_diffusion_model(self, *, force: bool = False) -> None:
        """Load the DiT in ``model.transformer_dtype`` (default ``model.dtype``), quantize its
        frozen base when the model declares a scope (``quant_leaf_names``; the
        ``model.transformer_fp8_matmul`` / ``model.transformer_4bit`` knobs), then make it
        trainable. A no-op once loaded unless ``force``."""
        if self.transformer is not None and not force:
            return
        dtype = self.model_config.get("transformer_dtype", self.model_config["dtype"])
        self.transformer = self._load_transformer(dtype)
        if self.quant_leaf_names is not None:
            from rengu_flow.model.dit_common.quantize import quantize_frozen_dit

            quantize_frozen_dit(
                self.transformer,
                self.model_config,
                leaf_names=self.quant_leaf_names,
                skip_substrings=self.quant_skip_substrings,
                label=self.quant_label,
            )
        self._init_transformer_params()

    # ---- caching hooks ----------------------------------------------------------------------

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def _latent_stats(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel ``(mean, std)`` ``(1, C, 1, 1)`` of a diffusers VAE config
        (``latents_mean`` / ``latents_std``): cached latents are ``(z - mean) / std``."""
        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1)
        return mean, std

    def _vae_pixels(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dataset pixels ``(B, 3, H, W)`` (already on the VAE's device/dtype) -> VAE input."""
        return tensor

    def encode_latents(self, vae, tensor: torch.Tensor) -> torch.Tensor:
        """Normalized latents ``(B, C, h, w)`` of pixels ``(B, 3, H, W)`` in ``[-1, 1]`` through a
        video-shaped (``T = 1``) image VAE. The posterior *mode*: a cached latent is reused every
        epoch, so one frozen random draw would bake that noise in for the whole run."""
        p = next(vae.parameters())
        tensor = self._vae_pixels(tensor.to(p.device, p.dtype))
        latents = vae.encode(tensor.unsqueeze(2)).latent_dist.mode().squeeze(2)
        mean, std = self._latent_stats(latents.device, latents.dtype)
        return (latents - mean) / std

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            return {"latents": self.encode_latents(vae, tensor)}

        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        """Cache fn: ``_encode_text`` on the encoder's device, stored as ragged rows (one per
        caption at its own valid length; see ``ragged_rows``)."""

        def fn(captions, is_video):
            device = next(text_encoder.parameters()).device
            return ragged_rows(*self._encode_text(text_encoder, captions, device))

        return fn

    def preview_prompt_embeds(self, prompts: list[str], preview_cfg: dict, device) -> list[torch.Tensor]:
        """Per prompt, its valid-token embeddings on CPU, memoized across previews (the text
        encoder is loaded only for prompts never seen). Encoded without autocast, exactly like
        the caption cache."""

        def encode(missing):
            with torch.autocast(torch.device(device).type, enabled=False):
                embeds, mask = self._encode_text(self.text_encoder, missing, device)
            return [embeds[i][mask[i]].cpu() for i in range(len(missing))]

        return self._memoized_preview_embeds(prompts, preview_cfg, device, encode)

    # ---- block swap and export --------------------------------------------------------------

    def get_block_swap_modules(self) -> list[torch.nn.Module]:
        """``transformer.<blocks_attr>`` when the spec declares ``block_swap`` (else none)."""
        spec = self.spec
        if self.transformer is None or spec is None or not spec.features.get("block_swap"):
            return []
        return list(getattr(self.transformer, self.blocks_attr))

    def _block_swap_root_modules(self) -> list:
        return [self.transformer]

    def save_model(self, save_dir, state_dict):
        """Write a diffusers-layout transformer folder (``config.json`` + weights), loadable by
        the DiT class's ``from_pretrained`` and as a diffusers pipeline's ``transformer``."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.transformer.save_config(save_dir)
        atomic_save_safetensors(save_dir / "diffusion_pytorch_model.safetensors", state_dict)

    # ---- shared pipeline defaults -----------------------------------------------------------

    def _init_transformer_params(self) -> None:
        """After loading the DiT: train mode, ``original_name`` on every parameter (export keys),
        and every parameter trainable for a full finetune (adapters freeze the base later)."""
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if "adapter" not in self.config:
                p.requires_grad_(True)

    def get_preprocess_media_file_fn(self, augmentation_resolver=None):
        """Image-only preprocessing (video DiTs override)."""
        from rengu_flow.data.preprocess_media import PreprocessMediaFile

        return PreprocessMediaFile(self.config, support_video=False, augmentation_resolver=augmentation_resolver)

    def load_and_fuse_adapter(self, path):
        raise NotImplementedError(f"load_and_fuse_adapter is not implemented for {self.name}")

    # ---- adapters ---------------------------------------------------------------------------

    def configure_adapter(self, adapter_config):
        # The default scope (spec.adapter_default_layer_groups; empty = every targeted linear).
        spec = self.spec
        default_groups = tuple(spec.adapter_default_layer_groups) if spec is not None else ()
        extra = {"default_layer_groups": default_groups} if default_groups else {}
        self.peft_config, self.adapter_type = adapter_dit.configure(
            self.transformer,
            adapter_config,
            targets=tuple(self.adapter_target_modules),
            layer_groups=self.adapter_layer_groups,
            **extra,
        )
        self.adapter_config = adapter_config
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config["dtype"])
        # PEFT cast the fresh LoRA init to the fp8 base dtype; the recast above keeps that
        # quantized lora_A, so re-init in the adapter dtype (init_from_existing loads after).
        fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
        for m in self.transformer.modules():
            if isinstance(m, LoraLayer) and m.get_base_layer().weight.dtype in fp8:
                for name in m.lora_A:
                    m.reset_lora_parameters(name, True)

    def save_adapter(self, save_dir, state_dict):
        kwargs = {}
        if self.adapter_export_prefix is not None:
            kwargs["export_prefix"] = self.adapter_export_prefix
        if self.adapter_export_lora_alpha:
            kwargs["lora_alpha_keys"] = True
        adapter_dit.save(
            save_dir, state_dict, self.adapter_config, getattr(self, "peft_config", None), **kwargs
        )

    def load_adapter_weights(self, adapter_path):
        adapter_type = getattr(self, "adapter_type", None) or (self.config.get("adapter") or {}).get("type")
        if adapter_type and adapter_type.startswith("lycoris_"):
            from rengu_flow.networks import lycoris_dit

            lycoris_dit.load(self.transformer, adapter_path)
        else:
            alpha = (getattr(self, "adapter_config", None) or self.config.get("adapter") or {}).get("alpha")
            adapter_dit.load_weights(self.transformer, adapter_path, expected_alpha=alpha)

    # ---- loss -------------------------------------------------------------------------------

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

    # ---- previews ---------------------------------------------------------------------------

    def ensure_vae_for_preview(self) -> None:
        """Reload the VAE when dataset caching parked it on ``meta``."""
        try:
            param = next(self._preview_vae_module().parameters())
        except StopIteration:
            return
        if param.device.type != "meta":
            return
        if is_main_process():
            print("rengu_flow: loading VAE weights for preview...", flush=True)
        self._reload_vae_for_preview()
        state = getattr(self, "_preview_restore_state", None)
        if state is None:
            self._preview_restore_state = {}
            state = self._preview_restore_state
        state["vae_was_meta"] = True

    def ensure_text_encoder_for_preview(self, device: str | torch.device = "cuda") -> None:
        """Make the text encoder available on *device* for a preview.

        When caching freed it to ``meta`` it is loaded from disk the *first* time, then kept
        resident on CPU between previews (see ``restore_after_preview``), so later previews
        only pay a CPU->GPU copy instead of a disk reload.
        """
        try:
            param = next(self.text_encoder.parameters())
        except StopIteration:
            return
        if param.device.type == "meta":
            if is_main_process():
                print("rengu_flow: loading text encoder weights for preview...", flush=True)
            self.text_encoder = self._reload_text_encoder_for_preview()
            # Freed by caching: its home between previews is CPU (host RAM), not meta/GPU.
            self._preview_te_rest_device = torch.device("cpu")
        elif getattr(self, "_preview_te_rest_device", None) is None:
            # Resident (e.g. used during training): remember where to put it back afterwards.
            self._preview_te_rest_device = param.device
        self.text_encoder.to(device)

    def _memoized_preview_embeds(self, keys: list, preview_cfg: dict, device, encode_missing) -> list:
        """Preview prompt embeddings memoized across previews (preview prompts repeat every
        preview, so the text encoder is only loaded for keys never seen).

        ``encode_missing(missing_keys)`` returns one CPU value per key, in order; it runs with
        the text encoder on *device*, which is offloaded afterwards per ``preview_cfg``."""
        cache = self.__dict__.setdefault("_preview_embed_cache", {})
        missing = [k for k in dict.fromkeys(keys) if k not in cache]
        if missing:
            self.ensure_text_encoder_for_preview(device)
            for key, value in zip(missing, encode_missing(missing), strict=True):
                cache[key] = value
            self.offload_text_encoder_after_encode(preview_cfg)
        return [cache[k] for k in keys]

    def _unload_text_encoder_freed_by_caching(self) -> None:
        """After a preview: preview prompts are memoized, so a text encoder that caching had
        freed (its resting device is CPU) is not needed again — unload it to ``meta`` instead of
        parking its weights in host RAM between previews."""
        rest = getattr(self, "_preview_te_rest_device", None)
        if rest is not None and rest.type == "cpu":
            self.text_encoder.to("meta")
            self._preview_te_rest_device = None

    def offload_text_encoder_after_encode(self, preview_cfg: dict) -> None:
        """Move the text encoder to CPU once the preview prompts are encoded."""
        if not preview_cfg.get("preview_offload_text_encoder", True):
            return
        if not getattr(self, "cache_text_embeddings", True):
            # Training-resident TE (on-the-fly encoding): a CPU round-trip reassigns every
            # param's .data to fresh storage while the fused optimizer and the compiled
            # training graph still hold the old GPU addresses — empty_cache then frees them
            # and the next step dereferences dangling pointers (cudaErrorIllegalAddress).
            return
        try:
            param = next(self.text_encoder.parameters())
        except StopIteration:
            return
        if param.device.type == "meta":
            return
        self.text_encoder.to("cpu")

    def _training_block_swap_active(self) -> bool:
        train_offloader = getattr(self, "_block_swap_offloader", None)
        return train_offloader is not None and getattr(train_offloader, "enabled", False)

    def _suspend_training_block_swap(self):
        """Park the training offloader (if active) for a preview and return it (or ``None``).

        Its hooks fire on any forward and its retained residents hold several GB; parked, the
        preview offloader manages the blocks alone. ``restore_after_preview`` resumes it.
        """
        train_offloader = getattr(self, "_block_swap_offloader", None)
        if self._training_block_swap_active():
            train_offloader.suspend()
        return train_offloader

    def _make_preview_offloader(self, blocks, blocks_to_swap: int, device, train_offloader):
        """Block-swap offloader for the preview Euler loop over *blocks*."""
        from rengu_flow.training.block_swap import BlockSwapOffloader

        # Mirror the training offloader's frozen/trainable split: if it keeps the (small)
        # adapter params GPU-resident (swap_trainable=False), the preview offloader must too, or
        # resume() finds them stranded on CPU. Default True when there is no training offloader.
        swap_trainable = getattr(train_offloader, "_swap_trainable", True)
        return BlockSwapOffloader(blocks, blocks_to_swap, device=device, swap_trainable=swap_trainable)

    def prepare_preview_memory(self, preview_cfg: dict) -> None:
        """Blocks of ``transformer.<blocks_attr>`` streamed by a preview offloader or the whole
        DiT resident (``_prepare_blocks_preview_memory``); never streamed when the spec's
        ``preview_spec.memory_knobs`` lacks ``blocks_to_swap`` (the sampler runs the DiT whole)."""
        spec = self.spec
        preview_spec = spec.preview_spec if spec is not None else None
        if preview_spec is not None and "blocks_to_swap" not in preview_spec.memory_knobs:
            preview_cfg = {**preview_cfg, "preview_blocks_to_swap": 0}
        self._prepare_blocks_preview_memory(preview_cfg, blocks_attr=self.blocks_attr)

    def _prepare_blocks_preview_memory(self, preview_cfg: dict, blocks_attr: str = "transformer_blocks") -> None:
        """``prepare_preview_memory`` for a DiT whose blocks are ``transformer.<blocks_attr>``
        (krea2, qwen_image21): park the training offloader, then either stream the blocks through
        a preview offloader (``preview_blocks_to_swap > 0``; only the small shared modules move to
        the GPU) or make the whole DiT resident."""
        if self.transformer is None:
            self.load_diffusion_model()
        target = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        state: dict = {"transformer_was_training": self.transformer.training}
        # Park the training offloader so the preview offloader manages the blocks alone and the
        # preview text encoder has room. resume() in restore_after_preview.
        train_offloader = self._suspend_training_block_swap()
        blocks_swap = int(preview_cfg.get("preview_blocks_to_swap", 0))
        if blocks_swap > 0:
            # Blocks stream from CPU during the Euler loop; only the small shared modules move.
            for name, module in self.transformer.named_children():
                if name != blocks_attr:
                    module.to(target)
            self._preview_offloader = self._make_preview_offloader(
                getattr(self.transformer, blocks_attr), blocks_swap, target, train_offloader
            )
        else:
            self._preview_offloader = None
            # When block swap is active, suspend() parked the frozen block weights on CPU. The first
            # parameter is a never-swapped top-level module, so `param.device` can't reveal that —
            # the old `!= target` short-circuit then skipped the move and left every block's fp8
            # weight (and scale buffer) on CPU → a cuda/cpu mismatch in the preview forward. So force
            # the whole-transformer move when swapping; `.to()` carries params AND buffers back.
            # Without block swap the DiT is already resident, so keep the skip to avoid a needless
            # `.to()` (which reassigns param storage on a DeepSpeed/compiled module). If the full DiT
            # doesn't fit for a no-swap preview, set preview_blocks_to_swap > 0.
            if self._training_block_swap_active() or next(self.transformer.parameters()).device != target:
                if is_main_process():
                    print(f"rengu_flow: moving DiT to {target} for preview...", flush=True)
                self.transformer.to(target)
        self.transformer.eval()
        self._preview_restore_state = state

    def restore_after_preview(self) -> None:
        state = getattr(self, "_preview_restore_state", None) or {}
        if self.preview_keep_reloaded_vae:
            state.pop("vae_was_meta", None)
        offloader = getattr(self, "_preview_offloader", None)
        if self._training_block_swap_active():
            # Training streams the blocks itself: resume() re-parks them on the CPU masters and
            # re-arms the hooks. The preview offloader's teardown would instead pull ALL blocks
            # onto the GPU — an instant OOM with a large unquantized base.
            self._block_swap_offloader.resume()
        elif offloader is not None:
            offloader.teardown()
        self._preview_offloader = None
        # A pipeline that offloaded the DiT for the VAE decode (no block swap) puts it back.
        if state.get("transformer_offloaded_for_decode") and self.transformer is not None:
            self.transformer.to("cuda")
        if state.get("vae_was_meta"):
            self._preview_vae_module().to("meta")
        # Park the text encoder on its resting device (CPU when caching freed it, else its
        # original device) — never back to meta, so the next preview skips the disk reload.
        rest = getattr(self, "_preview_te_rest_device", None) or torch.device("cpu")
        try:
            next(self.text_encoder.parameters())
            if rest.type == "cuda":
                # Release the decode's cached blocks first: this restore also runs from the
                # preview's finally-path after an OOM, where moving GBs back onto a full
                # allocator would OOM again and turn a failed preview into a dead process.
                from rengu_flow.utils.common import empty_cuda_cache

                empty_cuda_cache()
            self.text_encoder.to(rest)
        except StopIteration:
            pass
        if state.get("transformer_was_training") and self.transformer is not None:
            self.transformer.train()
        self._preview_restore_state = None
        if self.preview_keep_reloaded_vae:
            vae = self._preview_vae_module()
            if next(vae.parameters()).device.type != "meta":
                vae.to("cpu")
        if self.preview_unload_freed_text_encoder:
            self._unload_text_encoder_freed_by_caching()
