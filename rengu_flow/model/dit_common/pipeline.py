"""``DiTPipeline``: the pipeline methods the DiT models (krea2, cosmos_predict2, ...) share.

A DiT pipeline subclasses this instead of ``BasePipeline`` and gets, parametrized by class
attributes and three small hooks:

* adapters — ``configure_adapter`` / ``save_adapter`` / ``load_adapter_weights`` via
  ``networks.adapter_dit`` (``adapter_target_modules``, ``adapter_layer_groups``,
  ``adapter_export_prefix``);
* ``get_loss_fn`` — masked per-element diffusion loss in fp32;
* the preview lifecycle pieces that do not depend on the DiT layout —
  ``ensure_vae_for_preview``, ``ensure_text_encoder_for_preview``,
  ``offload_text_encoder_after_encode``, ``restore_after_preview`` — plus two helpers for the
  model's own ``prepare_preview_memory`` (``_suspend_training_block_swap``,
  ``_make_preview_offloader``).

Expected instance attributes: ``config``, ``model_config``, ``transformer``, ``vae``,
``text_encoder`` (and ``cache_text_embeddings``; missing means cached).
"""

from __future__ import annotations

import torch

from rengu_flow.model.base import BasePipeline
from rengu_flow.networks import adapter_dit
from rengu_flow.utils.common import is_main_process


class DiTPipeline(BasePipeline):
    # Class names whose nn.Linear children get adapters (adapter_dit.configure ``targets``).
    adapter_target_modules: list[str] = list(adapter_dit.ADAPTER_TARGET_MODULES)
    # Named globs for adapter.layer_groups (keys must match the model capability).
    adapter_layer_groups: dict[str, tuple[str, ...]] | None = None
    # Key prefix of exported adapters; None = adapter_dit.save's default ("diffusion_model.").
    adapter_export_prefix: str | None = None

    # ---- hooks (model-specific) -------------------------------------------------------------

    def _preview_vae_module(self) -> torch.nn.Module:
        """The nn.Module holding the VAE weights (checked for / parked on ``meta``)."""
        return self.vae

    def _reload_vae_for_preview(self) -> None:
        """Reload the VAE from disk into ``self.vae`` (caching parked it on ``meta``)."""
        raise NotImplementedError

    def _reload_text_encoder_for_preview(self) -> torch.nn.Module:
        """Load the text encoder from disk and return it (caching parked it on ``meta``)."""
        raise NotImplementedError

    # ---- adapters ---------------------------------------------------------------------------

    def configure_adapter(self, adapter_config):
        self.peft_config, self.adapter_type = adapter_dit.configure(
            self.transformer,
            adapter_config,
            targets=tuple(self.adapter_target_modules),
            layer_groups=self.adapter_layer_groups,
        )
        self.adapter_config = adapter_config
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config["dtype"])

    def save_adapter(self, save_dir, state_dict):
        kwargs = {}
        if self.adapter_export_prefix is not None:
            kwargs["export_prefix"] = self.adapter_export_prefix
        adapter_dit.save(
            save_dir, state_dict, self.adapter_config, getattr(self, "peft_config", None), **kwargs
        )

    def load_adapter_weights(self, adapter_path):
        adapter_type = getattr(self, "adapter_type", None) or (self.config.get("adapter") or {}).get("type")
        if adapter_type and adapter_type.startswith("lycoris_"):
            from rengu_flow.networks import lycoris_dit

            lycoris_dit.load(self.transformer, adapter_path)
        else:
            adapter_dit.load_weights(self.transformer, adapter_path)

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
