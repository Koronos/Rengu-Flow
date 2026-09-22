"""Cosmos Predict2 training pipeline (Qwen3 + T5 + Wan VAE + MiniTrainDIT)."""

from __future__ import annotations

from pathlib import Path

import safetensors
import torch
from torch import nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.data.preprocess_media import PreprocessMediaFile
from rengu_flow.model import dit_common
from rengu_flow.model.cosmos_predict2.config import get_dit_config
from rengu_flow.model.cosmos_predict2.dit import MiniTrainDIT
from rengu_flow.model.cosmos_predict2.layers import (
    FinalLayer,
    InitialLayer,
    LLMAdapterLayer,
    TransformerLayer,
)
from rengu_flow.model.cosmos_predict2.text import compute_text_embeddings, load_text_stack, tokenize
from rengu_flow.model.cosmos_predict2.vae import WanVAE, vae_encode
from rengu_flow.registry.models import register_model
from rengu_flow.registry.models import register_model_alias
from rengu_flow.utils.save_io import atomic_save_safetensors
from rengu_flow.utils.common import is_main_process, load_state_dict

KEEP_IN_HIGH_PRECISION = ["x_embedder", "t_embedder", "t_embedding_norm", "final_layer"]


time_shift = dit_common.time_shift


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    return lambda x: dit_common.calculate_shift(x, x1, x2, y1, y2)


# Named layer groups for adapter.layer_groups (globs over Block module paths).
ADAPTER_LAYER_GROUPS = {
    "self_attention": ("blocks.*.self_attn.*",),
    "cross_attention": ("blocks.*.cross_attn.*",),
    "mlp": ("blocks.*.mlp.*",),
}


@register_model("cosmos_predict2")
class CosmosPredict2Pipeline(dit_common.DiTPipeline):
    name = "cosmos_predict2"
    framerate = 16
    checkpointable_layers = ["TransformerLayer"]
    adapter_target_modules = ["Block", "TransformerBlock"]
    adapter_layer_groups = ADAPTER_LAYER_GROUPS
    pixels_round_to_multiple = 16

    def __init__(self, config):
        self.config = config
        self.model_config = config["model"]
        self._init_block_swap_state()
        self.cache_text_embeddings = self.model_config.get("cache_text_embeddings", True)

        self.vae = self._load_vae()

        (
            self.tokenizer,
            self.t5_tokenizer,
            self.text_encoder,
            self.is_generic_llm,
            self.name,
        ) = load_text_stack(self.model_config)
        self.text_encoder.requires_grad_(False)
        self.transformer = None

    def _load_vae(self) -> WanVAE:
        vae = WanVAE(vae_pth=self.model_config["vae_path"], device="cpu", dtype=self.model_config["dtype"])
        vae.mean = vae.mean.to("cuda")
        vae.std = vae.std.to("cuda")
        vae.scale = [vae.mean, 1.0 / vae.std]
        return vae

    def load_diffusion_model(self, *, force: bool = False) -> None:
        if self.transformer is not None and not force:
            return
        dtype = self.model_config["dtype"]
        transformer_dtype = self.model_config.get("transformer_dtype", dtype)

        state_dict = load_state_dict(self.model_config["transformer_path"])
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("net."):
                k = k[len("net.") :]
            new_state_dict[k] = v
        state_dict = new_state_dict

        dit_config = get_dit_config(state_dict)

        if "llm_adapter_path" in self.model_config:
            self.use_llm_adapter = True
            dit_config["use_llm_adapter"] = True
            llm_adapter_state_dict = {
                k: v.to(dtype) for k, v in load_state_dict(self.model_config["llm_adapter_path"]).items()
            }
        elif "llm_adapter.out_proj.weight" in state_dict:
            self.use_llm_adapter = True
            dit_config["use_llm_adapter"] = True
            llm_adapter_state_dict = None
        else:
            self.use_llm_adapter = False
            llm_adapter_state_dict = None

        with init_empty_weights():
            transformer = MiniTrainDIT(**dit_config)
            for name, p in transformer.named_parameters():
                if name not in state_dict:
                    continue
                dtype_to_use = (
                    dtype
                    if (
                        any(kw in name for kw in KEEP_IN_HIGH_PRECISION)
                        or "llm_adapter" in name
                        or p.ndim == 1
                    )
                    else transformer_dtype
                )
                set_module_tensor_to_device(
                    transformer, name, device="cpu", dtype=dtype_to_use, value=state_dict[name]
                )

        if self.use_llm_adapter and llm_adapter_state_dict is not None:
            llm_adapter = transformer.llm_adapter
            for name, p in llm_adapter.named_parameters():
                dtype_to_use = (
                    dtype
                    if (
                        any(kw in name for kw in KEEP_IN_HIGH_PRECISION)
                        or "llm_adapter" in name
                        or p.ndim == 1
                    )
                    else transformer_dtype
                )
                set_module_tensor_to_device(
                    llm_adapter, name, device="cpu", dtype=dtype_to_use, value=llm_adapter_state_dict[name]
                )

        self.transformer = transformer
        self._maybe_quantize_frozen_dit()
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if "adapter" not in self.config:
                p.requires_grad_(True)

    def _maybe_quantize_frozen_dit(self) -> None:
        """Optionally quantize the frozen DiT's matmul linears (default-off, A/B knobs).

        ``model.transformer_fp8_matmul`` -> fp8 scaled matmul (``model.fp8_matmul_dtype``,
        default e5m2). ``model.transformer_4bit`` -> bitsandbytes NF4 base (QLoRA-style).
        Mutually exclusive. The base stays frozen; no trainable params are added. The LoKr
        adapter (configured later) composes on top via the quantization-aware vendored forward.
        """
        fp8_matmul = bool(self.model_config.get("transformer_fp8_matmul", False))
        four_bit = bool(self.model_config.get("transformer_4bit", False))
        if not fp8_matmul and not four_bit:
            return
        if fp8_matmul and four_bit:
            if is_main_process():
                print(
                    "rengu_flow: both transformer_fp8_matmul and transformer_4bit set; "
                    "using transformer_4bit and ignoring fp8."
                )
            fp8_matmul = False

        from rengu_flow.training import quantize_dit

        if four_bit:
            n = quantize_dit.convert_dit_to_4bit(
                self.transformer, compute_dtype=torch.bfloat16
            )
            if is_main_process():
                print(f"rengu_flow: quantized {n} frozen DiT linears to 4-bit NF4 (bnb).")
        else:
            fp8_name = self.model_config.get("fp8_matmul_dtype", "e5m2")
            fp8_dtype = quantize_dit.resolve_fp8_dtype(fp8_name)
            n = quantize_dit.convert_dit_to_fp8_matmul(self.transformer, fp8_dtype=fp8_dtype)
            if is_main_process():
                print(
                    f"rengu_flow: converted {n} frozen DiT linears to fp8 scaled matmul "
                    f"({fp8_name})."
                )

    def model_specific_dataset_config_validation(self, dataset_config):
        frame_buckets = dataset_config.get("frame_buckets")
        if frame_buckets is not None and 1 not in frame_buckets:
            raise ConfigValidationError(
                "cosmos_predict2 image training requires frame_buckets to include 1 "
                f"(got {frame_buckets})."
            )

    def get_vae(self):
        return self.vae.model

    def get_text_encoders(self):
        if self.cache_text_embeddings:
            return [self.text_encoder]
        return []

    def load_and_fuse_adapter(self, path):
        raise NotImplementedError("load_and_fuse_adapter is not implemented for cosmos_predict2")

    def save_model(self, save_dir, state_dict):
        save_dir = Path(save_dir)
        state_dict = {"net." + k: v for k, v in state_dict.items()}
        atomic_save_safetensors(save_dir / "model.safetensors", state_dict)

    def get_preprocess_media_file_fn(self, augmentation_resolver=None):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            augmentation_resolver=augmentation_resolver,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            return {"latents": latents}

        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            batch_encoding = tokenize(self.tokenizer, captions)
            t5_batch_encoding = tokenize(self.t5_tokenizer, captions)
            encoded_text = compute_text_embeddings(
                text_encoder, batch_encoding.input_ids, batch_encoding.attention_mask
            )
            return {
                "prompt_embeds": encoded_text,
                "attn_mask": batch_encoding.attention_mask,
                "t5_input_ids": t5_batch_encoding.input_ids,
                "t5_attn_mask": t5_batch_encoding.attention_mask,
            }

        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs["latents"].float()
        mask = inputs["mask"]

        if self.cache_text_embeddings:
            prompt_data = (
                inputs["prompt_embeds"],
                inputs["attn_mask"],
                inputs["t5_input_ids"],
                inputs["t5_attn_mask"],
            )
        else:
            captions = inputs["caption"]
            batch_encoding = tokenize(self.tokenizer, captions)
            t5_batch_encoding = tokenize(self.t5_tokenizer, captions)
            prompt_data = (
                batch_encoding.input_ids,
                batch_encoding.attention_mask,
                t5_batch_encoding.input_ids,
                t5_batch_encoding.attention_mask,
            )

        bs, _channels, _num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode="nearest-exact")
            mask = mask.unsqueeze(2)

        t = dit_common.sample_timesteps(self.model_config, bs, latents.device, timestep_quantile)
        mu = None
        if self.model_config.get("flux_shift", False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        t = dit_common.shift_timesteps(t, self.model_config.get("shift", None), mu)
        noisy_latents, target, t = dit_common.add_flow_noise(latents, t)

        return (noisy_latents, t, *prompt_data), (target, mask)

    def get_block_swap_modules(self) -> list[nn.Module]:
        if self.transformer is None:
            return []
        return list(self.transformer.blocks)

    def _block_swap_root_modules(self) -> list:
        # The DiT holds the swappable transformer.blocks; the generic _place_for_block_swap puts the
        # rest (embedders, final_layer, llm_adapter) on the GPU. The text encoder is a root only when
        # it is still in the training graph (uncached); when cached, the cache unload placed it on
        # meta and it must stay there. block.forward runs for Cosmos (one TransformerLayer per block),
        # so its no-op wait/submit calls are harmless under the hook-based offloader.
        roots = [self.transformer]
        if not self.cache_text_embeddings and isinstance(self.text_encoder, nn.Module):
            roots.append(self.text_encoder)
        return roots

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder
        layers = [
            InitialLayer(transformer, text_encoder, self.is_generic_llm),
            LLMAdapterLayer(transformer.llm_adapter if self.use_llm_adapter else None),
        ]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def get_param_groups(self, parameters):
        base_params, self_attn_params, cross_attn_params, mlp_params, mod_params, llm_adapter_params = (
            [], [], [], [], [], []
        )
        for p in parameters:
            name = p.original_name
            if "llm_adapter" in name:
                llm_adapter_params.append(p)
            elif ".self_attn" in name:
                self_attn_params.append(p)
            elif ".cross_attn" in name:
                cross_attn_params.append(p)
            elif ".mlp" in name:
                mlp_params.append(p)
            elif ".adaln_modulation" in name:
                mod_params.append(p)
            else:
                base_params.append(p)

        base_lr = self.config["optimizer"].get("lr", None)
        self_attn_lr = self.model_config.get("self_attn_lr", base_lr)
        cross_attn_lr = self.model_config.get("cross_attn_lr", base_lr)
        mlp_lr = self.model_config.get("mlp_lr", base_lr)
        mod_lr = self.model_config.get("mod_lr", base_lr)
        # Freeze the embedded Qwen3 LLM adapter by default; it has outsized influence on
        # conditioning and degrades easily. Set llm_adapter_lr explicitly to train it.
        llm_adapter_lr = self.model_config.get("llm_adapter_lr", 0)

        if is_main_process():
            print(
                f"Using base_lr={base_lr}, self_attn_lr={self_attn_lr}, cross_attn_lr={cross_attn_lr}, "
                f"mlp_lr={mlp_lr}, mod_lr={mod_lr}, llm_adapter_lr={llm_adapter_lr}"
            )

        param_groups = []
        for lr, params in [
            (base_lr, base_params),
            (self_attn_lr, self_attn_params),
            (cross_attn_lr, cross_attn_params),
            (mlp_lr, mlp_params),
            (mod_lr, mod_params),
            (llm_adapter_lr, llm_adapter_params),
        ]:
            if lr == 0:
                for p in params:
                    p.requires_grad_(False)
            elif len(params) > 0:
                param_groups.append({"params": params, "lr": lr})
        return param_groups

    def freeze_text_encoders(self):
        pass

    # ---- previews (VAE/TE lifecycle and restore come from DiTPipeline) ---------------------

    def _preview_vae_module(self) -> nn.Module:
        return self.vae.model

    def _reload_vae_for_preview(self) -> None:
        self.vae = self._load_vae()

    def _reload_text_encoder_for_preview(self) -> nn.Module:
        _, _, text_encoder, _, _ = load_text_stack(self.model_config)
        text_encoder.requires_grad_(False)
        if is_main_process():
            print("rengu_flow: text encoder ready for preview.", flush=True)
        return text_encoder

    def ensure_transformer_for_preview(self, device: str | torch.device = "cuda") -> None:
        """Use in-memory DiT on GPU for Euler (do not reload weights from disk)."""
        if self.transformer is None:
            self.load_diffusion_model()
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            # "cuda" (no index) != "cuda:0" as device objects; resolve to the current device so an
            # already-resident DiT is recognized and we skip a needless .to() (which on a DeepSpeed /
            # compiled module would reassign param storage). See offload_transformer_for_decode.
            target = torch.device("cuda", torch.cuda.current_device())
        param = next(self.transformer.parameters())
        if param.device != target:
            if is_main_process():
                print(f"rengu_flow: moving DiT to {target} for preview...", flush=True)
            self.transformer.to(target)
        self.transformer.eval()

    def prepare_preview_memory(self, preview_cfg: dict) -> None:
        """Prepare DiT for preview sampling (eval mode; optional block swap)."""
        # Park the training offloader (if any) so its hooks don't fight the preview offloader
        # over block placement and its retained GPU copies are released. resume() on restore.
        train_offloader = self._suspend_training_block_swap()
        self.ensure_transformer_for_preview("cuda")
        state: dict = {}
        blocks_swap = int(preview_cfg.get("preview_blocks_to_swap", 0))
        if blocks_swap > 0:
            self._preview_offloader = self._make_preview_offloader(
                self.transformer.blocks, blocks_swap, "cuda", train_offloader
            )
        else:
            self._preview_offloader = None
        state["transformer_was_training"] = self.transformer.training
        self._preview_restore_state = state

    def offload_transformer_for_decode(self, preview_cfg: dict | None = None) -> None:
        """Move the DiT to CPU before the VAE decode so the decoder's conv3d has contiguous VRAM.

        The decode does not use the DiT, and on a tight GPU the DiT (fully resident for sampling)
        plus the VAE plus the decode activation peak can OOM. Skipped when a block-swap offloader
        already manages residency (it would fight the streamed layout). The next prompt's
        ``euler_sample_latents`` re-ensures the DiT on GPU; ``restore_after_preview`` returns it to
        the training device for the resumed step.

        Off by default (``preview_offload_dit_for_decode``): the CPU<->GPU round-trip reassigns every
        parameter's ``.data`` to fresh storage, which a DeepSpeed engine / fused optimizer (and any
        ``torch.compile`` graph) still references at the old GPU addresses — ``empty_cache`` then frees
        those, so the next NCCL collective (the post-preview barrier) dereferences dangling buffers and
        raises ``cudaErrorIllegalAddress``. Only opt in on a tight GPU that OOMs at decode *and* is not
        DeepSpeed-managed/compiled.
        """
        if self.transformer is None:
            return
        if not (preview_cfg or {}).get("preview_offload_dit_for_decode", False):
            return
        if getattr(self, "_preview_offloader", None) is not None:
            return
        if getattr(self, "_block_swap_offloader", None) is not None:
            return
        from rengu_flow.utils.common import empty_cuda_cache

        self.transformer.to("cpu")
        state = getattr(self, "_preview_restore_state", None)
        if state is not None:
            state["transformer_offloaded_for_decode"] = True
        empty_cuda_cache()

    def generate_preview_image(self, preview_cfg: dict, prompt: str, step: int, seed: int):
        from rengu_flow.model.cosmos_predict2.preview_sampling import generate_preview_image as _gen

        return _gen(self, preview_cfg, prompt, step, seed)


register_model_alias("anima", "cosmos_predict2")

