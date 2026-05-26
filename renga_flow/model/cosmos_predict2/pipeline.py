"""Cosmos Predict2 training pipeline (Qwen3 + T5 + Wan VAE + MiniTrainDIT)."""

from __future__ import annotations

import math
from pathlib import Path

import safetensors
import torch
from torch import nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from renga_flow.config.validation import ConfigValidationError
from renga_flow.data.preprocess_media import PreprocessMediaFile
from renga_flow.model.base import BasePipeline, make_contiguous
from renga_flow.model.cosmos_predict2.config import get_dit_config
from renga_flow.model.cosmos_predict2.dit import MiniTrainDIT
from renga_flow.model.cosmos_predict2.layers import (
    FinalLayer,
    InitialLayer,
    LLMAdapterLayer,
    NoopOffloader,
    TransformerLayer,
)
from renga_flow.model.cosmos_predict2.text import compute_text_embeddings, load_text_stack, tokenize
from renga_flow.model.cosmos_predict2.vae import WanVAE, vae_encode
from renga_flow.networks import adapter_dit
from renga_flow.registry.models import register_model, register_model_alias
from renga_flow.utils.common import is_main_process, load_state_dict

KEEP_IN_HIGH_PRECISION = ["x_embedder", "t_embedder", "t_embedding_norm", "final_layer"]


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


@register_model("cosmos_predict2")
class CosmosPredict2Pipeline(BasePipeline):
    name = "cosmos_predict2"
    framerate = 16
    checkpointable_layers = ["TransformerLayer"]
    adapter_target_modules = ["Block", "TransformerBlock"]
    pixels_round_to_multiple = 16

    def __init__(self, config):
        self.config = config
        self.model_config = config["model"]
        self.offloader = NoopOffloader()
        dtype = self.model_config["dtype"]
        self.cache_text_embeddings = self.model_config.get("cache_text_embeddings", True)
        self.multiscale_loss_weight = self.model_config.get("multiscale_loss_weight", None)

        self.vae = WanVAE(vae_pth=self.model_config["vae_path"], device="cpu", dtype=dtype)
        self.vae.mean = self.vae.mean.to("cuda")
        self.vae.std = self.vae.std.to("cuda")
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        (
            self.tokenizer,
            self.t5_tokenizer,
            self.text_encoder,
            self.is_generic_llm,
            self.name,
        ) = load_text_stack(self.model_config)
        self.text_encoder.requires_grad_(False)
        self.transformer = None

    def load_diffusion_model(self):
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
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if "adapter" not in self.config:
                p.requires_grad_(True)

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

    def configure_adapter(self, adapter_config):
        self.peft_config, self.adapter_type = adapter_dit.configure(
            self.transformer, adapter_config
        )
        self.adapter_config = adapter_config
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config["dtype"])

    def save_adapter(self, save_dir, state_dict):
        adapter_dit.save(
            save_dir, state_dict, self.adapter_config, getattr(self, "peft_config", None)
        )

    def load_adapter_weights(self, adapter_path):
        adapter_dit.load_weights(self.transformer, adapter_path)

    def load_and_fuse_adapter(self, path):
        raise NotImplementedError("load_and_fuse_adapter is not implemented for cosmos_predict2")

    def save_model(self, save_dir, state_dict):
        save_dir = Path(save_dir)
        state_dict = {"net." + k: v for k, v in state_dict.items()}
        safetensors.torch.save_file(
            state_dict, save_dir / "model.safetensors", metadata={"format": "pt"}
        )

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=True, framerate=self.framerate)

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
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get("shift", None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get("flux_shift", False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents
        t = t.view(-1, 1)

        return (noisy_latents, t, *prompt_data), (target, mask)

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
        llm_adapter_lr = self.model_config.get("llm_adapter_lr", base_lr)

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

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast("cuda", enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                from renga_flow.model.loss_utils import compute_diffusion_loss_per_element

                loss = compute_diffusion_loss_per_element(output, target, self.config)
                if mask is not None and mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss

        return loss_fn

    def freeze_text_encoders(self):
        pass


register_model_alias("anima", "cosmos_predict2")

