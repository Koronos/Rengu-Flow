"""SDXL pipeline model: diffusers StableDiffusionXL with to_layers() for DeepSpeed pipeline."""

import re
from pathlib import Path

import diffusers
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.utils.logging import logger
import safetensors
from safetensors.torch import save_file

from renga_flow.data.preprocess_media import PreprocessMediaFile
from renga_flow.model.base import BasePipeline, make_contiguous
from renga_flow.registry.models import register_model
from renga_flow.utils.common import cuda_autocast, is_main_process
from renga_flow.utils.save_io import atomic_save_safetensors
from renga_flow.utils.diffusers_tf5_compat import apply_diffusers_transformers_v5_single_file_patch

# Optional: import network adapters (lora_sdxl always; lokr_sdxl may use LyCORIS or vendored)
from renga_flow import networks as networks_module


# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
]

unet_conversion_map_resnet = [
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
for i in range(3):
    for j in range(2):
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))
        if i > 0:
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))
    for j in range(4):
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))
        if i < 2:
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))
    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))
unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))
unet_conversion_map_layer.append(("middle_block.1.", "mid_block.attentions.0."))
for j in range(2):
    unet_conversion_map_layer.append((f"middle_block.{2*j}.", f"mid_block.resnets.{j}."))


def convert_unet_state_dict(unet_state_dict):
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    return {sd_name: unet_state_dict[hf_name] for hf_name, sd_name in mapping.items()}


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]
for i in range(4):
    for j in range(2):
        vae_conversion_map.append((f"encoder.down.{i}.block.{j}.", f"encoder.down_blocks.{i}.resnets.{j}."))
    if i < 3:
        vae_conversion_map.append((f"down.{i}.downsample.", f"down_blocks.{i}.downsamplers.0."))
        vae_conversion_map.append((f"up.{3-i}.upsample.", f"up_blocks.{i}.upsamplers.0."))
    for j in range(3):
        vae_conversion_map.append((f"decoder.up.{3-i}.block.{j}.", f"decoder.up_blocks.{i}.resnets.{j}."))
for i in range(2):
    vae_conversion_map.append((f"mid.block_{i+1}.", f"mid_block.resnets.{i}."))

vae_conversion_map_attn = [
    ("norm.", "group_norm."),
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
]


def reshape_weight_for_sd(w):
    if w.ndim != 1:
        return w.reshape(*w.shape, 1, 1)
    return w


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    for k, v in new_state_dict.items():
        for weight_name in ("q", "k", "v", "proj_out"):
            if f"mid.attn_1.{weight_name}.weight" in k:
                new_state_dict[k] = reshape_weight_for_sd(v)
                break
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#

textenc_conversion_lst = [
    ("transformer.resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "text_model.final_layer_norm."),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))
code2idx = {"q": 0, "k": 1, "v": 2}


def convert_openclip_text_enc_state_dict(text_enc_dict):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if k.endswith(".self_attn.q_proj.weight") or k.endswith(".self_attn.k_proj.weight") or k.endswith(".self_attn.v_proj.weight"):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code[0]]] = v
            continue
        if k.endswith(".self_attn.q_proj.bias") or k.endswith(".self_attn.k_proj.bias") or k.endswith(".self_attn.v_proj.bias"):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code[0]]] = v
            continue
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
        new_state_dict[relabelled_key] = v
    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)
    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)
    return new_state_dict


def convert_openai_text_enc_state_dict(text_enc_dict):
    return text_enc_dict


def prepare_scheduler_for_custom_training(noise_scheduler):
    if hasattr(noise_scheduler, "all_snr"):
        return
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    noise_scheduler.all_snr = (alpha / sigma) ** 2


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    logger.info("fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        return 1 - alphas

    betas = enforce_zero_terminal_snr(noise_scheduler.betas)
    alphas = 1.0 - betas
    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = torch.cumprod(alphas, dim=0)


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    return loss * snr_weight


def apply_debiased_estimation(loss, timesteps, noise_scheduler, v_prediction=False):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)
    weight = 1 / (snr_t + 1) if v_prediction else 1 / torch.sqrt(snr_t)
    return loss * weight.to(loss.device)


@register_model("sdxl")
class SDXLPipeline(BasePipeline):
    name = "sdxl"
    checkpointable_layers = [
        "InitialLayer",
        "DownBlockInnerLayer",
        "MidBlockInnerLayer",
        "UpBlockInnerLayer",
        "FinalLayer",
    ]

    def __init__(self, config):
        self.config = config
        self.model_config = self.config["model"]
        self.v_pred = self.model_config.get("v_pred", False)
        self.min_snr_gamma = self.model_config.get("min_snr_gamma", None)
        self.debiased_estimation_loss = self.model_config.get("debiased_estimation_loss", None)
        self.cache_text_embeddings = self.model_config.get("cache_text_embeddings", True)
        self.clip_skip = self.model_config.get("clip_skip", None)
        self._pipeline = None

        if self.v_pred:
            logger.info("Using v-prediction loss")
        if self.min_snr_gamma is not None:
            logger.info(f"Using min_snr_gamma={self.min_snr_gamma}")
        if self.debiased_estimation_loss:
            logger.info("Using debiased_estimation_loss")

    @property
    def diffusers_pipeline(self):
        self.load_diffusion_model()
        return self._pipeline

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def _set_param_original_name(self):
        for state_dict_key_prefix, module in (
            ("unet.", self.unet),
            ("text_encoder.", self.text_encoder),
            ("text_encoder_2.", self.text_encoder_2),
        ):
            for pname, p in module.named_parameters():
                p.original_name = state_dict_key_prefix + pname

    def load_diffusion_model(self) -> None:
        if self._pipeline is not None:
            return
        apply_diffusers_transformers_v5_single_file_patch()
        self._pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(
            self.model_config["checkpoint_path"],
            torch_dtype=self.model_config["dtype"],
            add_watermarker=False,
        )
        self._pipeline.tokenizer_2 = self._pipeline.tokenizer
        self._pipeline.scheduler = diffusers.DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        prepare_scheduler_for_custom_training(self._pipeline.scheduler)
        if self.v_pred:
            fix_noise_scheduler_betas_for_zero_terminal_snr(self._pipeline.scheduler)
        self._pipeline.upcast_vae()
        self._pipeline.unet.train()
        self._pipeline.text_encoder.train()
        self._pipeline.text_encoder_2.train()
        self._set_param_original_name()

    def freeze_text_encoders(self) -> None:
        """Freeze text encoder parameters when doing full-model finetuning with only UNet trained."""
        if self.model_config.get("freeze_text_encoders", False):
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
            for p in self.text_encoder_2.parameters():
                p.requires_grad_(False)
            if is_main_process():
                logger.info("Full-model SDXL: text encoders frozen (training UNet only)")

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        if not self.cache_text_embeddings:
            return []
        pipe = self.diffusers_pipeline
        return [pipe.text_encoder, pipe.text_encoder_2]

    def configure_adapter(self, adapter_config):
        self.adapter_config = adapter_config
        self.adapter_type = adapter_config["type"]
        if self.adapter_type == "lora" and adapter_config.get("init_from_existing"):
            return
        if self.adapter_type == "lora":
            unet, te, te2 = networks_module.lora_sdxl.configure(
                self.unet,
                self.text_encoder,
                self.text_encoder_2,
                adapter_config,
            )
            self._pipeline.unet = unet
            self._pipeline.text_encoder = te
            self._pipeline.text_encoder_2 = te2
        elif self.adapter_type == "lokr":
            networks_module.lokr_sdxl.configure(
                self.unet,
                self.text_encoder,
                self.text_encoder_2,
                adapter_config,
            )
        else:
            raise NotImplementedError(f"Adapter type {self.adapter_type} is not implemented")

    def save_adapter(self, save_dir, state_dict):
        save_dir = Path(save_dir)
        adapter_type = getattr(self, "adapter_type", self.config.get("adapter", {}).get("type", "lora"))
        if adapter_type == "lora":
            networks_module.lora_sdxl.save(save_dir, state_dict, self.adapter_config)
        elif adapter_type == "lokr":
            networks_module.lokr_sdxl.save(save_dir, state_dict, self.adapter_config)
        else:
            raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")

    def load_adapter_weights(self, adapter_path):
        adapter_path = Path(adapter_path)
        files = list(adapter_path.glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"No .safetensors file found in {adapter_path}")
        state = safetensors.torch.load_file(files[0])
        is_lokr = any("lokr_" in k for k in state.keys())
        adapter_type = getattr(self, "adapter_type", None) or (self.config.get("adapter") or {}).get("type")
        if is_lokr or adapter_type == "lokr":
            networks_module.lokr_sdxl.load(self, adapter_path)
        else:
            networks_module.lora_sdxl.load(self.diffusers_pipeline, adapter_path)
        self._set_param_original_name()

    def load_and_fuse_adapter(self, path):
        path = Path(path)
        files = list(path.glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"No .safetensors file found in {path}")
        state = safetensors.torch.load_file(files[0])
        is_lokr = any("lokr_" in k for k in state.keys())

        if is_lokr:
            self._load_and_fuse_lokr(path, state)
        else:
            self._load_and_fuse_lora(path, state)

    def _load_and_fuse_lora(self, path, state):
        """Load LoRA from path/state into pipeline and fuse into base weights."""
        pipe = self.diffusers_pipeline
        networks_module.lora_sdxl.load(pipe, path)
        fuse_lora = getattr(pipe, "fuse_lora", None)
        if fuse_lora is not None:
            fuse_lora(fuse_unet=True, fuse_text_encoder=True, lora_scale=1.0)
        else:
            for module in (pipe.unet, pipe.text_encoder, pipe.text_encoder_2):
                if hasattr(module, "merge_and_unload"):
                    module.merge_and_unload()
        self._set_param_original_name()

    def _load_and_fuse_lokr(self, path, state):
        """Configure LoKr if needed, load weights, then fuse into base weights."""
        adapter_type = getattr(self, "adapter_type", None) or (self.config.get("adapter") or {}).get("type")
        if adapter_type != "lokr":
            adapter_config = networks_module.lokr_sdxl.infer_lokr_config_from_state(state)
            adapter_config["type"] = "lokr"
            existing = self.config.get("adapter") or {}
            if "rank" not in adapter_config and "dim" in existing:
                adapter_config["rank"] = existing["dim"]
            if adapter_config.get("dtype") is None:
                model_dtype = self.model_config.get("dtype", torch.float32)
                adapter_config["dtype"] = model_dtype if isinstance(model_dtype, torch.dtype) else torch.float32
            self.configure_adapter(adapter_config)
        self.load_adapter_weights(path)
        networks_module.lokr_sdxl.fuse(self)
        self._set_param_original_name()

    def save_model(self, save_dir, diffusers_sd):
        save_dir = Path(save_dir)
        unet_state_dict, text_enc_dict, text_enc_2_dict = {}, {}, {}
        for name, p in diffusers_sd.items():
            if name.startswith("unet."):
                unet_state_dict[name[len("unet.") :]] = p
            elif name.startswith("text_encoder."):
                text_enc_dict[name[len("text_encoder.") :]] = p
            elif name.startswith("text_encoder_2."):
                text_enc_2_dict[name[len("text_encoder_2.") :]] = p
            else:
                raise RuntimeError(f"Unexpected parameter: {name}")
        vae_state_dict = self.vae.state_dict()
        unet_state_dict = convert_unet_state_dict(unet_state_dict)
        unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}
        vae_state_dict = convert_vae_state_dict(vae_state_dict)
        vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}
        text_enc_dict = convert_openai_text_enc_state_dict(text_enc_dict)
        text_enc_dict = {"conditioner.embedders.0.transformer." + k: v for k, v in text_enc_dict.items()}
        text_enc_2_dict = convert_openclip_text_enc_state_dict(text_enc_2_dict)
        text_enc_2_dict = {"conditioner.embedders.1.model." + k: v for k, v in text_enc_2_dict.items()}
        text_enc_2_dict["conditioner.embedders.1.model.text_projection"] = text_enc_2_dict.pop(
            "conditioner.embedders.1.model.text_projection.weight"
        ).T.contiguous()
        state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict, **text_enc_2_dict}
        atomic_save_safetensors(save_dir / "model.safetensors", state_dict)

    def get_preprocess_media_file_fn(self, augmentation_resolver=None):
        return PreprocessMediaFile(
            self.config,
            support_video=False,
            round_height=16,
            round_width=16,
            augmentation_resolver=augmentation_resolver,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {"latents": latents}

        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        pipe = self.diffusers_pipeline
        is_te2 = text_encoder is pipe.text_encoder_2

        def fn(captions, is_video):
            if is_te2:
                prompt_embeds_2, pooled = self._encode_prompt_embeds_batch(
                    captions, pipe.tokenizer_2, text_encoder, return_pooled_prompt_embeds=True
                )
                return {
                    "prompt_embeds_2": prompt_embeds_2,
                    "pooled_prompt_embeds": pooled,
                }
            return {
                "prompt_embeds": self._encode_prompt_embeds_batch(
                    captions, pipe.tokenizer, text_encoder, return_pooled_prompt_embeds=False
                )
            }

        return fn

    def _encode_prompt_embeds_batch(
        self, captions, tokenizer, text_encoder, return_pooled_prompt_embeds=False
    ):
        chunks_out = []
        pooled_list = []
        for caption in captions:
            input_ids = self._get_input_ids([caption], tokenizer)
            embed, pooled = self._encode_prompt_embeds_from_input_ids(
                input_ids, tokenizer, text_encoder, return_pooled_prompt_embeds
            )
            chunks_out.append(embed)
            if return_pooled_prompt_embeds:
                pooled_list.append(pooled)
        prompt_embeds = torch.cat(chunks_out, dim=0)
        if return_pooled_prompt_embeds:
            return prompt_embeds, torch.cat(pooled_list, dim=0)
        return prompt_embeds

    def _encode_prompt_embeds_from_input_ids(
        self, input_ids, tokenizer, text_encoder, return_pooled_prompt_embeds=False
    ):
        te_device = next(text_encoder.parameters()).device
        input_ids = input_ids.to(te_device)
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        bs, device = input_ids.shape[0], te_device
        chunks = torch.split(input_ids, tokenizer.model_max_length - 2, dim=-1)
        processed_chunks = []
        for chunk in chunks:
            chunk = torch.cat(
                [torch.full((bs, 1), bos, device=device), chunk, torch.full((bs, 1), pad, device=device)],
                dim=-1,
            )
            first_pad_idx = torch.argmax((chunk == pad).to(torch.int32), dim=-1)
            chunk[torch.arange(chunk.shape[0]), first_pad_idx] = eos
            processed_chunks.append(chunk)
        embed_chunks = []
        pooled_prompt_embeds = None
        for i, input_ids_chunk in enumerate(processed_chunks):
            prompt_embeds = text_encoder(input_ids_chunk, output_hidden_states=True)
            if i == 0 and return_pooled_prompt_embeds:
                pooled_prompt_embeds = prompt_embeds[0]
            hidden = (
                prompt_embeds.hidden_states[-(self.clip_skip + 2)]
                if self.clip_skip is not None
                else prompt_embeds.hidden_states[-2]
            )
            embed_chunks.append(hidden)
        out = torch.cat(embed_chunks, dim=1)
        if return_pooled_prompt_embeds:
            return out, pooled_prompt_embeds
        return out, None

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs["latents"].float()
        mask = inputs["mask"]
        bs, channels, h, w = latents.shape
        device = latents.device
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode="nearest-exact")
        noise = torch.randn_like(latents, device=device)
        max_timestep = self.scheduler.config.num_train_timesteps
        if timestep_quantile is not None:
            timesteps = torch.full((bs,), int(timestep_quantile * max_timestep), device=device)
        else:
            timesteps = torch.randint(0, max_timestep, (bs,), device=device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        target = self.scheduler.get_velocity(latents, noise, timesteps) if self.v_pred else noise
        pixel_height = latents.shape[-2] * self.vae_scale_factor
        pixel_width = latents.shape[-1] * self.vae_scale_factor
        original_size = target_size = (pixel_height, pixel_width)
        add_time_ids = self._get_add_time_ids(
            original_size, (0, 0), target_size, dtype=torch.float32, text_encoder_projection_dim=self.text_encoder_2.config.projection_dim
        ).expand(bs, -1)

        if self.cache_text_embeddings:
            encoder_hidden_states = torch.cat(
                [inputs["prompt_embeds"], inputs["prompt_embeds_2"]], dim=-1
            )
            pooled_prompt_embeds = inputs["pooled_prompt_embeds"]
            return (
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                pooled_prompt_embeds,
                add_time_ids,
            ), (target, mask)

        caption = inputs["caption"]
        input_ids = self._get_input_ids(caption, self.tokenizer)
        input_ids_2 = self._get_input_ids(caption, self.tokenizer_2)
        return (noisy_latents, timesteps, input_ids, input_ids_2, add_time_ids), (target, mask)

    def _get_input_ids(self, prompt, tokenizer):
        return tokenizer(prompt, padding="longest", truncation=False, add_special_tokens=False, return_tensors="pt").input_ids.to(torch.int64)

    def to_layers(self):
        layers = [
            InitialLayer(
                self.diffusers_pipeline,
                cache_text_embeddings=self.cache_text_embeddings,
                clip_skip=self.clip_skip,
            )
        ]
        unet = self.diffusers_pipeline.unet
        for block in unet.down_blocks:
            layers.extend(UnetDownBlockLayer(block).to_layers())
        if unet.mid_block is not None:
            layers.extend(UnetMidBlockLayer(unet.mid_block).to_layers())
        for i, block in enumerate(unet.up_blocks):
            layers.extend(UnetUpBlockLayer(block, i == len(unet.up_blocks) - 1).to_layers())
        layers.append(FinalLayer(unet, self))
        return layers

    def get_param_groups(self, parameters):
        unet_params, text_encoder_params, text_encoder_2_params = [], [], []
        for p in parameters:
            if p.original_name.startswith("unet."):
                unet_params.append(p)
            elif p.original_name.startswith("text_encoder."):
                text_encoder_params.append(p)
            elif p.original_name.startswith("text_encoder_2."):
                text_encoder_2_params.append(p)
            else:
                raise RuntimeError(f"Unexpected parameter: {p.original_name}")
        base_lr = self.config["optimizer"].get("lr", None)
        unet_lr = self.model_config.get("unet_lr", base_lr)
        text_encoder_lr = self.model_config.get("text_encoder_1_lr", base_lr)
        text_encoder_2_lr = self.model_config.get("text_encoder_2_lr", base_lr)
        if is_main_process():
            print(f"Using unet_lr={unet_lr}, text_encoder_1_lr={text_encoder_lr}, text_encoder_2_lr={text_encoder_2_lr}")
        result = [{"params": unet_params}]
        if unet_lr is not None:
            result[-1]["lr"] = unet_lr
        result.append({"params": text_encoder_params})
        if text_encoder_lr is not None:
            result[-1]["lr"] = text_encoder_lr
        result.append({"params": text_encoder_2_params})
        if text_encoder_2_lr is not None:
            result[-1]["lr"] = text_encoder_2_lr
        return result

    def get_loss_fn(self):
        def loss_fn(output, label):
            output, timesteps = output
            target, mask = label
            with torch.autocast("cuda", enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                from renga_flow.model.loss_utils import compute_diffusion_loss_per_element

                loss = compute_diffusion_loss_per_element(output, target, self.config)
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean([1, 2, 3])
                if self.min_snr_gamma is not None:
                    loss = apply_snr_weight(loss, timesteps, self.scheduler, self.min_snr_gamma, self.v_pred)
                if self.debiased_estimation_loss is not None:
                    loss = apply_debiased_estimation(loss, timesteps, self.scheduler, self.v_pred)
                loss = loss.mean()
            return loss

        return loss_fn


class InitialLayer(nn.Module):
    def __init__(self, diffusers_pipeline, cache_text_embeddings=False, clip_skip=None):
        super().__init__()
        self.cache_text_embeddings = cache_text_embeddings
        self.clip_skip = clip_skip
        self.diffusers_pipeline = diffusers_pipeline
        # Do not register TE submodules when embeddings are cached (TE weights live on meta after cache).
        if cache_text_embeddings:
            self.text_encoder = None
            self.text_encoder_2 = None
        else:
            self.text_encoder = self.diffusers_pipeline.text_encoder
            self.text_encoder_2 = self.diffusers_pipeline.text_encoder_2
        self.tokenizer = self.diffusers_pipeline.tokenizer
        self.tokenizer_2 = self.diffusers_pipeline.tokenizer_2
        self.time_proj = self.diffusers_pipeline.unet.time_proj
        self.time_embedding = self.diffusers_pipeline.unet.time_embedding
        self.add_embedding = self.diffusers_pipeline.unet.add_embedding
        self.time_embed_act = self.diffusers_pipeline.unet.time_embed_act
        self.encoder_hid_proj = self.diffusers_pipeline.unet.encoder_hid_proj
        self.conv_in = self.diffusers_pipeline.unet.conv_in

    @property
    def unet(self):
        return self.diffusers_pipeline.unet

    def forward(self, inputs):
        with cuda_autocast():
            for tensor in inputs:
                if torch.is_floating_point(tensor):
                    tensor.requires_grad_(True)
            sample, timestep, te_a, te_b, add_time_ids = inputs
            default_overall_up_factor = 2 ** self.unet.num_upsamplers
            forward_upsample_size = any(dim % default_overall_up_factor != 0 for dim in sample.shape[-2:])
            forward_upsample_size = torch.tensor(forward_upsample_size).to(sample.device)
            if self.cache_text_embeddings or torch.is_floating_point(te_a):
                encoder_hidden_states, pooled_prompt_embeds = te_a, te_b
            else:
                encoder_hidden_states, pooled_prompt_embeds = self.get_text_conditioning(te_a, te_b)
            add_time_ids = add_time_ids.to(pooled_prompt_embeds.dtype)
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            t_emb = self.unet.get_time_embed(sample=sample, timestep=timestep)
            emb = self.unet.time_embedding(t_emb, None)
            aug_emb = self.unet.get_aug_embed(emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
            emb = emb + aug_emb if aug_emb is not None else emb
            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)
            encoder_hidden_states = self.unet.process_encoder_hidden_states(encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
            sample = self.conv_in(sample)
            down_block_res_samples = (sample,)
            return make_contiguous(sample, timestep, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size)

    def get_text_conditioning(self, input_ids, input_ids_2):
        prompt_embeds = self.get_prompt_embeds(input_ids, self.tokenizer, self.text_encoder)
        prompt_embeds_2, pooled_prompt_embeds = self.get_prompt_embeds(input_ids_2, self.tokenizer_2, self.text_encoder_2, return_pooled_prompt_embeds=True)
        return torch.concat([prompt_embeds, prompt_embeds_2], dim=-1), pooled_prompt_embeds

    def get_prompt_embeds(self, input_ids, tokenizer, text_encoder, return_pooled_prompt_embeds=False):
        te_device = next(text_encoder.parameters()).device
        input_ids = input_ids.to(te_device)
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        bs, device = input_ids.shape[0], te_device
        chunks = torch.split(input_ids, tokenizer.model_max_length - 2, dim=-1)
        processed_chunks = []
        for chunk in chunks:
            chunk = torch.cat([torch.full((bs, 1), bos, device=device), chunk, torch.full((bs, 1), pad, device=device)], dim=-1)
            first_pad_idx = torch.argmax((chunk == pad).to(torch.int32), dim=-1)
            chunk[torch.arange(chunk.shape[0]), first_pad_idx] = eos
            processed_chunks.append(chunk)
        embed_chunks = []
        pooled_prompt_embeds = None
        for i, input_ids_chunk in enumerate(processed_chunks):
            prompt_embeds = text_encoder(input_ids_chunk, output_hidden_states=True)
            if i == 0 and return_pooled_prompt_embeds:
                pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-(self.clip_skip + 2)] if self.clip_skip is not None else prompt_embeds.hidden_states[-2]
            embed_chunks.append(prompt_embeds)
        out = torch.cat(embed_chunks, dim=1)
        if return_pooled_prompt_embeds:
            return out, pooled_prompt_embeds
        return out


class DownBlockInnerLayer(nn.Module):
    def __init__(self, resnet, attn, append_residual_hidden_states=True):
        super().__init__()
        self.resnet = resnet
        self.attn = attn
        self.append_residual_hidden_states = append_residual_hidden_states

    def forward(self, inputs):
        with cuda_autocast():
            hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
            hidden_states = self.resnet(hidden_states, emb)
            if self.attn is not None:
                hidden_states = self.attn(hidden_states, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
            res_hidden_states += (hidden_states,)
            return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class MidBlockInnerLayer(nn.Module):
    def __init__(self, resnet, attn):
        super().__init__()
        self.resnet = resnet
        self.attn = attn

    def forward(self, inputs):
        with cuda_autocast():
            hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
            hidden_states = self.resnet(hidden_states, emb)
            if self.attn is not None:
                hidden_states = self.attn(hidden_states, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
            return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class UpBlockInnerLayer(nn.Module):
    def __init__(self, resnet, attn):
        super().__init__()
        self.resnet = resnet
        self.attn = attn

    def forward(self, inputs):
        with cuda_autocast():
            hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
            res_tmp = res_hidden_states[-1]
            res_hidden_states = res_hidden_states[:-1]
            hidden_states = torch.cat([hidden_states, res_tmp], dim=1)
            hidden_states = self.resnet(hidden_states, emb)
            if self.attn is not None:
                hidden_states = self.attn(hidden_states, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
            return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class DownsamplerLayer(nn.Module):
    def __init__(self, downsamplers):
        super().__init__()
        self.downsamplers = downsamplers

    def forward(self, inputs):
        with cuda_autocast():
            hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            res_hidden_states += (hidden_states,)
            return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class UpsamplerLayer(nn.Module):
    def __init__(self, upsamplers, is_final_block):
        super().__init__()
        self.upsamplers = upsamplers
        self.is_final_block = is_final_block

    def forward(self, inputs):
        with cuda_autocast():
            hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
            upsample_size = res_hidden_states[-1].shape[2:] if not self.is_final_block and forward_upsample_size else None
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class UnetDownBlockLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        with cuda_autocast():
            sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs
            if getattr(self.block, "has_cross_attention", False):
                sample, res_samples = self.block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = self.block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
            return make_contiguous(sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size)

    def to_layers(self):
        layers = []
        resnets = self.block.resnets
        attentions = getattr(self.block, "attentions", [None] * len(resnets))
        for resnet, attention in zip(resnets, attentions):
            layers.append(DownBlockInnerLayer(resnet, attention))
        if self.block.downsamplers is not None:
            layers.append(DownsamplerLayer(self.block.downsamplers))
        return layers


class UnetMidBlockLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        with cuda_autocast():
            sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs
            if getattr(self.block, "has_cross_attention", False):
                sample = self.block(sample, emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample = self.block(sample, emb)
            return make_contiguous(sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size)

    def to_layers(self):
        layers = []
        resnets = self.block.resnets
        attentions = self.block.attentions
        layers.append(MidBlockInnerLayer(resnets[0], None))
        for attn, resnet in zip(attentions, resnets[1:]):
            layers.append(MidBlockInnerLayer(resnet, attn))
        return layers


class UnetUpBlockLayer(nn.Module):
    def __init__(self, block, is_final_block):
        super().__init__()
        self.block = block
        self.is_final_block = is_final_block

    def forward(self, inputs):
        with cuda_autocast():
            sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs
            res_samples = down_block_res_samples[-len(self.block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(self.block.resnets)]
            upsample_size = down_block_res_samples[-1].shape[2:] if not self.is_final_block and forward_upsample_size else None
            if getattr(self.block, "has_cross_attention", False):
                sample = self.block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, encoder_hidden_states=encoder_hidden_states, upsample_size=upsample_size)
            else:
                sample = self.block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size)
            return make_contiguous(sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size)

    def to_layers(self):
        layers = []
        resnets = self.block.resnets
        attentions = getattr(self.block, "attentions", [None] * len(resnets))
        for resnet, attention in zip(resnets, attentions):
            layers.append(UpBlockInnerLayer(resnet, attention))
        if self.block.upsamplers is not None:
            layers.append(UpsamplerLayer(self.block.upsamplers, self.is_final_block))
        return layers


class FinalLayer(nn.Module):
    def __init__(self, unet, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.conv_norm_out = unet.conv_norm_out
        self.conv_act = unet.conv_act
        self.conv_out = unet.conv_out

    def forward(self, inputs):
        with cuda_autocast():
            sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            return self.conv_out(sample), timesteps
