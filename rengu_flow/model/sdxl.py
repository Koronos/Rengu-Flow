"""SDXL pipeline model: diffusers StableDiffusionXL with to_layers() for DeepSpeed pipeline."""

import re
from pathlib import Path

import diffusers
import torch
from torch import nn
import torch.nn.functional as F
import safetensors

from rengu_flow.data.preprocess_media import PreprocessMediaFile
from rengu_flow.model.base import BasePipeline, make_contiguous
from rengu_flow.registry.models import register_model
from rengu_flow.utils.common import cuda_autocast, is_main_process
from rengu_flow.utils.logging import logger
from rengu_flow.utils.save_io import atomic_save_safetensors
from rengu_flow.utils.diffusers_tf5_compat import apply_diffusers_transformers_v5_single_file_patch

# Optional: import network adapters (lora_sdxl always; lokr_vendored may use LyCORIS or vendored)
from rengu_flow import networks as networks_module


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
    # The SNR table (min-SNR / debiased weighting) is derived from alphas_cumprod: a table built
    # before the rescale would keep the original (non-zero) terminal SNR.
    if hasattr(noise_scheduler, "all_snr"):
        del noise_scheduler.all_snr
        prepare_scheduler_for_custom_training(noise_scheduler)


def build_noise_scheduler(v_pred: bool = False, zero_terminal_snr: bool | None = None):
    """Training scheduler, also reused by the previews (``pipe(...)`` samples with it).

    ``v_pred`` selects the scheduler's prediction type so previews decode the UNet output as a
    velocity (not epsilon). Zero-terminal-SNR rescales the betas BEFORE the SNR table is built
    and switches sampling to trailing timestep spacing, which that schedule needs.
    """
    if zero_terminal_snr is None:
        zero_terminal_snr = bool(v_pred)
    scheduler = diffusers.DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        prediction_type="v_prediction" if v_pred else "epsilon",
        timestep_spacing="trailing" if zero_terminal_snr else "leading",
    )
    if zero_terminal_snr:
        fix_noise_scheduler_betas_for_zero_terminal_snr(scheduler)
    prepare_scheduler_for_custom_training(scheduler)
    return scheduler


# --- CLIP tokenization (shared by caching, the live text-encoder path and InitialLayer) -------
# Every 75-token chunk is [BOS] + ids + [EOS] + pad up to the CLIP context (77), exactly what
# diffusers / ComfyUI / A1111 / kohya feed the encoders at inference. Training used to feed
# variable-length sequences ([BOS] ids [EOS], no pad), a distribution the model never sees later.
CLIP_CONTEXT_LENGTH = 77
CLIP_CHUNK_TOKENS = CLIP_CONTEXT_LENGTH - 2


def _clip_pad_id(tokenizer) -> int:
    pad = tokenizer.pad_token_id
    return tokenizer.eos_token_id if pad is None else pad


def _clip_chunk_ids(ids: list[int], tokenizer) -> list[list[int]]:
    """Split raw token ids into 75-token chunks, each wrapped/padded to 77 tokens (>= 1 chunk)."""
    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, _clip_pad_id(tokenizer)
    pieces = [ids[i : i + CLIP_CHUNK_TOKENS] for i in range(0, len(ids), CLIP_CHUNK_TOKENS)] or [[]]
    return [[bos, *piece, eos, *([pad] * (CLIP_CHUNK_TOKENS - len(piece)))] for piece in pieces]


def tokenize_clip_chunks(prompt, tokenizer) -> torch.Tensor:
    """(batch, n_chunks * 77) int64 ids. ``n_chunks`` is the longest caption's chunk count;
    shorter captions are filled with empty chunks ([BOS][EOS][pad...])."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    token_lists = tokenizer(prompts, add_special_tokens=False, truncation=False)["input_ids"]
    per_caption = [_clip_chunk_ids(list(ids), tokenizer) for ids in token_lists]
    n_chunks = max(len(c) for c in per_caption)
    empty = _clip_chunk_ids([], tokenizer)[0]
    rows = [
        [tok for chunk in (chunks + [empty] * (n_chunks - len(chunks))) for tok in chunk]
        for chunks in per_caption
    ]
    return torch.tensor(rows, dtype=torch.int64)


def encode_clip_chunks(input_ids, text_encoder, clip_skip=None, return_pooled_prompt_embeds=False):
    """Run (batch, n_chunks * 77) ids through a CLIP encoder chunk by chunk and concatenate the
    chunks' penultimate (or ``clip_skip``) hidden states along the sequence axis. The pooled
    output comes from the first chunk."""
    te_device = next(text_encoder.parameters()).device
    input_ids = input_ids.to(te_device)
    layer = -2 if clip_skip is None else -(clip_skip + 2)
    embed_chunks = []
    pooled_prompt_embeds = None
    for i, chunk in enumerate(torch.split(input_ids, CLIP_CONTEXT_LENGTH, dim=-1)):
        out = text_encoder(chunk, output_hidden_states=True)
        if i == 0 and return_pooled_prompt_embeds:
            pooled_prompt_embeds = out[0]
        embed_chunks.append(out.hidden_states[layer])
    embeds = torch.cat(embed_chunks, dim=1)
    if return_pooled_prompt_embeds:
        return embeds, pooled_prompt_embeds
    return embeds


def pad_cached_text_embeds(embeds, name: str = "prompt_embeds") -> torch.Tensor:
    """Cached per-caption embeddings (a stacked tensor, or a list when the collate found rows of
    different chunk counts) -> one (batch, max_len, dim) tensor. Shorter rows get zero chunks."""
    rows = list(embeds.unbind(0)) if torch.is_tensor(embeds) else list(embeds)
    for r in rows:
        if r.shape[0] % CLIP_CONTEXT_LENGTH != 0:
            raise ValueError(
                f"Cached {name} has length {r.shape[0]}, not a multiple of {CLIP_CONTEXT_LENGTH}: the text "
                "cache predates the padded-to-77 CLIP encoding. Rebuild it with --regenerate_text_cache."
            )
    longest = max(r.shape[0] for r in rows)
    padded = [F.pad(r, (0, 0, 0, longest - r.shape[0])) if r.shape[0] < longest else r for r in rows]
    return torch.stack(padded)


from rengu_flow.training.loss_weighting import apply_debiased_estimation, apply_min_snr_weight


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
        self._init_block_swap_state()
        self.v_pred = self.model_config.get("v_pred", False)
        self.min_snr_gamma = self.model_config.get("min_snr_gamma", None)
        self.debiased_estimation_loss = self.model_config.get("debiased_estimation_loss", None)
        # Zero-terminal-SNR betas default on for v-pred (the usual recipe); switchable.
        self.zero_terminal_snr = bool(self.model_config.get("zero_terminal_snr", self.v_pred))
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
        # `_pipeline` and dunder probes must NOT trigger the lazy pipeline load, or a half-built
        # instance recurses forever (diffusers_pipeline -> load_diffusion_model -> `self._pipeline`
        # -> __getattr__). This happens when the model is unpickled in a Windows multiprocess
        # 'spawn' worker (where __init__ never ran, so `_pipeline` is absent). Real diffusers
        # private attrs (e.g. `_execution_device`) still delegate.
        if name == "_pipeline" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
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
        self._pipeline.scheduler = build_noise_scheduler(self.v_pred, self.zero_terminal_snr)
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

    def keep_submodel_on_cpu_after_cache(self, submodel) -> bool:
        # Full-model SDXL writes a complete checkpoint (UNet + VAE + both text encoders), so every
        # submodel's weights must survive on CPU. Adapter runs only emit the adapter — keep just the
        # VAE, which save_model still reads. (Otherwise frozen submodels go to meta to free RAM.)
        if self.config.get("adapter"):
            # Previews re-encode prompts with the text encoders, so they must survive on CPU (meta
            # discards the weights and the preview crashes). Costs ~1.4 GB host RAM, only when on.
            from rengu_flow.utils.preview import previews_configured

            if previews_configured(self.config):
                return True
            return submodel is self.vae
        return True

    def materialize_for_preview(self):
        """Move VAE + text encoders (parked on CPU/meta after caching) onto the GPU for a preview.

        Returns a restore record consumed by ``restore_preview_submodels``. When a submodel is on
        ``meta`` (caching freed it and previews weren't pre-configured — e.g. a "Preview now" click),
        its weights are reloaded once from the checkpoint and then parked on CPU between previews.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = self.diffusers_pipeline
        subs = {"vae": pipe.vae, "text_encoder": pipe.text_encoder, "text_encoder_2": pipe.text_encoder_2}
        on_meta = any(
            next(m.parameters(), torch.empty(0)).device.type == "meta" for m in subs.values()
        )
        if on_meta:
            self._reload_frozen_submodels_from_checkpoint()
            pipe = self.diffusers_pipeline
            subs = {"vae": pipe.vae, "text_encoder": pipe.text_encoder, "text_encoder_2": pipe.text_encoder_2}
        rest: dict = {}
        for key, m in subs.items():
            p = next(m.parameters(), None)
            rest[key] = p.device if p is not None else torch.device("cpu")
            m.to(device)
        # The VAE is upcast to fp32 (upcast_vae, for stable encode) but the UNet emits bf16 latents,
        # and diffusers only re-casts latents when the VAE is fp16 — so decode hits an fp32-bias vs
        # bf16-input mismatch. Run the decode in the compute dtype; restored below.
        compute_dtype = next(pipe.unet.parameters()).dtype
        rest["_vae_dtype"] = next(pipe.vae.parameters()).dtype
        pipe.vae.to(compute_dtype)
        return rest

    def restore_preview_submodels(self, rest: dict) -> None:
        """Park VAE + text encoders back on their pre-preview device (CPU, never meta — so the next
        preview skips the disk reload) and restore the VAE's training dtype."""
        pipe = self.diffusers_pipeline
        vae_dtype = rest.get("_vae_dtype")
        if vae_dtype is not None:
            pipe.vae.to(vae_dtype)
        for key in ("vae", "text_encoder", "text_encoder_2"):
            m = getattr(pipe, key)
            dev = rest.get(key, torch.device("cpu"))
            if getattr(dev, "type", None) == "meta":
                dev = torch.device("cpu")
            m.to(dev)

    def _reload_frozen_submodels_from_checkpoint(self):
        """Reload VAE + text encoder weights from the checkpoint when caching freed them to meta.

        Steals the freshly-loaded frozen submodels into the live pipeline (the live UNet, which
        carries the trained adapter, is left untouched)."""
        if is_main_process():
            logger.info("rengu_flow: reloading SDXL text encoders/VAE for preview...")
        fresh = diffusers.StableDiffusionXLPipeline.from_single_file(
            self.model_config["checkpoint_path"],
            torch_dtype=self.model_config["dtype"],
            add_watermarker=False,
        )
        fresh.upcast_vae()
        for key in ("vae", "text_encoder", "text_encoder_2"):
            getattr(fresh, key).requires_grad_(False)
            setattr(self._pipeline, key, getattr(fresh, key))
        del fresh

    def get_text_encoders(self):
        if not self.cache_text_embeddings:
            return []
        pipe = self.diffusers_pipeline
        return [pipe.text_encoder, pipe.text_encoder_2]

    def configure_adapter(self, adapter_config):
        self.adapter_config = adapter_config
        self.adapter_type = adapter_config["type"]
        # init_from_existing no longer short-circuits: the LoRA is always wrapped through the normal
        # path (adapter dtype, frozen base) and the weights are copied in by load_adapter_weights.
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
            networks_module.lokr_vendored.configure(
                self.unet,
                self.text_encoder,
                self.text_encoder_2,
                adapter_config,
            )
        elif self.adapter_type.startswith("lycoris_"):
            networks_module.lycoris_sdxl.configure(
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
            networks_module.lokr_vendored.save(save_dir, state_dict, self.adapter_config)
        elif adapter_type.startswith("lycoris_"):
            networks_module.lycoris_sdxl.save(save_dir, state_dict, self.adapter_config)
        else:
            raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")

    def load_adapter_weights(self, adapter_path):
        adapter_path = Path(adapter_path)
        files = list(adapter_path.glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"No .safetensors file found in {adapter_path}")
        state = safetensors.torch.load_file(files[0])
        adapter_type = getattr(self, "adapter_type", None) or (self.config.get("adapter") or {}).get("type")
        # Dispatch by the configured type first: lycoris exports reuse lokr_/lora_
        # key fragments, so key-sniffing only decides when no type is configured.
        if adapter_type and adapter_type.startswith("lycoris_"):
            networks_module.lycoris_sdxl.load(self.diffusers_pipeline, adapter_path)
        elif adapter_type == "lokr" or (
            adapter_type is None and any("lokr_" in k for k in state.keys())
            and not networks_module.lycoris_sdxl.looks_like_lycoris_state(state)
        ):
            networks_module.lokr_vendored.load(self, adapter_path)
        elif hasattr(self.unet, "peft_config"):
            # Training with the LoRA already wrapped (init_from_existing): copy into the wrapper.
            networks_module.lora_sdxl.load_into_wrapped(
                self.unet, self.text_encoder, self.text_encoder_2, adapter_path
            )
        else:
            networks_module.lora_sdxl.load(self.diffusers_pipeline, adapter_path)
        self._set_param_original_name()

    def load_and_fuse_adapter(self, path):
        path = Path(path)
        files = list(path.glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"No .safetensors file found in {path}")
        state = safetensors.torch.load_file(files[0])
        adapter_type = getattr(self, "adapter_type", None) or (self.config.get("adapter") or {}).get("type")

        if (adapter_type and adapter_type.startswith("lycoris_")) or (
            adapter_type is None and networks_module.lycoris_sdxl.looks_like_lycoris_state(state)
        ):
            networks_module.lycoris_sdxl.load_and_fuse(self.diffusers_pipeline, path)
            self._set_param_original_name()
        elif adapter_type == "lokr" or any("lokr_" in k for k in state.keys()):
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
            adapter_config = networks_module.lokr_vendored.infer_lokr_config_from_state(state)
            adapter_config["type"] = "lokr"
            existing = self.config.get("adapter") or {}
            if "rank" not in adapter_config and "dim" in existing:
                adapter_config["rank"] = existing["dim"]
            if adapter_config.get("dtype") is None:
                model_dtype = self.model_config.get("dtype", torch.float32)
                adapter_config["dtype"] = model_dtype if isinstance(model_dtype, torch.dtype) else torch.float32
            self.configure_adapter(adapter_config)
        self.load_adapter_weights(path)
        networks_module.lokr_vendored.fuse(self)
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
        # When the text encoders are frozen (freeze_text_encoders) or their embeddings are
        # cached, they are absent from the trained state dict. Source them from the live modules
        # so the exported checkpoint is still a complete SDXL model. DatasetManager.cache keeps
        # them on CPU for full-model SDXL precisely so this read succeeds.
        if not text_enc_dict:
            text_enc_dict = {k: v.detach().cpu() for k, v in self.text_encoder.state_dict().items()}
        if not text_enc_2_dict:
            text_enc_2_dict = {k: v.detach().cpu() for k, v in self.text_encoder_2.state_dict().items()}
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
            # The distribution's mode: a cached latent is reused every epoch, so one frozen
            # random draw would bake that noise in for the whole run.
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.mode()
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
        # One caption at a time: a cached row holds only its own chunks (batch-independent, so the
        # cache stays valid when the batch composition changes). Rows of different chunk counts
        # come back as a list, which the cache stores ragged; prepare_inputs re-pads per batch.
        embeds_out = []
        pooled_list = []
        for caption in captions:
            input_ids = self._get_input_ids([caption], tokenizer)
            embed, pooled = self._encode_prompt_embeds_from_input_ids(
                input_ids, tokenizer, text_encoder, return_pooled_prompt_embeds
            )
            embeds_out.append(embed[0])
            if return_pooled_prompt_embeds:
                pooled_list.append(pooled[0])
        if len({e.shape for e in embeds_out}) == 1:
            prompt_embeds = torch.stack(embeds_out)
        else:
            prompt_embeds = embeds_out
        if return_pooled_prompt_embeds:
            return prompt_embeds, torch.stack(pooled_list)
        return prompt_embeds

    def _encode_prompt_embeds_from_input_ids(
        self, input_ids, tokenizer, text_encoder, return_pooled_prompt_embeds=False
    ):
        out = encode_clip_chunks(input_ids, text_encoder, self.clip_skip, return_pooled_prompt_embeds)
        if return_pooled_prompt_embeds:
            return out
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
            # Captions of different chunk counts (>75 tokens) arrive as lists: pad to the batch max.
            encoder_hidden_states = torch.cat(
                [
                    pad_cached_text_embeds(inputs["prompt_embeds"], "prompt_embeds"),
                    pad_cached_text_embeds(inputs["prompt_embeds_2"], "prompt_embeds_2"),
                ],
                dim=-1,
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
        return tokenize_clip_chunks(prompt, tokenizer)

    def get_block_swap_modules(self) -> list[nn.Module]:
        unet = self.diffusers_pipeline.unet
        modules: list[nn.Module] = list(unet.down_blocks)
        if unet.mid_block is not None:
            modules.append(unet.mid_block)
        modules.extend(unet.up_blocks)
        return modules

    def _block_swap_root_modules(self) -> list:
        # The UNet holds the swappable down/mid/up blocks; the generic _place_for_block_swap puts
        # the rest (conv_in, time/add embeddings, conv_out, …) on the GPU. The hook-based offloader
        # works even though to_layers() flattens each block into several pipeline layers. When text
        # embeddings are cached the encoders are unloaded to meta (not roots); when they are trained
        # (cache_text_embeddings = false) they stay in the graph, so place them on the GPU too.
        roots = [self.diffusers_pipeline.unet]
        if not self.cache_text_embeddings:
            roots += [self.text_encoder, self.text_encoder_2]
        return roots

    def to_layers(self):
        layers = [
            InitialLayer(
                self.diffusers_pipeline,
                cache_text_embeddings=self.cache_text_embeddings,
                clip_skip=self.clip_skip,
            )
        ]
        unet = self.diffusers_pipeline.unet
        offloader = self.offloader
        block_idx = 0
        for block in unet.down_blocks:
            layers.extend(UnetDownBlockLayer(block, block_idx, offloader).to_layers())
            block_idx += 1
        if unet.mid_block is not None:
            layers.extend(UnetMidBlockLayer(unet.mid_block, block_idx, offloader).to_layers())
            block_idx += 1
        for i, block in enumerate(unet.up_blocks):
            layers.extend(
                UnetUpBlockLayer(block, i == len(unet.up_blocks) - 1, block_idx, offloader).to_layers()
            )
            block_idx += 1
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
                from rengu_flow.model.loss_utils import compute_diffusion_loss_per_element

                loss = compute_diffusion_loss_per_element(output, target, self.config)
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean([1, 2, 3])
                if self.min_snr_gamma is not None:
                    loss = apply_min_snr_weight(
                        loss, timesteps, self.scheduler, self.min_snr_gamma, v_prediction=self.v_pred
                    )
                if self.debiased_estimation_loss:
                    loss = apply_debiased_estimation(
                        loss, timesteps, self.scheduler, v_prediction=self.v_pred
                    )
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
        return encode_clip_chunks(input_ids, text_encoder, self.clip_skip, return_pooled_prompt_embeds)


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
    def __init__(self, block, block_idx: int, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    def forward(self, inputs):
        with cuda_autocast():
            self.offloader.wait_for_block(self.block_idx)
            sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs
            if getattr(self.block, "has_cross_attention", False):
                sample, res_samples = self.block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = self.block(hidden_states=sample, temb=emb)
            self.offloader.submit_move_blocks_forward(self.block_idx)
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
    def __init__(self, block, block_idx: int, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    def forward(self, inputs):
        with cuda_autocast():
            self.offloader.wait_for_block(self.block_idx)
            sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs
            if getattr(self.block, "has_cross_attention", False):
                sample = self.block(sample, emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample = self.block(sample, emb)
            self.offloader.submit_move_blocks_forward(self.block_idx)
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
    def __init__(self, block, is_final_block, block_idx: int, offloader):
        super().__init__()
        self.block = block
        self.is_final_block = is_final_block
        self.block_idx = block_idx
        self.offloader = offloader

    def forward(self, inputs):
        with cuda_autocast():
            self.offloader.wait_for_block(self.block_idx)
            sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs
            res_samples = down_block_res_samples[-len(self.block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(self.block.resnets)]
            upsample_size = down_block_res_samples[-1].shape[2:] if not self.is_final_block and forward_upsample_size else None
            if getattr(self.block, "has_cross_attention", False):
                sample = self.block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, encoder_hidden_states=encoder_hidden_states, upsample_size=upsample_size)
            else:
                sample = self.block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size)
            self.offloader.submit_move_blocks_forward(self.block_idx)
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
