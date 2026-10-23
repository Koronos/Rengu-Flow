"""MOSS-Image DiT (single-stream, ``[image tokens ; text tokens]``), ported for training.

Adapted from the official ``MOSSImageTransformer`` in ``pipeline_moss_image.py`` of the
``OpenMOSS-Team/MOSS-Image-Base`` Hugging Face repository (Apache-2.0, revision pinned in
``NOTICE.md``). Module and parameter names are the official checkpoint's, so release files load
verbatim and exports keep the official layout (what the adapter exporters and ComfyUI match on).

What differs from the reference, and why:

* ``forward`` is the reference's per-sample loop (each sample at its true text length, no text
  padding anywhere), kept for equivalence tests and as the semantic definition.
* The training path is **batched**: ``embed`` / ``Block`` / ``final`` run a right-padded text batch
  with per-sample RoPE positions (image tokens get ``f = text_len_i``) and a key mask in both the
  text-refiner blocks and the joint blocks, so every sample gets exactly the reference result.
* The compute dtype of activations is read from a norm weight (never from a ``proj`` weight: a
  frozen fp8 / NF4 base has no usable ``weight.dtype``); with bf16 or fp32 weights it equals the
  reference's ``proj.q.weight.dtype``.
"""

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def validate_skip_layers(layers, depth, name="skip_layers") -> frozenset:
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth}).")
    return frozenset(layers)


def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(q, k, v, rope, heads, key_valid=None):
    """SDPA over ``(B, S, heads * head_dim)``; ``key_valid`` ``(B, S)`` bool masks padded keys."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    mask = None if key_valid is None else key_valid[:, None, None, :]
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask).transpose(1, 2).flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, key_valid=None):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads, key_valid))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, key_valid=None):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, key_valid)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, key_valid=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.norm1.weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](attention(q, k, self.proj["v"](y), rope, self.heads, key_valid)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


class RoPE(nn.Module):
    """Four-axis ``(r, f, h, w)`` rotary table; ``bands`` is a derived fp32 constant that stays in the
    state dict for checkpoint compatibility."""

    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .to(bf16) would round the frequencies; they are constants, rebuilt in fp32.
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)

    def _ids(self, height, width, device):
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.zeros_like(h), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        return image

    def _table(self, ids):
        phase = ids.unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)

    def forward(self, height, width, text_length, device):
        """The reference table ``(1, h*w + text_length, 128)`` of one sample."""
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        image = self._ids(height, width, device)
        image[:, 1] = text_length
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        return self._table(torch.cat((image, text), 0)).unsqueeze(0)

    def batched(self, height, width, lengths, text_pad_length, device):
        """``(B, h*w + text_pad_length, 128)``: row ``i`` is the reference table of a sample whose text
        has ``lengths[i]`` tokens (image ``f = lengths[i]``); the text rows ``>= lengths[i]`` are
        padding (their key is masked)."""
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        batch = len(lengths)
        image = self._ids(height, width, device).unsqueeze(0).repeat(batch, 1, 1)
        image[:, :, 1] = torch.tensor([float(n) for n in lengths], device=device, dtype=torch.float32)[:, None]
        text = torch.zeros(text_pad_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_pad_length, device=device)
        return self._table(torch.cat((image, text.unsqueeze(0).expand(batch, -1, -1)), 1))


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    """The MOSS-Image DiT with the official state-dict names (see the module docstring)."""

    @register_to_config
    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    # ---- pieces shared by the reference loop, the batched training path and the previews ------

    def condition(self, timestep, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(modulation (B, 6, dim), head modulation (B, 2, dim))`` of the model time input
        ``timestep`` (``1000 * sigma``, scalar or ``(B,)``), fp32 sinusoid as the reference."""
        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        return mod, head

    def embed_video(self, hidden_states):
        video = hidden_states.flatten(2).transpose(1, 2)
        return self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())

    def embed(self, hidden_states, timestep, context, text_mask=None):
        """Batched embedding of a right-padded text batch: ``(x (B, h*w + L, dim), modulation, head,
        rope (B, h*w + L, 128), key_valid (B, h*w + L) bool | None)``. ``text_mask`` ``(B, L)``;
        ``None`` or all-valid means no padding (no key mask)."""
        batch, _, height, width = hidden_states.shape
        tokens = height * width
        text_len = context.shape[1]
        if text_mask is not None and bool(text_mask.all()):
            text_mask = None
        lengths = [text_len] * batch if text_mask is None else [int(n) for n in text_mask.sum(1).tolist()]
        if min(lengths) == 0:
            raise ValueError("Each sample needs at least one valid text token.")
        mod, head = self.condition(timestep, batch, hidden_states.device)
        video = self.embed_video(hidden_states)
        rope = self.rope.batched(height, width, lengths, text_len, hidden_states.device).to(video.dtype)
        text_valid = None if text_mask is None else text_mask.to(hidden_states.device).bool()
        text = self.text_embedder(context, rope[:, tokens:], text_valid)
        key_valid = None
        if text_valid is not None:
            key_valid = torch.cat((torch.ones(batch, tokens, dtype=torch.bool, device=text_valid.device), text_valid), 1)
        return torch.cat((video, text), 1), mod, head, rope, key_valid

    def final(self, x, head, height: int, width: int):
        """Velocity ``(B, latent_channels, h, w)`` from the stream ``x`` (image tokens first)."""
        x = self.final_norm_video(x[:, : height * width].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(x.shape[0], -1, height, width)

    # ---- the reference forward (per sample, true text length) ----------------------------------

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=True,
        skip_layers=None,
    ):
        skip_layers = validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        mod, head = self.condition(timestep, batch, hidden_states.device)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            video = self.embed_video(hidden_states[index : index + 1])
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index : index + 1])
            outputs.append(self.final(x, head[index : index + 1], height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


def detect_config(state_dict: dict, rope_base: float = 10000.0) -> dict:
    """The architecture of a (split-qkv) official state dict: dims are read from the weights, never
    hardcoded, so every released variant loads."""
    dim, latent_channels = state_dict["video_embedder.weight"].shape
    head_dim = state_dict["layers.0.q_norm.weight"].shape[0]
    layer_ids = {int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")}
    if layer_ids != set(range(len(layer_ids))) or not layer_ids:
        raise ValueError("the checkpoint's layers.N indices are not contiguous from 0")
    return {
        "dim": int(dim),
        "ffn_dim": int(state_dict["layers.0.proj.ffn_down.weight"].shape[1]),
        "num_heads": int(dim) // int(head_dim),
        "num_layers": len(layer_ids),
        "text_dim": int(state_dict["text_embedder.proj.weight"].shape[1]),
        "freq_dim": int(state_dict["time_embedding.0.weight"].shape[1]),
        "latent_channels": int(latent_channels),
        "rope_base": float(rope_base),
    }
