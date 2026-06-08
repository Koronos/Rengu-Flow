"""In-place quantization of a frozen Cosmos DiT's matmul linears.

Two alternative, mutually-exclusive schemes, both default-off and both designed to leave
the **adapter (LoKr / LoRA)** trainable on top of the (now quantized) frozen base:

  (A) fp8 scaled matmul  -- :func:`convert_dit_to_fp8_matmul`
      The frozen weight of each big block linear is quantized **once** to fp8 (default
      ``e5m2`` -- Cosmos is fp8-sensitive) with a per-output-row scale kept as a buffer.
      The activation is quantized dynamically per call and the product runs through
      ``torch._scaled_mm`` accumulating into bf16. No new trainable parameters.

  (B) 4-bit NF4 (bitsandbytes) -- :func:`convert_dit_to_4bit`
      Each big block linear is replaced by ``bitsandbytes.nn.Linear4bit`` (nf4, bf16
      compute, uint8 storage) initialised from the original weight. This is the standard
      QLoRA base; it frees VRAM so the caller can turn activation checkpointing off.

Both replacements **subclass / mimic** ``nn.Linear`` so the lycoris/vendored LoKr targeting
(``isinstance(module, nn.Linear)``) still matches them. The vendored LoKr forward
(:mod:`rengu_flow.networks.lokr_sdxl`) is quantization-aware: when a target module exposes a
``base_linear`` callable it routes the base matmul through that (the quantized path) and only
adds the trainable Kronecker delta on top -- so LoKr trains on top of the quantized base.

Only the matmul-heavy block linears are quantized. Embedders, the final layer, all 1-D params,
``KEEP_IN_HIGH_PRECISION`` modules and the ``llm_adapter`` are skipped (they stay in their
loaded precision), mirroring ``load_diffusion_model``'s dtype policy.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

# Block linears we quantize. These are the matmul-heavy frozen linears inside each
# transformer Block (see cosmos_predict2/dit.py): attention projections + FFN.
_QUANT_LEAF_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2"}
)

# Substrings of the *full* module name that must never be quantized (mirror the
# KEEP_IN_HIGH_PRECISION / llm_adapter / embedder policy of load_diffusion_model).
_SKIP_NAME_SUBSTRINGS = (
    "x_embedder",
    "t_embedder",
    "t_embedding_norm",
    "final_layer",
    "llm_adapter",
    "adaln_modulation",
    "pos_embedder",
)


def _is_quantizable_block_linear(full_name: str, module: nn.Module) -> bool:
    """True iff *module* is a big frozen block linear we should quantize."""
    if not isinstance(module, nn.Linear):
        return False
    if any(sub in full_name for sub in _SKIP_NAME_SUBSTRINGS):
        return False
    leaf = full_name.rsplit(".", 1)[-1]
    if leaf not in _QUANT_LEAF_NAMES:
        return False
    # Defensive: only 2-D weights (a Linear always has one, but be explicit).
    return getattr(module, "weight", None) is not None and module.weight.ndim == 2


def _iter_quant_targets(transformer: nn.Module):
    """Yield ``(parent_module, child_attr_name, full_name, linear)`` for each target."""
    modules = dict(transformer.named_modules())
    for full_name, module in list(transformer.named_modules()):
        if not _is_quantizable_block_linear(full_name, module):
            continue
        parent_name, _, child_attr = full_name.rpartition(".")
        parent = modules[parent_name] if parent_name else transformer
        yield parent, child_attr, full_name, module


# ---------------------------------------------------------------------------
# (A) fp8 scaled matmul
# ---------------------------------------------------------------------------

_FP8_DTYPES = {
    "e5m2": getattr(torch, "float8_e5m2", None),
    "e4m3": getattr(torch, "float8_e4m3fn", None),
    "float8_e5m2": getattr(torch, "float8_e5m2", None),
    "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
}


def resolve_fp8_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    dt = _FP8_DTYPES.get(key)
    if dt is None:
        raise ValueError(
            f"Unsupported fp8_matmul_dtype {name!r}; use 'e5m2' (default) or 'e4m3'."
        )
    return dt


def _fp8_max(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).max)


class Fp8MatmulLinear(nn.Linear):
    """A frozen ``nn.Linear`` whose forward runs an fp8 scaled matmul.

    The weight is quantized once at construction to ``weight_fp8_dtype`` with a per-output-row
    scale; the high-precision ``weight``/``bias`` are kept (frozen) so adapters that read
    ``module.weight`` (LoKr) still see a real tensor. ``base_linear`` is the quantized matmul
    the LoKr forward composes its delta on top of.
    """

    def __init__(self, linear: nn.Linear, weight_fp8_dtype: torch.dtype):
        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device="meta",
        )
        self.weight_fp8_dtype = weight_fp8_dtype
        # Keep the original (frozen) hi-precision weight/bias as real tensors.
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._quantize_weight()

    @torch.no_grad()
    def _quantize_weight(self) -> None:
        w = self.weight.detach().float()  # [out, in]
        amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)  # per-row
        fp8_max = _fp8_max(self.weight_fp8_dtype)
        scale = amax / fp8_max  # hi = fp8 * scale
        w_fp8 = (w / scale).clamp(-fp8_max, fp8_max).to(self.weight_fp8_dtype)
        # Buffers (not params): never enter the optimizer; move with .to(device).
        self.register_buffer("weight_fp8", w_fp8, persistent=False)
        self.register_buffer("weight_scale", scale.to(torch.float32), persistent=False)

    def base_linear(self, x: torch.Tensor) -> torch.Tensor:
        """fp8 scaled matmul: dynamic row-wise activation quant -> ``_scaled_mm`` -> compute dtype.

        Row-wise scaling convention (sm89+/Hopper): the lhs ``x_fp8`` [M, K] carries a float32
        per-row scale ``scale_a`` [M, 1]; the rhs (weight, transposed to [K, N]) carries a
        per-column scale ``scale_b`` [1, N], which is exactly the per-output-row weight scale
        from :meth:`_quantize_weight` transposed. ``_scaled_mm`` for fp8 is a CUDA path; on CPU
        this raises (the caller's smoke catches and reports that).
        """
        out_dtype = x.dtype if x.is_floating_point() else torch.bfloat16
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x_fp8_dtype = self.weight_fp8_dtype
        fp8_max = _fp8_max(x_fp8_dtype)
        x_amax = x2d.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-12).float()
        x_scale = (x_amax / fp8_max).to(torch.float32)  # [M, 1]
        x_fp8 = (x2d.float() / x_scale).clamp(-fp8_max, fp8_max).to(x_fp8_dtype)
        out = torch._scaled_mm(
            x_fp8,                               # [M, K]
            self.weight_fp8.t(),                 # [K, N], column-major view of [N, K]
            scale_a=x_scale,                     # [M, 1] per-row
            scale_b=self.weight_scale.reshape(1, -1),  # [1, N] per-column (= per-output-row)
            bias=None,
            out_dtype=out_dtype,
        )
        if self.bias is not None:
            out = out + self.bias.to(out_dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.base_linear(x)


def convert_dit_to_fp8_matmul(
    transformer: nn.Module, *, fp8_dtype: torch.dtype
) -> int:
    """Replace the frozen DiT's big block linears with :class:`Fp8MatmulLinear` (in place).

    Returns the number of linears converted. The frozen base gains no trainable params.
    """
    count = 0
    for parent, child_attr, _full_name, linear in _iter_quant_targets(transformer):
        new = Fp8MatmulLinear(linear, weight_fp8_dtype=fp8_dtype)
        setattr(parent, child_attr, new)
        count += 1
    return count


# ---------------------------------------------------------------------------
# (B) 4-bit NF4 (bitsandbytes)
# ---------------------------------------------------------------------------

def convert_dit_to_4bit(
    transformer: nn.Module, *, compute_dtype: Optional[torch.dtype] = None
) -> int:
    """Replace the frozen DiT's big block linears with ``bnb.nn.Linear4bit`` (in place).

    NF4 quant, bf16 compute, uint8 storage. The original weight is loaded into the
    ``Params4bit`` so the quantization happens from the resident hi-precision weight.
    Returns the number of linears converted. Requires bitsandbytes; the 4-bit *quantize*
    and matmul kernels are GPU-only at runtime (constructing on CPU is fine).
    """
    import bitsandbytes as bnb

    if compute_dtype is None:
        compute_dtype = torch.bfloat16
    count = 0
    for parent, child_attr, _full_name, linear in _iter_quant_targets(transformer):
        has_bias = linear.bias is not None
        new = bnb.nn.Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,
            quant_type="nf4",
            quant_storage=torch.uint8,
            device="meta",
        )
        # Re-create Params4bit from the resident weight so quantization runs on real data.
        weight = linear.weight.detach()
        new.weight = bnb.nn.Params4bit(
            weight,
            requires_grad=False,
            quant_type="nf4",
            quant_storage=torch.uint8,
        )
        if has_bias:
            new.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        new.weight.requires_grad_(False)
        setattr(parent, child_attr, new)
        count += 1
    return count


def base_linear_of(module: nn.Module):
    """Return a callable doing the (quantized) base matmul for *module*, or None.

    Used by the vendored LoKr forward so it can add its trainable delta on top of the
    quantized base path instead of recomputing ``F.linear(x, module.weight)`` (which would
    bypass fp8 / break for 4-bit packed weights).
    """
    if isinstance(module, Fp8MatmulLinear):
        return module.base_linear
    # bitsandbytes Linear4bit: its own forward IS the dequant+matmul base path.
    try:
        import bitsandbytes as bnb

        if isinstance(module, bnb.nn.Linear4bit):
            return lambda x, m=module: bnb.nn.Linear4bit.forward(m, x)
    except Exception:
        pass
    return None
