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
(:mod:`rengu_flow.networks.lokr_vendored`) is quantization-aware: when a target module exposes a
``base_linear`` callable it routes the base matmul through that (the quantized path) and only
adds the trainable Kronecker delta on top -- so LoKr trains on top of the quantized base.

Only the matmul-heavy block linears are quantized. Embedders, the final layer, all 1-D params,
``KEEP_IN_HIGH_PRECISION`` modules and the ``llm_adapter`` are skipped (they stay in their
loaded precision), mirroring ``load_diffusion_model``'s dtype policy.

MEASURED OUTCOME (Cosmos LoKr, RTX 4080 16GB, 1024px, 2026-06): NEITHER scheme helped, kept here
for other hardware/models. Baselines: full-ckpt 1.82s/7.6GB, SAC 1.74s/9.45GB.
  * 4-bit: with activation_checkpointing=false it OOMs (14.6GB) — checkpointing saves *activation*
    memory, 4-bit only shrinks *weight* memory, so it can't enable dropping AC. With AC on it just
    adds dequant overhead. It's a VRAM-fit lever (run a bigger model), not a speed lever.
  * fp8: ``torch._scaled_mm`` on the 4080 requires the **weight operand to be e4m3** (e5m2 weight
    is rejected; e5m2×e5m2 unsupported), but Cosmos is fp8-sensitive precisely to e4m3 (outliers
    > ±448). Even setting it to e4m3 (and after fixing the missing autograd derivative + the bf16
    row-wise output requirement), it ran at **~2.95s/11.3GB = ~70% SLOWER**: the per-step activation
    quant over ~280 linears + the autograd.Function graph breaks (no compile fusion) dwarf the fp8
    gemm gain. A fused/delayed-scaling fp8 (torchao) might net positive but conflicts with the LoKr
    forward override and still forces e4m3 on Ada. Revisit fp8 on Hopper (e5m2 matmul) or a
    non-fp8-sensitive model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
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


def _is_quantizable_block_linear(
    full_name: str, module: nn.Module, leaf_names=None, skip_substrings=None
) -> bool:
    """True iff *module* is a big frozen block linear we should quantize."""
    if not isinstance(module, nn.Linear):
        return False
    if any(sub in full_name for sub in (skip_substrings or _SKIP_NAME_SUBSTRINGS)):
        return False
    leaf = full_name.rsplit(".", 1)[-1]
    if leaf not in (leaf_names or _QUANT_LEAF_NAMES):
        return False
    # Defensive: only 2-D weights (a Linear always has one, but be explicit).
    return getattr(module, "weight", None) is not None and module.weight.ndim == 2


def _iter_quant_targets(transformer: nn.Module, leaf_names=None, skip_substrings=None):
    """Yield ``(parent_module, child_attr_name, full_name, linear)`` for each target.

    ``leaf_names`` / ``skip_substrings`` override the cosmos-tuned defaults so other
    DiT families (e.g. krea2) can pass their own scope instead of silently matching
    nothing."""
    modules = dict(transformer.named_modules())
    for full_name, module in list(transformer.named_modules()):
        if not _is_quantizable_block_linear(full_name, module, leaf_names, skip_substrings):
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


class _Fp8ScaledMatmul(torch.autograd.Function):
    """fp8 ``_scaled_mm`` forward + a hand-written backward (the op has no native derivative).

    The frozen base weight is pre-quantized to fp8 (``weight_fp8`` [N, K] + per-output-row
    ``weight_scale`` [N]); the activation is quantized row-wise here. Backward only needs the
    input gradient (weight is frozen), computed with the high-precision ``weight_hp`` —
    ``y = x @ Wᵀ`` ⇒ ``grad_x = grad_out @ W`` — a QLoRA-style straight-through (fp8 forward,
    hi-precision backward) that lets gradients flow to the LoKr delta and earlier layers.
    """

    @staticmethod
    def forward(ctx, x2d, weight_fp8, weight_scale, weight_hp, fp8_dtype, out_dtype):
        fp8_max = _fp8_max(fp8_dtype)
        x_amax = x2d.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-12).float()
        x_scale = (x_amax / fp8_max).to(torch.float32)               # [M, 1] per-row
        x_fp8 = (x2d.detach().float() / x_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        out = torch._scaled_mm(
            x_fp8,                                   # [M, K]
            weight_fp8.t(),                          # [K, N]
            scale_a=x_scale,                         # [M, 1]
            scale_b=weight_scale.reshape(1, -1),     # [1, N] per-output-row
            bias=None,
            out_dtype=out_dtype,
        )
        ctx.save_for_backward(weight_hp)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (weight_hp,) = ctx.saved_tensors
        grad_x = grad_out.to(weight_hp.dtype) @ weight_hp            # [M, N] @ [N, K] -> [M, K]
        return grad_x.to(grad_out.dtype), None, None, None, None, None


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
        """fp8 scaled matmul via an autograd.Function (``_scaled_mm`` has no native derivative).

        Row-wise scaling convention (sm89+/Hopper): the lhs ``x_fp8`` [M, K] carries a float32
        per-row scale ``scale_a`` [M, 1]; the rhs (weight, transposed to [K, N]) carries a
        per-column scale ``scale_b`` [1, N] (= the per-output-row weight scale transposed). The
        weight is frozen, so the backward only needs grad wrt the input, computed with the kept
        high-precision weight (``grad_x = grad_out @ W``) — standard QLoRA-style straight-through.
        ``_scaled_mm`` is a CUDA fp8 path; on CPU it raises (caller's smoke catches that).
        """
        # row-wise _scaled_mm only emits bf16/fp16; fp32 activations -> bf16 output (model is bf16).
        out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        out = _Fp8ScaledMatmul.apply(
            x2d, self.weight_fp8, self.weight_scale, self.weight, self.weight_fp8_dtype, out_dtype
        )
        if self.bias is not None:
            out = out + self.bias.to(out_dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.base_linear(x)


def convert_dit_to_fp8_matmul(
    transformer: nn.Module, *, fp8_dtype: torch.dtype, leaf_names=None, skip_substrings=None
) -> int:
    """Replace the frozen DiT's big block linears with :class:`Fp8MatmulLinear` (in place).

    Returns the number of linears converted. The frozen base gains no trainable params.
    """
    count = 0
    for parent, child_attr, _full_name, linear in _iter_quant_targets(
        transformer, leaf_names, skip_substrings
    ):
        new = Fp8MatmulLinear(linear, weight_fp8_dtype=fp8_dtype)
        setattr(parent, child_attr, new)
        count += 1
    return count


# ---------------------------------------------------------------------------
# (A2) fp8 TENSORWISE scaled matmul — the sm89-viable scheme
# ---------------------------------------------------------------------------
#
# The rowwise scheme above measured ~70% slower on Ada (eager per-call quant + the
# CUTLASS rowwise kernel itself running BELOW bf16 speed: 77 vs 91 TFLOPS on krea2
# shapes). Tensorwise scales hit the cuBLASLt kernel instead: measured 198-208 TFLOPS
# on an RTX 4080 — a clean 2x over bf16 — and under torch.compile the activation quant
# fuses, netting 1.49x (bf16 dgrad) / 1.78x (fp8 dgrad) on a full checkpointed krea2
# block (tmp/bench2/fp8_block_bench.py, 2026-07-05). Eager it nets ~nothing; compile
# at BLOCK scope is a prerequisite (whole-module compile graph-breaks it all away).
#
# Storage: the fp8 tensor IS the weight (no hi-precision copy) — 1 byte/param, so a
# 12.9B krea2 base drops 25.6 GB -> 12.9 GB and block swap (if any) moves half the
# bytes. Backward is QLoRA-style straight-through: grad_x from the dequantized (or
# fp8-transposed) weight; the frozen base never needs grad_weight.

_E4M3 = getattr(torch, "float8_e4m3fn", None)


class _Fp8TensorwiseMatmul(torch.autograd.Function):
    """fp8 forward on the fast tensorwise cuBLASLt kernel, with PER-OUTPUT-ROW weight
    scales applied as a fused epilogue.

    The weight is quantized row-normalized (each output row uses the full e4m3 range),
    which cuts quant error to the per-row outlier level — but _scaled_mm's fast kernel
    only takes scalar scales, so the GEMM runs with scale_b=1 and the row scales
    multiply the output afterwards (a pointwise op torch.compile fuses with the
    surrounding eltwise; ~2% cost, vs the rowwise CUTLASS kernel which runs BELOW
    bf16 speed on Ada). dgrad folds the row scales into grad_out before the
    contraction (they live on the contracted dim there)."""

    @staticmethod
    def forward(ctx, x2d, w8, wscale_row, one, grad_mode):
        fp8_max = _fp8_max(w8.dtype)
        xs = (x2d.detach().abs().amax().clamp_min(1e-12).float() / fp8_max).reshape(())
        x8 = (x2d.detach().float() / xs).clamp(-fp8_max, fp8_max).to(w8.dtype)
        out = torch._scaled_mm(
            x8, w8.t(), scale_a=xs, scale_b=one, out_dtype=torch.bfloat16
        )
        out = out * wscale_row.to(out.dtype)  # [N] epilogue, compile-fused
        ctx.save_for_backward(w8, wscale_row, one)
        ctx.grad_mode = grad_mode
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w8, wscale_row, one = ctx.saved_tensors
        if ctx.grad_mode == "fp8":
            fp8_max = _fp8_max(w8.dtype)
            g = grad_out.detach().float() * wscale_row.float()  # fold row scales (contracted dim)
            gs = (g.abs().amax().clamp_min(1e-12) / fp8_max).reshape(())
            g8 = (g / gs).clamp(-fp8_max, fp8_max).to(w8.dtype)
            w8_col = w8.t().contiguous().t()  # [N, K] column-major, transposed on the fly
            grad_x = torch._scaled_mm(
                g8, w8_col, scale_a=gs, scale_b=one, out_dtype=torch.bfloat16
            )
        else:
            grad_x = (grad_out * wscale_row.to(grad_out.dtype)) @ w8.to(torch.bfloat16)
        return grad_x.to(grad_out.dtype), None, None, None, None


class Fp8TensorwiseLinear(nn.Linear):
    """Frozen ``nn.Linear`` stored ONLY as tensorwise-scaled e4m3 (1 byte/param).

    ``weight`` is the fp8 tensor itself (a frozen Parameter, so block swap moves it);
    ``weight_scale`` is a scalar fp32 buffer. Adapters target it through the usual
    ``isinstance(module, nn.Linear)`` walk and compose via ``base_linear`` (LoKr) or by
    wrapping ``forward`` (PEFT LoRA). CPU calls fall back to a dequantized matmul so
    CPU tests/previews still run; ``_scaled_mm`` is CUDA-only.
    """

    def __init__(self, linear: nn.Linear, grad_mode: str = "bf16"):
        super().__init__(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None, device="meta",
        )
        if _E4M3 is None:
            raise RuntimeError("torch.float8_e4m3fn is unavailable in this torch build.")
        if grad_mode not in ("bf16", "fp8"):
            raise ValueError(f"fp8 grad_mode must be 'bf16' or 'fp8', got {grad_mode!r}.")
        self.grad_mode = grad_mode
        # Quantize on the GPU when there is one: the fp32 amax/div over a 6144x16384
        # weight is seconds-vs-minutes CPU work across a 12.9B model (the weight round-
        # trips over PCIe once; the result is parked back on the source device).
        src_device = linear.weight.device
        calc_device = "cuda" if torch.cuda.is_available() else src_device
        w = linear.weight.detach().to(calc_device).float()
        fp8_max = _fp8_max(_E4M3)
        # Per-output-row scales: each row is normalized to the full e4m3 range, so one
        # outlier row no longer sets the quant step for the whole tensor.
        scale = (w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / fp8_max)
        self.weight = nn.Parameter(
            (w / scale).clamp(-fp8_max, fp8_max).to(_E4M3).to(src_device), requires_grad=False
        )
        self.register_buffer(
            "weight_scale", scale.reshape(-1).float().to(src_device), persistent=False
        )
        self.register_buffer("scale_one", torch.ones((), dtype=torch.float32), persistent=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def base_linear(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:  # CPU fallback: dequantized matmul (tests, CPU previews)
            w = self.weight.to(torch.float32) * self.weight_scale[:, None]
            return F.linear(x.float(), w, None if self.bias is None else self.bias.float()).to(x.dtype)
        shp = x.shape
        out = _Fp8TensorwiseMatmul.apply(
            x.reshape(-1, shp[-1]), self.weight, self.weight_scale, self.scale_one, self.grad_mode
        )
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*shp[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.base_linear(x)


def convert_dit_to_fp8_tensorwise(
    transformer: nn.Module, *, grad_mode: str = "bf16", leaf_names=None, skip_substrings=None
) -> int:
    """Replace the frozen DiT's big block linears with :class:`Fp8TensorwiseLinear`.

    Returns the number of linears converted. Unlike the rowwise scheme, the original
    hi-precision weight is DROPPED (halves weight memory vs bf16)."""
    count = 0
    for parent, child_attr, _full_name, linear in _iter_quant_targets(
        transformer, leaf_names, skip_substrings
    ):
        new = Fp8TensorwiseLinear(linear, grad_mode=grad_mode)
        setattr(parent, child_attr, new)
        count += 1
    return count


# ---------------------------------------------------------------------------
# (B) 4-bit NF4 (bitsandbytes)
# ---------------------------------------------------------------------------

def convert_dit_to_4bit(
    transformer: nn.Module,
    *,
    compute_dtype: Optional[torch.dtype] = None,
    leaf_names=None,
    skip_substrings=None,
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
    for parent, child_attr, _full_name, linear in _iter_quant_targets(
        transformer, leaf_names, skip_substrings
    ):
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
        if torch.cuda.is_available():
            # bnb quantizes inside Params4bit.to('cuda'). Block swap moves raw `p.data`
            # (bypassing bnb's override), so a deferred-quantization param would reach the
            # kernel unquantized and without quant_state (bitsandbytes AssertionError at
            # step 1). Quantize eagerly — one weight on the GPU at a time — then park the
            # packed uint8 data back on CPU; the (tiny) quant_state tensors stay on the
            # GPU where the forward kernel needs them. The no-swap path is unaffected: a
            # later model.to('cuda') just moves the already-quantized data.
            new.weight = new.weight.to(torch.device("cuda", torch.cuda.current_device()))
            new.weight.data = new.weight.data.cpu()
        setattr(parent, child_attr, new)
        count += 1
    return count


def base_linear_of(module: nn.Module):
    """Return a callable doing the (quantized) base matmul for *module*, or None.

    Used by the vendored LoKr forward so it can add its trainable delta on top of the
    quantized base path instead of recomputing ``F.linear(x, module.weight)`` (which would
    bypass fp8 / break for 4-bit packed weights).
    """
    if isinstance(module, (Fp8MatmulLinear, Fp8TensorwiseLinear)):
        return module.base_linear
    # bitsandbytes Linear4bit: its own forward IS the dequant+matmul base path.
    try:
        import bitsandbytes as bnb

        if isinstance(module, bnb.nn.Linear4bit):
            return lambda x, m=module: bnb.nn.Linear4bit.forward(m, x)
    except Exception:
        pass
    return None
