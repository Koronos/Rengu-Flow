"""Frozen-base quantization knobs shared by the DiT pipelines (krea2, qwen_image21).

``model.transformer_fp8_matmul`` (tensorwise e4m3, the sm89-viable scheme) and
``model.transformer_4bit`` (NF4 via bitsandbytes) are mutually exclusive (enforced in
``config.defaults``). Each model passes its own scope — which linears are big per-block
matmuls worth quantizing and which small/delicate modules stay in compute dtype.
"""

from __future__ import annotations

import torch

from rengu_flow.utils.common import is_main_process


def quantize_frozen_dit(
    transformer: torch.nn.Module,
    model_config: dict,
    *,
    leaf_names: frozenset[str],
    skip_substrings: tuple[str, ...],
    label: str,
) -> int:
    """Quantize the frozen DiT's matmul linears per ``model_config``; returns how many were
    converted (0 when both knobs are off). The base stays frozen; the quantization-aware
    ``lokr`` adapter composes on top."""
    fp8_matmul = bool(model_config.get("transformer_fp8_matmul", False))
    four_bit = bool(model_config.get("transformer_4bit", False))
    if not fp8_matmul and not four_bit:
        return 0

    from rengu_flow.training import quantize_dit

    scope = {"leaf_names": leaf_names, "skip_substrings": skip_substrings}
    if four_bit:
        n = quantize_dit.convert_dit_to_4bit(transformer, compute_dtype=torch.bfloat16, **scope)
        if is_main_process():
            print(f"rengu_flow: quantized {n} frozen {label} DiT linears to 4-bit NF4 (bnb).")
        return n
    # Tensorwise e4m3 (the sm89-viable scheme): 2x GEMM throughput under block-scope
    # compile AND 1 byte/param storage (no hi-precision copy).
    grad_mode = str(model_config.get("fp8_grad_mode", "bf16"))
    n = quantize_dit.convert_dit_to_fp8_tensorwise(transformer, grad_mode=grad_mode, **scope)
    if is_main_process():
        print(
            f"rengu_flow: converted {n} frozen {label} DiT linears to fp8 tensorwise "
            f"matmul (grad_mode={grad_mode})."
        )
    return n
