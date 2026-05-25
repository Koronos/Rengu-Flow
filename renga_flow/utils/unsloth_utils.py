# Unsloth-style activation checkpointing (offload to CPU). Optional dependency.
# Logic aligned with diffusion-pipe utils/unsloth_utils.

from __future__ import annotations

import torch
from deepspeed.runtime.activation_checkpointing.checkpointing import detach_variable


class _UnslothOffloadedGradientCheckpointer(torch.autograd.Function):
    """Saves VRAM by offloading activations to RAM; uses non-blocking moves."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        args = detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)
        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(inp.grad for inp in inputs)


@torch._disable_dynamo
def unsloth_checkpoint(function, *args):
    """Activation checkpoint that offloads to CPU (Unsloth-style)."""
    return _UnslothOffloadedGradientCheckpointer.apply(function, *args)
