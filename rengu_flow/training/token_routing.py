"""TREAD-style token routing: train middle blocks on a random subset of image tokens.

Training-only FLOP reduction for long-sequence DiTs (arXiv 2501.04765): a batch-shared
random subset of *image* tokens (text is always kept) skips the blocks inside the route
window; at the window's end the processed tokens are scattered back over the pre-route
hidden states (identity bypass), so the output stays full-length, the loss covers every
token, and dropped tokens still carry gradient to earlier blocks through the skip.
Inference and eval are never routed (callers gate on ``training and grad_enabled``).

Batch-shared selection (one permutation for the whole batch) keeps rotary embeddings and
key-padding masks exact under routing: both are index-sliced with the same keep index.
Clean-room implementation of the published mechanism (MAE-style shuffle-and-slice).
"""

from __future__ import annotations

import torch


def resolve_route(num_blocks: int, start_block: int, end_block: int) -> tuple[int, int]:
    """Normalize (possibly negative) route block indices to ``0 <= start < end < num_blocks``.

    The route window is inclusive: blocks in ``[start, end]`` run on the reduced sequence.
    """
    start = start_block if start_block >= 0 else num_blocks + start_block
    end = end_block if end_block >= 0 else num_blocks + end_block
    if not (0 < start < end < num_blocks - 1):
        raise ValueError(
            f"tread route [{start_block}, {end_block}] resolves to [{start}, {end}] on "
            f"{num_blocks} blocks; need 0 < start < end < {num_blocks - 1} (keep the first "
            "and last block unrouted)."
        )
    return start, end


def sample_keep_index(
    text_len: int, image_len: int, drop_ratio: float, device: torch.device
) -> torch.Tensor:
    """Batch-shared keep index over the ``[text, image]`` concat sequence.

    Drops ``round(image_len * drop_ratio)`` random image tokens; text tokens are always
    kept. Returns sorted long indices into the full sequence (order-preserving), length
    ``text_len + image_len - dropped``.
    """
    n_drop = int(image_len * drop_ratio)
    if n_drop <= 0:
        return torch.arange(text_len + image_len, device=device)
    keep_img = torch.randperm(image_len, device=device)[: image_len - n_drop]
    keep_img, _ = torch.sort(keep_img)
    return torch.cat(
        [torch.arange(text_len, device=device), keep_img + text_len]
    )


def route_start(hidden: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    """(B, S, D) -> (B, K, D): keep only ``keep_idx`` tokens."""
    return hidden.index_select(1, keep_idx)


def route_end(
    routed: torch.Tensor, full: torch.Tensor, keep_idx: torch.Tensor
) -> torch.Tensor:
    """Scatter processed tokens back over the pre-route sequence (identity bypass).

    Differentiable on both inputs: kept tokens' grad flows through ``routed``, dropped
    tokens' grad flows through ``full`` straight to the blocks before the window.
    """
    return full.index_copy(1, keep_idx, routed.to(full.dtype))
