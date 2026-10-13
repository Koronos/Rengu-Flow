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

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
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


class TreadSettings(NamedTuple):
    """Validated ``[tread]`` table. ``route`` is the resolved inclusive ``(start, end)`` block
    window, None when the block count is unknown (config time without a registered count)."""

    drop_ratio: float
    disable_after_frac: float
    route: tuple[int, int] | None


def parse_tread(config: dict[str, Any], n_blocks: int) -> TreadSettings | None:
    """Parse and validate the ``[tread]`` table of ``config`` (None when absent/empty).

    Torch-free: shared by config validation (``registry/model_config_rules``, with the model's
    registered block count, 0 = unknown, route range unchecked) and the pipeline's
    ``to_layers`` (with the real block count). Raises ``ConfigValidationError``.
    """
    from rengu_flow.config.validation import ConfigValidationError

    tread = config.get("tread")
    if not tread:
        return None
    if not isinstance(tread, dict):
        raise ConfigValidationError("[tread] must be a table.")
    if tread.get("drop_ratio") is None:
        raise ConfigValidationError(
            "[tread] needs drop_ratio (fraction of image tokens routed around the middle "
            "blocks, e.g. 0.5); remove the [tread] table to turn routing off."
        )
    try:
        drop_ratio = float(tread["drop_ratio"])
        disable_after_frac = float(tread.get("disable_after_frac", 1.0))
        start_block = int(tread.get("start_block", 2))
        end_block = int(tread.get("end_block", -3))
    except (TypeError, ValueError) as e:
        raise ConfigValidationError(f"[tread] has a non-numeric value: {e}") from e
    if not 0.0 < drop_ratio < 1.0:
        raise ConfigValidationError(f"tread.drop_ratio must be in (0, 1), got {drop_ratio}.")
    if not 0.0 < disable_after_frac <= 1.0:
        raise ConfigValidationError(
            f"tread.disable_after_frac must be in (0, 1], got {disable_after_frac}."
        )
    route = None
    if n_blocks:
        try:
            route = resolve_route(n_blocks, start_block, end_block)
        except ValueError as e:
            raise ConfigValidationError(str(e)) from e
    return TreadSettings(drop_ratio, disable_after_frac, route)


def sample_keep_index(
    text_len: int, image_len: int, drop_ratio: float, device: torch.device
) -> torch.Tensor:
    """Batch-shared keep index over the ``[text, image]`` concat sequence.

    Drops ``round(image_len * drop_ratio)`` random image tokens; text tokens are always
    kept. Returns sorted long indices into the full sequence (order-preserving), length
    ``text_len + image_len - dropped``.
    """
    import torch

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
