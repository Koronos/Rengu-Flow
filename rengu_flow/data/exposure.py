"""Estimate how many times each image is trained, per resolution, under a resolution schedule
and the dataset controls (num_repeats / subsample_ratio / max_images).

Pure functions so the estimate is unit-testable; the trainer prints a report at startup
(rank 0) so a user can size epochs/stages to a target exposure instead of padding by guess.

Model (single-resolution batches, the iteration cycling the active pool proportionally):
  views_at_R_in_stage = stage_steps * batch_size * (entries_R / entries_active)
  exposure_per_image_at_R = views_at_R / distinct_images_R
where ``entries`` counts iteration-order rows (so it already includes ``num_repeats`` and any
per-epoch cap), and ``distinct`` counts unique images. With a uniform ``num_repeats`` the repeat
factor cancels and exposure_R = sum over stages containing R of ``stage_steps * batch / distinct_active``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def schedule_stage_spans(
    stage_resolutions: Iterable[Iterable[int]],
    cum_fractions: Iterable[float],
    total_steps: int,
) -> list[tuple[frozenset[int], int]]:
    """``[(active_resolutions, step_span)]`` for each schedule stage over the step budget.

    ``cum_fractions`` are the cumulative stage fractions (the dataset's ``_schedule_cum_frac``);
    a stage's span is its slice of ``total_steps``. Without a schedule, pass a single stage with
    cum fraction ``1.0`` and the full resolution set.
    """
    spans: list[tuple[frozenset[int], int]] = []
    prev = 0
    for res, cum in zip(stage_resolutions, cum_fractions):
        end = round(float(cum) * int(total_steps))
        spans.append((frozenset(int(r) for r in res), max(0, end - prev)))
        prev = end
    return spans


def estimate_image_exposure(
    stages: Iterable[tuple[Iterable[int], int]],
    weight: Mapping[int, float],
    distinct: Mapping[int, float],
    batch_size: int,
) -> dict[int, float]:
    """Average number of times each distinct image at a resolution is trained over the run.

    ``stages``: ``[(active_resolutions, step_span)]``. ``weight``: per-resolution iteration-order
    rows (includes ``num_repeats`` / per-epoch cap) — used to split each step's batch across the
    active resolutions. ``distinct``: per-resolution unique image count — the divisor that turns
    image-views into per-image exposure. ``batch_size``: image-views consumed per optimizer step.
    """
    exposure: dict[int, float] = {int(r): 0.0 for r in weight}
    for active, span in stages:
        active = [int(r) for r in active]
        total_w = sum(float(weight.get(r, 0.0)) for r in active)
        if total_w <= 0 or span <= 0:
            continue
        for r in active:
            w = float(weight.get(r, 0.0))
            d = float(distinct.get(r, 0.0))
            if w <= 0 or d <= 0:
                continue
            views = span * batch_size * (w / total_w)
            exposure[r] = exposure.get(r, 0.0) + views / d
    return exposure


def format_exposure_report(
    exposure: Mapping[int, float], *, target: float | None = None
) -> str:
    """Human-readable report lines (rank 0). Flags resolutions below ``target`` if given."""
    header = "rengu_flow: estimated image exposure (avg times each image is trained per resolution)"
    lines = [header]
    for r in sorted(exposure):
        v = exposure[r]
        flag = ""
        if target is not None and v + 1e-9 < float(target):
            flag = f"  <-- below target {target:g}x"
        lines.append(f"  resolution {r}: ~{v:.1f}x{flag}")
    if not exposure:
        lines.append("  (no resolutions configured)")
    return "\n".join(lines)
