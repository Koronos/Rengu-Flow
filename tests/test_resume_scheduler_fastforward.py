"""On resume, the LR scheduler is rebuilt from config and fast-forwarded to the resumed step
(instead of restoring saved scheduler state). For the formula-based registry schedulers this must
reproduce the uninterrupted run's LR trajectory exactly, so an unchanged-config resume is
LR-identical. Exercises the real resolve_scheduler / apply_warmup paths with torch.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from rengu_flow.optim.resolver import apply_warmup, resolve_scheduler

TOTAL_STEPS = 100


def _build(sched_type: str, warmup: int):
    p = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([p], lr=1e-4)
    cfg = {"lr_scheduler": sched_type, "warmup_steps": warmup, "epochs": 10}
    sched = resolve_scheduler(sched_type, opt, cfg, TOTAL_STEPS, TOTAL_STEPS // 10)
    sched = apply_warmup(opt, sched, warmup)
    return opt, sched


@pytest.mark.parametrize("sched_type", ["constant", "linear", "cosine", "rex", "wsd"])
@pytest.mark.parametrize("warmup", [0, 5])
@pytest.mark.parametrize("resume_at", [1, 6, 50])
def test_fastforward_reproduces_uninterrupted_trajectory(sched_type, warmup, resume_at):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # step()-before-optimizer.step() during fast-forward

        # Uninterrupted reference: build once, step to the end, record the LR entering each step.
        opt_u, sched_u = _build(sched_type, warmup)
        traj = []
        for _ in range(TOTAL_STEPS):
            traj.append(opt_u.param_groups[0]["lr"])
            opt_u.step()
            sched_u.step()

        # Resume: fresh build, fast-forward `resume_at` steps, then continue to the end.
        opt_r, sched_r = _build(sched_type, warmup)
        for _ in range(resume_at):
            sched_r.step()
        for i in range(resume_at, TOTAL_STEPS):
            assert math.isclose(
                opt_r.param_groups[0]["lr"], traj[i], rel_tol=1e-9, abs_tol=1e-12
            ), f"{sched_type} warmup={warmup} resume@{resume_at}: LR diverges at step {i}"
            opt_r.step()
            sched_r.step()


def test_edited_peak_lr_scales_the_curve():
    """A resume that edits the peak LR (1e-4 -> 3e-4) yields the new-peak curve, not the old."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sched_type in ("constant", "linear", "cosine", "rex", "wsd"):
            opt_a, sched_a = _build(sched_type, 0)
            p = torch.nn.Parameter(torch.zeros(4))
            opt_b = torch.optim.SGD([p], lr=3e-4)
            cfg = {"lr_scheduler": sched_type, "warmup_steps": 0, "epochs": 10}
            sched_b = apply_warmup(
                opt_b, resolve_scheduler(sched_type, opt_b, cfg, TOTAL_STEPS, 10), 0
            )
            for _ in range(3):
                sched_a.step()
                sched_b.step()
            a, b = opt_a.param_groups[0]["lr"], opt_b.param_groups[0]["lr"]
            assert a > 0 and math.isclose(b / a, 3.0, rel_tol=1e-6), (
                f"{sched_type}: edited peak did not scale (ratio {b / a:.4f})"
            )
