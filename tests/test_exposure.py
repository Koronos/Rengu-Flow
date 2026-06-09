"""Unit tests for the per-resolution image-exposure estimate."""

from rengu_flow.data.exposure import (
    estimate_image_exposure,
    format_exposure_report,
    schedule_stage_spans,
)


def test_schedule_stage_spans_splits_budget_by_cumulative_fraction():
    # 3 stages at 0.4 / 0.4 / 0.2 over a 1500-step budget.
    stages = schedule_stage_spans(
        [[512, 1024], [768, 1024], [1024]],
        [0.4, 0.8, 1.0],
        1500,
    )
    assert [span for _res, span in stages] == [600, 600, 300]
    assert stages[0][0] == frozenset({512, 1024})


def test_exposure_no_schedule_is_uniform_across_resolutions():
    # One stage, all resolutions active for the whole run. Each image (300 distinct, batch 1,
    # 1500 steps) is trained 1500/300 = 5 times, regardless of resolution.
    pools = {512: 100, 768: 100, 1024: 100}
    stages = [(frozenset({512, 768, 1024}), 1500)]
    exp = estimate_image_exposure(stages, weight=pools, distinct=pools, batch_size=1)
    assert exp == {512: 5.0, 768: 5.0, 1024: 5.0}


def test_exposure_quantifies_schedule_imbalance():
    # The reported worry made concrete: with the user's stages, 1024 (in all three stages) is
    # trained far more than 512/768 (each in only one stage). Equal pools of 100, batch 1.
    pools = {512: 100, 768: 100, 1024: 100}
    stages = schedule_stage_spans(
        [[512, 1024], [768, 1024], [1024]], [0.4, 0.8, 1.0], 1500
    )
    exp = estimate_image_exposure(stages, weight=pools, distinct=pools, batch_size=1)
    # stage1 [512,1024]: 600/(200)=3 each; stage2 [768,1024]: 3 each; stage3 [1024]: 300/100=3.
    assert exp[512] == 3.0
    assert exp[768] == 3.0
    assert exp[1024] == 9.0  # 3 + 3 + 3


def test_exposure_uniform_num_repeats_cancels():
    # num_repeats=3 multiplies entries (weight) but not distinct images; exposure is unchanged
    # vs num_repeats=1 because the repeat factor cancels in the proportional split.
    distinct = {512: 100, 1024: 100}
    weight = {512: 300, 1024: 300}  # 3x repeats
    stages = [(frozenset({512, 1024}), 1000)]
    exp = estimate_image_exposure(stages, weight=weight, distinct=distinct, batch_size=1)
    assert exp[512] == exp[1024]
    assert round(exp[512], 6) == round(1000 / 200, 6)  # == no-repeats case


def test_exposure_batch_size_scales_linearly():
    pools = {512: 100}
    stages = [(frozenset({512}), 100)]
    assert estimate_image_exposure(stages, pools, pools, batch_size=4)[512] == 4.0


def test_format_exposure_report_flags_below_target():
    # target=5: 512 (3.0x) is below and flagged; 1024 (9.0x) is above and not flagged.
    report = format_exposure_report({512: 3.0, 1024: 9.0}, target=5)
    assert "resolution 512: ~3.0x  <-- below target 5x" in report
    assert "resolution 1024: ~9.0x" in report
    assert "below target" not in report.splitlines()[-1]  # 1024 line (last) not flagged
