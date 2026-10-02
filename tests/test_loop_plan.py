"""Pure training-loop decisions (rengu_flow.training.loop_plan) and the resume sequence.

``main._run_training`` is a single long GPU function, so the choices it makes about step budget,
seeding, resume order and multi-GPU agreement are tested here as plain functions / with fakes.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rengu_flow.training.loop_plan import (
    ResumeState,
    can_emergency_save,
    data_parallel_size,
    examples_to_steps,
    force_constant_lr_enabled,
    rank_seed,
    restore_training_state,
    run_already_complete,
    seed_everything,
    step_budget,
    wsd_fork_step_for_resume,
)

pytestmark = pytest.mark.no_ui_db


@pytest.mark.parametrize(
    "max_steps,total,expected",
    [(None, 500, 500), (200, 500, 200), (900, 500, 900)],
)
def test_step_budget_is_where_the_loop_stops(max_steps, total, expected):
    # The LR horizon (and the WSD onset) are derived from this: max_steps must win even when
    # there is no resolution schedule, so the LR curve ends where the loop does.
    assert step_budget(max_steps, total) == expected


def test_wsd_onset_follows_max_steps_horizon():
    from rengu_flow.optim.resolver import wsd_decay_onset_step

    config = {"lr_scheduler_args": {"decay": 0.1}}
    horizon = step_budget(max_steps=200, total_steps=1000)
    assert wsd_decay_onset_step(config, horizon) == 180  # not 900 (the epochs-derived total)


@pytest.mark.parametrize(
    "world,stages,expected",
    [(1, 1, 1), (4, 1, 4), (4, 2, 2), (8, 4, 2), (2, 4, 1)],
)
def test_data_parallel_size_divides_out_pipeline_stages(world, stages, expected):
    assert data_parallel_size(world, stages) == expected


def test_rank_seed_offsets_by_rank():
    assert rank_seed(42, 0) == 42
    assert rank_seed(42, 3) == 45


@pytest.mark.parametrize(
    "onset,resumed_step,expected",
    [
        (180, 1, 180),      # fresh run: arm the fork
        (180, 100, 180),    # resumed before the onset: still ahead
        (180, 180, 180),    # last saved step 179 < onset: fork not yet written
        (180, 181, None),   # last saved step 180 >= onset: fork already written, keep it
        (180, 500, None),
        (0, 1, None),
        (None, 1, None),
    ],
)
def test_wsd_fork_not_rearmed_after_onset(onset, resumed_step, expected):
    assert wsd_fork_step_for_resume(onset, resumed_step) == expected


@pytest.mark.parametrize("step,budget,done", [(1, 10, False), (10, 10, False), (11, 10, True)])
def test_run_already_complete(step, budget, done):
    assert run_already_complete(step, budget) is done


@pytest.mark.parametrize("world,ok", [(1, True), (2, False), (8, False)])
def test_emergency_save_only_single_process(world, ok):
    assert can_emergency_save(world) is ok


@pytest.mark.parametrize(
    "config,expected",
    [({}, False), ({"force_constant_lr": False}, False), ({"force_constant_lr": True}, True)],
)
def test_force_constant_lr_is_a_boolean(config, expected):
    assert force_constant_lr_enabled(config) is expected


def test_examples_to_steps_converts_and_rejects_sub_batch_cadence():
    assert examples_to_steps("save_every_n_examples", 100, 4) == 25
    with pytest.raises(ValueError, match="save_every_n_examples"):
        examples_to_steps("save_every_n_examples", 3, 4)


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = random.random()
    seed_everything(123)
    assert random.random() == a


# --------------------------------------------------------------------------- resume sequence


class _FakeScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def _resume(tmp_path, *, config=None, load_path="ckpt", client_state=None, events=None, **over):
    events = events if events is not None else []
    optimizer = SimpleNamespace(param_groups=[{"params": [], "lr": 1e-4}])
    engine = MagicMock()
    engine.lr_scheduler = _FakeScheduler()
    state = client_state or {
        "step": 40,
        "examples": 160,
        "steps_per_epoch": 10,
        "custom_loader": {"epoch": 3},
        "rng_state": None,
    }
    engine.load_checkpoint.side_effect = lambda *a, **k: (events.append("load"), (load_path, state))[1]
    loader = MagicMock()
    loader.load_state_dict.side_effect = lambda s: events.append("loader")
    ema = MagicMock()
    sched = SimpleNamespace(current=lambda step: (step - 1) // 10 + 1, epochs=5)
    kwargs = dict(
        model_engine=engine,
        optimizer=optimizer,
        run_dir=str(tmp_path),
        resume_tag=None,
        reset_optimizer=True,
        reset_dataloader=False,
        train_dataloader=loader,
        training_ema=ema,
        parameters_to_train=[],
        epoch_schedule=sched,
        steps_per_epoch=10,
        global_batch_size=4,
        config=config or {"optimizer": {"type": "adamw", "lr": 1e-4}},
        is_main=False,
        barrier=lambda: events.append("barrier"),
        log=lambda *a, **k: None,
    )
    kwargs.update(over)
    return restore_training_state(**kwargs), engine, events


def test_resume_restores_step_examples_epoch_and_fast_forwards_scheduler(tmp_path):
    result, engine, _ = _resume(tmp_path)
    assert result == ResumeState(step=41, examples=164, epoch=5, resumed=True)
    assert engine.lr_scheduler.steps == 40  # one scheduler step per completed training step


def test_resume_force_constant_lr_false_still_fast_forwards(tmp_path):
    _, engine, _ = _resume(tmp_path, config={"force_constant_lr": False})
    assert engine.lr_scheduler.steps == 40


def test_resume_force_constant_lr_true_pins_scheduler(tmp_path):
    _, engine, _ = _resume(tmp_path, config={"force_constant_lr": True})
    assert engine.lr_scheduler.steps == 0


def test_resume_without_checkpoint_starts_fresh(tmp_path):
    result, engine, _ = _resume(tmp_path, load_path=None, client_state={})
    assert result.resumed is False and result.step == 1
    assert engine.lr_scheduler.steps == 0


def test_resume_restores_rng_after_load_and_nothing_reseeds_after(tmp_path, monkeypatch):
    """RNG restore happens after the engine/dataloader load, and is the LAST thing touching the
    global RNG (the old code re-seeded with train_seed right after, discarding it)."""
    events: list[str] = []
    monkeypatch.setattr(
        "rengu_flow.utils.rng_state.restore_rng_state", lambda st: events.append(f"rng:{st}")
    )
    state = {
        "step": 40,
        "examples": 160,
        "custom_loader": {"epoch": 3},
        "rng_state": {"python": "S"},
    }
    _resume(tmp_path, events=events, client_state=state)
    assert events.index("load") < events.index("loader") < events.index("rng:{'python': 'S'}")


def test_resume_reseeds_ema_when_checkpoint_has_no_ema_file(tmp_path):
    import torch

    from rengu_flow.training.ema import TrainingEMA

    p = torch.nn.Parameter(torch.tensor([1.0]))
    ema = TrainingEMA([p], decay=0.9)
    with torch.no_grad():
        p.fill_(5.0)  # weights loaded from the checkpoint after the EMA was built
    ckpt_dir = tmp_path / "global_step40"
    ckpt_dir.mkdir()
    _resume(tmp_path, load_path=str(ckpt_dir), training_ema=ema, parameters_to_train=[p])
    assert ema.shadow[id(p)].item() == 5.0
