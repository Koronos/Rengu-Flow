"""Tests for renga_flow.utils.training_metrics."""

from unittest.mock import MagicMock

import torch

from renga_flow.utils.training_metrics import (
    get_automagic_lrs,
    get_prodigy_d,
    log_training_step,
)


def test_get_prodigy_d_averages_groups():
    opt = MagicMock()
    opt.param_groups = [{"d": 2.0}, {"d": 4.0}]
    assert get_prodigy_d(opt) == 3.0


def test_get_automagic_lrs_mean():
    opt = MagicMock()
    p = torch.nn.Parameter(torch.zeros(1))
    opt.param_groups = [{"params": [p]}]
    opt.state = {p: {}}
    opt._get_lr = MagicMock(return_value=torch.tensor(0.5))
    lrs, avg = get_automagic_lrs(opt)
    assert lrs.numel() == 1
    assert avg == 0.5


def test_log_training_step_prodigy_scalar():
    tb = MagicMock()
    opt = MagicMock()
    opt.__class__.__name__ = "Prodigy"
    opt.param_groups = [{"d": 1.0}]
    log_training_step(
        tb_writer=tb,
        wandb_enable=False,
        optimizer=opt,
        loss=0.1,
        x_axis=10,
        step=10,
        logging_steps=10,
        is_main=True,
    )
    tb.add_scalar.assert_any_call("train/loss", 0.1, 10)
    tb.add_scalar.assert_any_call("train/prodigy_d", 1.0, 10)


def test_log_training_step_automagic_histogram_when_avg_positive():
    tb = MagicMock()

    class GenericOptim:
        def __init__(self):
            p = torch.nn.Parameter(torch.zeros(1))
            self.param_groups = [{"params": [p]}]
            self.state = {p: {}}

        def _get_lr(self, group, state):
            return torch.tensor(0.2)

    opt = GenericOptim()
    log_training_step(
        tb_writer=tb,
        wandb_enable=False,
        optimizer=opt,
        loss=0.1,
        x_axis=5,
        step=5,
        logging_steps=5,
        is_main=True,
    )
    tb.add_histogram.assert_called_once()
    avg_calls = [
        c for c in tb.add_scalar.call_args_list if c[0][0] == "train/automagic_avg_lr"
    ]
    assert len(avg_calls) == 1
    assert avg_calls[0][0][2] == 5


def test_log_training_step_skips_when_not_logging_step():
    tb = MagicMock()
    opt = MagicMock()
    opt.__class__.__name__ = "AdamW"
    log_training_step(
        tb_writer=tb,
        wandb_enable=False,
        optimizer=opt,
        loss=0.1,
        x_axis=1,
        step=11,
        logging_steps=10,
        is_main=True,
    )
    tb.add_scalar.assert_not_called()
