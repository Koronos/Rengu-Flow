"""EMA shadow: update semantics and the update_interval decay compounding."""

import torch
from torch import nn

from rengu_flow.training.ema import TrainingEMA


def _params(*values: float) -> list[nn.Parameter]:
    return [nn.Parameter(torch.tensor([v])) for v in values]


def test_update_moves_shadow_toward_weights():
    p = _params(0.0)
    ema = TrainingEMA(p, decay=0.9)
    with torch.no_grad():
        p[0].fill_(10.0)
    ema.update(p)
    # shadow = 0.9*0 + 0.1*10 = 1.0
    assert abs(ema.shadow[id(p[0])].item() - 1.0) < 1e-6


def test_update_interval_compounds_decay():
    # decay^N per update over 1/N the updates == same smoothing horizon as per-step.
    p = _params(0.0)
    ema = TrainingEMA(p, decay=0.9, update_interval=3)
    assert abs(ema.decay - 0.9**3) < 1e-12
    assert ema.update_interval == 3
    with torch.no_grad():
        p[0].fill_(10.0)
    ema.update(p)
    # shadow = 0.729*0 + 0.271*10
    assert abs(ema.shadow[id(p[0])].item() - 2.71) < 1e-6


def test_from_config_reads_update_interval():
    p = _params(1.0)
    ema = TrainingEMA.from_config({"ema_decay": 0.999, "ema_update_interval": 10}, p)
    assert ema.update_interval == 10
    assert abs(ema.decay - 0.999**10) < 1e-12
    default = TrainingEMA.from_config({"ema_decay": 0.999}, p)
    assert default.update_interval == 1
    assert abs(default.decay - 0.999) < 1e-12
