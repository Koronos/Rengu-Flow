"""EMA shadow: update, apply-at-export, checkpoint persistence, and the eval-mode ordering.

The load-bearing property is the export path: with a lookahead optimizer, ``optimizer.eval()``
must restore the true iterate BEFORE the EMA weights are swapped in — otherwise eval() would
overwrite them. The last test drives ``Saver._run_pipeline_export`` end to end to lock that order.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rengu_flow.training.ema import (
    TrainingEMA,
    load_ema_checkpoint,
    save_ema_checkpoint,
)
from rengu_flow.utils.saver import Saver


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


def test_average_parameters_swaps_then_restores():
    p = _params(2.0)
    ema = TrainingEMA(p, decay=0.5)
    ema.shadow[id(p[0])].fill_(7.0)  # pretend the average diverged from the live weight
    with ema.average_parameters(p):
        assert p[0].item() == 7.0, "EMA weights must be live inside the context"
    assert p[0].item() == 2.0, "live weights must be restored on exit"


def test_state_dict_roundtrip():
    p = _params(1.0, 2.0)
    ema = TrainingEMA(p, decay=0.99)
    ema.shadow[id(p[0])].fill_(3.0)
    ema.shadow[id(p[1])].fill_(4.0)
    state = ema.state_dict(p)

    # Fresh EMA (shadow at current weights) then load the serialized shadow back.
    fresh = TrainingEMA(p, decay=0.99)
    fresh.load_state_dict(state, p)
    assert fresh.shadow[id(p[0])].item() == 3.0
    assert fresh.shadow[id(p[1])].item() == 4.0
    assert fresh.decay == 0.99


def test_load_state_dict_rejects_mismatched_param_count():
    p = _params(1.0, 2.0)
    ema = TrainingEMA(p, decay=0.9)
    state = ema.state_dict(p)
    try:
        ema.load_state_dict(state, _params(1.0))  # only 1 param now
        raise AssertionError("expected ValueError on param-count mismatch")
    except ValueError:
        pass


def test_save_and_load_checkpoint_via_latest_marker(tmp_path: Path):
    p = _params(1.0, 2.0)
    ema = TrainingEMA(p, decay=0.9)
    ema.shadow[id(p[0])].fill_(5.0)
    ema.shadow[id(p[1])].fill_(6.0)

    tag = "global_step10"
    (tmp_path / tag).mkdir()
    (tmp_path / "latest").write_text(tag)

    with patch("rengu_flow.utils.common.is_main_process", return_value=True):
        save_ema_checkpoint(tmp_path, ema, p)
    assert (tmp_path / tag / "ema.pt").is_file()

    fresh = TrainingEMA(p, decay=0.9)
    # accelerate engine returns the checkpoint FILE path; loader must resolve its parent dir.
    load_ema_checkpoint(str(tmp_path / tag / "torch_engine.pt"), fresh, p)
    assert fresh.shadow[id(p[0])].item() == 5.0
    assert fresh.shadow[id(p[1])].item() == 6.0


def test_load_checkpoint_absent_file_is_noop(tmp_path: Path):
    p = _params(1.0)
    ema = TrainingEMA(p, decay=0.9)
    load_ema_checkpoint(str(tmp_path / "global_step1" / "torch_engine.pt"), ema, p)
    assert ema.shadow[id(p[0])].item() == 1.0  # unchanged


def test_export_applies_ema_inside_optimizer_eval(tmp_path: Path):
    """The full ordering: eval() restores true iterate → EMA weights swapped in → export
    reads EMA → live weights restored → train() re-displaces."""

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([10.0]))  # displaced (train mode)

    net = _Net()

    class _SpyOptimizer:
        mode = "train"

        def __init__(self):
            self.events: list[str] = []

        def eval(self):
            self.mode = "eval"
            self.events.append("eval")
            with torch.no_grad():
                net.w.fill_(1.0)  # true iterate

        def train(self):
            self.mode = "train"
            self.events.append("train")
            with torch.no_grad():
                net.w.fill_(10.0)  # re-displace

    spy = _SpyOptimizer()
    ema = TrainingEMA([net.w], decay=0.9)
    ema.shadow[id(net.w)].fill_(5.0)  # the average

    args = MagicMock()
    args.config = str(tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("# test")
    model_engine = MagicMock()
    model_engine.optimizer = spy
    saver = Saver(args, {}, True, tmp_path, MagicMock(), MagicMock(),
                  model_engine, net, training_ema=ema)
    saver._async_writer = None

    weight_at_export: list[float] = []
    mode_at_export: list[str] = []
    saver._run_pipeline_export_sync = (  # type: ignore[method-assign]
        lambda name, *, adapter_only: (
            weight_at_export.append(net.w.item()),
            mode_at_export.append(spy.mode),
        )
    )

    saver._run_pipeline_export("step5", adapter_only=True)

    assert weight_at_export == [5.0], "export must read the EMA average, not true/displaced weights"
    assert mode_at_export == ["eval"]
    assert net.w.item() == 10.0, "live (displaced) weight must be restored after export"
    assert spy.events == ["eval", "train"]


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
