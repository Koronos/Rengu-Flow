"""A resume checkpoint must be written with the optimizer in eval mode.

Eval/train-style optimizers (MSAM/Nekaon lookahead, ScheduleFree, Lookahead) keep
the between-step live weights deliberately displaced from the true iterate; only
``optimizer.eval()`` restores the real weights. A checkpoint saved in train mode
stores the displaced weights, and the optimizer cannot know to undo that on resume
(``load_state_dict`` resets its "perturbation present" flag), so the run resumes
off-point and degrades — exactly the symptom on a Nekaon + BOFT run. The preview /
eval paths already bracket their reads with eval()/train(); the checkpoint save must
do the same.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rengu_flow.utils.saver import Saver


class _SpyOptimizer:
    """Records eval/train transitions so a test can assert the mode during the save."""

    def __init__(self) -> None:
        self.mode = "train"
        self.events: list[str] = []

    def eval(self) -> None:  # noqa: A003 - mirrors optimizer.eval() API
        self.mode = "eval"
        self.events.append("eval")

    def train(self) -> None:
        self.mode = "train"
        self.events.append("train")


def _make_saver(tmp_path: Path, model_engine: MagicMock) -> Saver:
    args = MagicMock()
    args.config = str(tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("# test")
    train_dataloader = MagicMock()
    train_dataloader.state_dict.return_value = {"epoch": 1}
    return Saver(
        args,
        {},
        True,
        tmp_path,
        MagicMock(),
        train_dataloader,
        model_engine,
        MagicMock(),
    )


def test_checkpoint_saved_in_optimizer_eval_mode(tmp_path: Path):
    spy = _SpyOptimizer()
    model_engine = MagicMock()
    model_engine.optimizer = spy
    mode_at_save: list[str] = []
    model_engine.save_checkpoint.side_effect = lambda *a, **k: mode_at_save.append(spy.mode)

    saver = _make_saver(tmp_path, model_engine)
    with patch("rengu_flow.utils.saver.dist") as mock_dist:
        mock_dist.barrier = MagicMock()
        with patch("rengu_flow.utils.saver.is_main_process", return_value=True):
            ok = saver.save_checkpoint(5, 50)

    assert ok is True
    assert mode_at_save == ["eval"], "checkpoint was not written while the optimizer was in eval mode"
    assert spy.events and spy.events[0] == "eval"
    assert spy.events[-1] == "train", "optimizer must be returned to train mode after the save"
    assert spy.mode == "train"


def test_checkpoint_restores_train_mode_on_disk_full(tmp_path: Path):
    """Even when the save fails (disk full), the optimizer must be returned to train mode."""
    spy = _SpyOptimizer()
    model_engine = MagicMock()
    model_engine.optimizer = spy
    model_engine.save_checkpoint.side_effect = OSError(28, "No space left on device")

    saver = _make_saver(tmp_path, model_engine)
    (tmp_path / "global_step1").mkdir()
    with patch("rengu_flow.utils.saver.dist") as mock_dist:
        mock_dist.barrier = MagicMock()
        with patch("rengu_flow.utils.saver.is_main_process", return_value=True):
            ok = saver.save_checkpoint(5, 50)

    assert ok is False
    assert "eval" in spy.events and spy.events[-1] == "train"
    assert spy.mode == "train"


def test_checkpoint_save_without_eval_capable_optimizer(tmp_path: Path):
    """Plain optimizers (no eval/train) still checkpoint fine — the eval wrap is a no-op."""
    model_engine = MagicMock()
    model_engine.optimizer = object()  # no eval/train methods

    saver = _make_saver(tmp_path, model_engine)
    with patch("rengu_flow.utils.saver.dist") as mock_dist:
        mock_dist.barrier = MagicMock()
        with patch("rengu_flow.utils.saver.is_main_process", return_value=True):
            ok = saver.save_checkpoint(5, 50)

    assert ok is True
