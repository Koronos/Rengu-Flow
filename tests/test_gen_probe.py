"""CPU tests for the deterministic generalization probe (val loss + train probe + GAP).

No GPU / DeepSpeed: the model engine and dataloaders are mocked. We verify the gap math
(val − train) and that frozen-noise determinism yields identical results across two calls.
"""

from unittest.mock import patch

from rengu_flow.utils.gen_probe import generalization_probe
from rengu_track import NullSink


class _MockEngine:
    """Returns a per-loader constant loss (tagged via set_eval_quantile interplay)."""

    def reset_activation_shape(self):
        pass

    def eval_batch(self, iterator, num_micro_batches=None):
        import torch

        # The iterator carries the loss value the fake loader set for itself.
        return torch.tensor(next(iterator))


class _MockLoader:
    """Minimal PipelineDataLoader stand-in: one batch per pass, fixed per-loader loss."""

    def __init__(self, loss_value: float):
        self.loss_value = loss_value
        self.epoch = 1
        self.eval_quantile = None
        self.reset_count = 0

    def set_eval_quantile(self, q):
        self.eval_quantile = q

    def sync_epoch(self):
        self.epoch = 2  # one batch then loop exits

    def reset(self):
        self.reset_count += 1
        self.epoch = 1


class _MockModel:
    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass

    def prepare_block_swap_training(self):
        pass


def _run(val_loss: float, train_loss: float | None, probe_batches=None):
    val_loader = _MockLoader(val_loss)
    train_loader = _MockLoader(train_loss) if train_loss is not None else None

    def fake_iter(dataloader, engine, num_micro_batches=None):
        # Yield this loader's loss so eval_batch returns it.
        return iter([dataloader.loss_value])

    with patch("rengu_flow.utils.eval.get_data_iterator_for_step", side_effect=fake_iter):
        return generalization_probe(
            _MockModel(),
            _MockEngine(),
            val_loader,
            train_loader,
            NullSink(),
            step=100,
            eval_gradient_accumulation_steps=1,
            disable_block_swap=False,
            probe_batches=probe_batches,
        )


def test_gap_is_val_minus_train():
    out = _run(0.7, 0.4)
    assert out is not None
    assert abs(out["val_loss"] - 0.7) < 1e-5
    assert abs(out["train_probe"] - 0.4) < 1e-5
    # GAP = val - train_probe (the overfitting signal).
    assert abs(out["val_gap"] - (0.7 - 0.4)) < 1e-5


def test_determinism_identical_across_calls():
    a = _run(0.55, 0.33)
    b = _run(0.55, 0.33)
    assert a == b


def test_no_train_loader_logs_val_only():
    out = _run(0.6, None)
    assert out is not None
    assert "val_loss" in out
    assert "val_gap" not in out
    assert "train_probe" not in out


def test_no_val_loader_is_noop():
    # No val source → graceful no-op, never crashes training.
    out = generalization_probe(
        _MockModel(),
        _MockEngine(),
        None,
        None,
        NullSink(),
        step=1,
        eval_gradient_accumulation_steps=1,
        disable_block_swap=False,
    )
    assert out is None
