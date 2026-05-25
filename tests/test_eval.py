"""Tests for evaluation module (§1.6). Use mocks; no GPU or DeepSpeed."""

import pytest

from renga_flow.utils.eval import (
    TIMESTEP_QUANTILES_FOR_EVAL,
    evaluate,
    evaluate_single,
)


def test_timestep_quantiles_for_eval_constant():
    """TIMESTEP_QUANTILES_FOR_EVAL is the expected list of 9 quantiles."""
    assert TIMESTEP_QUANTILES_FOR_EVAL == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert len(TIMESTEP_QUANTILES_FOR_EVAL) == 9


def test_evaluate_empty_dataloaders_returns_immediately():
    """evaluate() with no eval_dataloaders returns without calling model or engine."""
    mock_model = object()
    mock_engine = object()
    evaluate(
        mock_model,
        mock_engine,
        {},
        None,
        0,
        1,
        False,
        optimizer=None,
        wandb_enable=False,
    )
    # No exception; we did not touch model/engine (no prepare_block_swap_* etc.)


def test_evaluate_single_contract_mock_engine_and_loader():
    """evaluate_single with mock engine and loader returns mean loss and calls reset."""
    import torch

    class MockEngine:
        def reset_activation_shape(self):
            pass

        def eval_batch(self, iterator, num_micro_batches=None):
            return torch.tensor(0.5)

        def is_first_stage(self):
            return True

        def is_last_stage(self):
            return True

    class MockLoader:
        def __init__(self):
            self.epoch = 1
            self._reset_called = False
            self._quantile = None

        def set_eval_quantile(self, q):
            self._quantile = q

        def sync_epoch(self):
            self.epoch = 2  # After one "step" we go to epoch 2 so loop exits

        def reset(self):
            self._reset_called = True
            self.epoch = 1

    loader = MockLoader()
    engine = MockEngine()
    # get_data_iterator_for_step needs real iterator behavior; we need to mock at pipeline level.
    # Instead, test that with a single "step" (epoch becomes 2) we get one loss and reset is called.
    # We have to patch get_data_iterator_for_step to return an iterator of one item so eval_batch gets one micro-batch.
    from unittest.mock import patch, MagicMock

    def fake_get_iterator(dataloader, eng, num_micro_batches=None):
        return iter([None])  # one micro-batch

    with patch("renga_flow.utils.eval.get_data_iterator_for_step", side_effect=fake_get_iterator):
        mean_loss = evaluate_single(engine, loader, 1, 0.5, pbar=None)
    assert mean_loss == 0.5
    assert loader._reset_called
    assert loader._quantile == 0.5
