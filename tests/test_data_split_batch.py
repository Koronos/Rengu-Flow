"""Tests for split_batch."""

import torch

import pytest

from renga_flow.data.loader import split_batch


@pytest.mark.parametrize("pieces, batch_size", [(1, 4), (2, 4), (4, 8)])
def test_split_batch_pieces_and_sizes(pieces, batch_size):
    """Split produces correct number of pieces and sizes sum to original batch size."""
    features = (torch.randn(batch_size, 3, 8, 8),)
    label = (torch.randn(batch_size, 3, 8, 8), torch.ones(batch_size, 1, 8, 8))
    result = split_batch((features, label), pieces)
    assert len(result) == pieces
    micro_size = batch_size // pieces
    for i in range(pieces):
        assert result[i][0][0].shape == (micro_size, 3, 8, 8)
        assert result[i][1][0].shape == (micro_size, 3, 8, 8)
    total = sum(r[0][0].size(0) for r in result)
    assert total == batch_size


def test_split_batch_handles_none():
    """When a tensor is None, split_batch produces empty tensors per piece."""
    features = (torch.randn(2, 1),)
    label = (None, None)
    result = split_batch((features, label), 2)
    assert len(result) == 2
    assert result[0][0][0].shape == (1, 1)
    for piece in result:
        assert len(piece[1]) == 2
        assert piece[1][0].numel() == 0
        assert piece[1][1].numel() == 0
