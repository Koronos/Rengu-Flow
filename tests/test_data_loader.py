"""Tests for PipelineDataLoader (with mock model and engine)."""

import pytest
import torch

from renga_flow.data import PipelineDataLoader, SyntheticSDXLDataset


def _make_mock_model():
    class MockModel:
        def prepare_inputs(self, batch, timestep_quantile=None):
            latents = batch["latents"]
            mask = batch["mask"]
            features = (latents, latents, latents, latents, latents)
            label = (latents, mask)
            return features, label

    return MockModel()


def _make_mock_engine():
    class MockEngine:
        is_pipe_parallel = False

    return MockEngine()


def test_pipeline_data_loader_empty_dataset_raises():
    ds = SyntheticSDXLDataset(num_batches=0, micro_batch_size=1)
    mock_model = _make_mock_model()
    mock_engine = _make_mock_engine()
    with pytest.raises(RuntimeError) as exc_info:
        PipelineDataLoader(ds, mock_engine, 1, mock_model)
    assert "empty" in str(exc_info.value).lower()


def test_pipeline_data_loader_len():
    ds = SyntheticSDXLDataset(num_batches=2, micro_batch_size=1)
    mock_model = _make_mock_model()
    mock_engine = _make_mock_engine()
    loader = PipelineDataLoader(ds, mock_engine, gradient_accumulation_steps=1, model=mock_model)
    assert len(loader) == len(ds) * 1


def test_pipeline_data_loader_one_iteration():
    ds = SyntheticSDXLDataset(num_batches=2, micro_batch_size=1, latent_height=64, latent_width=64)
    mock_model = _make_mock_model()
    mock_engine = _make_mock_engine()
    loader = PipelineDataLoader(ds, mock_engine, gradient_accumulation_steps=1, model=mock_model)
    it = iter(loader)
    micro_batch = next(it)
    features, label = micro_batch
    assert len(features) == 5
    assert all(f.shape[0] == 1 for f in features)
    assert len(label) == 2
    assert label[0].shape[0] == 1
    assert label[1].shape[0] == 1


def test_pipeline_data_loader_reset():
    """reset() restores epoch, num_batches_pulled, next_micro_batch and reinitializes batch iterator."""
    ds = SyntheticSDXLDataset(num_batches=2, micro_batch_size=1, latent_height=64, latent_width=64)
    mock_model = _make_mock_model()
    mock_engine = _make_mock_engine()
    loader = PipelineDataLoader(ds, mock_engine, gradient_accumulation_steps=1, model=mock_model)
    assert loader.epoch == 1
    assert loader.num_batches_pulled == 0
    it = iter(loader)
    next(it)
    next(it)
    loader.reset()
    assert loader.epoch == 1
    assert loader.num_batches_pulled == 0
    assert loader.next_micro_batch is None
    # Can iterate again from the start
    micro_batch = next(iter(loader))
    assert micro_batch is not None
