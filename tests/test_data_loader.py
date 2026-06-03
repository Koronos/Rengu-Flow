"""Tests for PipelineDataLoader (with mock model and engine)."""

from unittest.mock import patch

import pytest

from rengu_flow.data import PipelineDataLoader, SyntheticSDXLDataset


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


def test_pipeline_data_loader_thread_prefetch():
    ds = SyntheticSDXLDataset(num_batches=2, micro_batch_size=1, latent_height=64, latent_width=64)
    mock_model = _make_mock_model()
    mock_engine = _make_mock_engine()
    batches = [ds[0], ds[1]]

    class PrefetchDataLoader:
        def __iter__(self):
            return iter(batches)

    loader = PipelineDataLoader(
        ds,
        mock_engine,
        gradient_accumulation_steps=1,
        model=mock_model,
        dataloader_prefetch=True,
    )
    loader.dataloader = PrefetchDataLoader()
    loader.data = loader._pull_batches_from_dataloader()
    micro = next(iter(loader))
    assert micro is not None
    loader._stop_prefetch_thread()


def test_pipeline_data_loader_dataloader_kwargs():
    ds = SyntheticSDXLDataset(num_batches=1, micro_batch_size=1)
    mock_model = _make_mock_model()
    mock_engine = _make_mock_engine()
    with patch("rengu_flow.data.loader.torch.utils.data.DataLoader") as mock_dl:
        mock_dl.return_value = iter([])
        PipelineDataLoader(
            ds,
            mock_engine,
            gradient_accumulation_steps=1,
            model=mock_model,
            num_dataloader_workers=2,
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=False,
        )
        _, kwargs = mock_dl.call_args
        assert kwargs["num_workers"] == 2
        assert kwargs["pin_memory"] is True
        assert kwargs["prefetch_factor"] == 3
        assert kwargs["persistent_workers"] is False


def test_pipeline_data_loader_propagates_epoch_to_dataset():
    """set_epoch is called on the dataset at creation and on each epoch rollover."""

    class RotatingSynthetic(SyntheticSDXLDataset):
        rotation_active = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.epochs_seen = []

        def set_epoch(self, epoch):
            self.epochs_seen.append(epoch)

    ds = RotatingSynthetic(num_batches=2, micro_batch_size=1, latent_height=64, latent_width=64)
    mock_model = _make_mock_model()
    mock_engine = _make_mock_engine()
    loader = PipelineDataLoader(ds, mock_engine, gradient_accumulation_steps=1, model=mock_model)
    # Epoch 1 is set when the dataloader is first created.
    assert ds.epochs_seen == [1]
    it = iter(loader)
    # Drive past the end of epoch 1 to trigger the rollover.
    for _ in range(len(ds) + 1):
        next(it)
    assert loader.epoch == 2
    assert ds.epochs_seen[-1] == 2


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
