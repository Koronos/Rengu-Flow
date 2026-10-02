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


def test_resume_at_epoch_boundary_yields_real_batches():
    """A checkpoint saved at the last step of an epoch records num_batches_pulled == len(dataset)
    (the pulled count includes the one-batch prefetch). Resuming must roll into the next epoch
    and yield real batches — not iterate an empty skip-everything dataloader forever (which the
    engine renders as infinite zero-loss steps)."""
    ds = SyntheticSDXLDataset(num_batches=2, micro_batch_size=1, latent_height=64, latent_width=64)
    loader = PipelineDataLoader(ds, _make_mock_engine(), gradient_accumulation_steps=1, model=_make_mock_model())
    loader.load_state_dict({"epoch": 1, "num_batches_pulled": len(ds)})
    assert loader.epoch == 2, "fully-consumed epoch must roll forward on load"
    micro_batch = next(iter(loader))
    assert micro_batch is not None


def test_first_pull_stopiteration_rolls_epoch():
    """Even if the freshly (re)created dataloader comes up empty (partial skip states), the FIRST
    pull must roll the epoch like the buffered path instead of leaking StopIteration to the engine."""
    ds = SyntheticSDXLDataset(num_batches=2, micro_batch_size=1, latent_height=64, latent_width=64)
    loader = PipelineDataLoader(ds, _make_mock_engine(), gradient_accumulation_steps=1, model=_make_mock_model())
    # Simulate a resume that skipped one short of everything, then exhaust the rest.
    loader.load_state_dict({"epoch": 1, "num_batches_pulled": len(ds) - 1})
    it = iter(loader)
    batches = [next(it) for _ in range(3)]  # crosses the boundary into epoch 2
    assert all(b is not None for b in batches)
    assert loader.epoch >= 2


class _EpochSpy(SyntheticSDXLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_seen = []

    def set_epoch(self, epoch):
        self.epochs_seen.append(epoch)


def test_non_owner_loader_never_touches_dataset_epoch():
    """The val-gap train probe shares the live train dataset: creating, reset()ing and rolling
    its loader must not call set_epoch on it (reset used to rewind train to epoch 1)."""
    ds = _EpochSpy(num_batches=2, micro_batch_size=1, latent_height=64, latent_width=64)
    loader = PipelineDataLoader(
        ds, _make_mock_engine(), 1, _make_mock_model(), owns_dataset_epoch=False
    )
    it = iter(loader)
    for _ in range(len(ds) + 1):
        next(it)
    loader.reset()
    next(iter(loader))
    assert ds.epochs_seen == []


def test_state_dict_counts_only_consumed_batches():
    """num_batches_pulled counts the one-batch prefetch; the checkpointed value must be the
    batches actually consumed, or every resume silently skips one untrained batch."""
    ds = _EpochSpy(num_batches=5, micro_batch_size=2, latent_height=64, latent_width=64)
    loader = PipelineDataLoader(ds, _make_mock_engine(), 2, _make_mock_model())
    it = iter(loader)
    for _ in range(2 * 3):  # 3 full steps of 2 micro-batches
        next(it)
    assert loader.num_batches_pulled == 4  # prefetch already started batch 4
    assert loader.state_dict()["num_batches_pulled"] == 3


def test_resume_does_not_skip_untrained_batch():
    ds = _EpochSpy(num_batches=5, micro_batch_size=1, latent_height=64, latent_width=64)
    loader = PipelineDataLoader(ds, _make_mock_engine(), 1, _make_mock_model())
    it = iter(loader)
    for _ in range(2):
        next(it)
    state = loader.state_dict()
    assert state["num_batches_pulled"] == 2
    resumed = PipelineDataLoader(ds, _make_mock_engine(), 1, _make_mock_model())
    resumed.load_state_dict(state)
    assert resumed._resume_skip == 2
    # And the position survives a second save without drifting.
    next(iter(resumed))
    assert resumed.state_dict()["num_batches_pulled"] == 3
