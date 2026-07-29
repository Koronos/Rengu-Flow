"""Iterator materialization policy for pipeline and single-device engines."""

from rengu_flow.utils.pipeline import get_data_iterator_for_step


class _CountingIterator:
    def __init__(self):
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        return self.count


class _Engine:
    micro_batches = 3

    def __init__(self, preload):
        self.preload_micro_batches = preload

    def is_first_stage(self):
        return True

    def is_last_stage(self):
        return True


def test_single_device_policy_consumes_micro_batches_lazily():
    loader = _CountingIterator()
    batches = get_data_iterator_for_step(loader, _Engine(preload=False))
    assert loader.count == 0
    assert next(batches) == 1
    assert loader.count == 1
    assert list(batches) == [2, 3]


def test_pipeline_policy_preloads_entire_accumulation_step():
    loader = _CountingIterator()
    batches = get_data_iterator_for_step(loader, _Engine(preload=True))
    assert loader.count == 3
    assert list(batches) == [1, 2, 3]
