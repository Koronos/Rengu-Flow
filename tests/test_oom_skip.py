"""CPU tests for train.oom_skip helpers."""

import pytest
import torch

from rengu_flow.utils.oom_skip import (
    OomSkipState,
    handle_oom_skip,
    is_cuda_oom,
    reset_engine_timers,
)


class _FakeTimer:
    def __init__(self):
        self.started_ = True

    def reset(self):
        self.started_ = False


class _FakeTimerGroup:
    def __init__(self):
        self.timers = {"train_batch": _FakeTimer(), "fwd_microstep": _FakeTimer()}


def test_is_cuda_oom_out_of_memory_error():
    assert is_cuda_oom(torch.cuda.OutOfMemoryError("x"))


def test_is_cuda_oom_runtime_message():
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))


def test_is_cuda_oom_other_runtime_error():
    assert not is_cuda_oom(RuntimeError("something else"))


def test_oom_skip_state_resets_on_success():
    state = OomSkipState(max_consecutive=3)
    state.consecutive = 2
    state.record_success()
    assert state.consecutive == 0


def test_oom_skip_state_aborts_after_max():
    state = OomSkipState(max_consecutive=2)
    state.record_skip()
    state.record_skip()
    with pytest.raises(RuntimeError, match="OOM during training"):
        state.record_skip()


def test_handle_oom_skip_zeros_grad():
    engine = type("E", (), {})()
    engine.optimizer = type("O", (), {"zeroed": False})()

    def zero_grad(set_to_none=True):
        engine.optimizer.zeroed = True

    engine.optimizer.zero_grad = zero_grad
    state = OomSkipState(max_consecutive=3)
    handle_oom_skip(state, engine, clear_cache=False)
    assert engine.optimizer.zeroed is True


def test_reset_engine_timers_clears_started_state():
    engine = type("E", (), {})()
    engine.timers = _FakeTimerGroup()
    reset_engine_timers(engine)
    assert all(not t.started_ for t in engine.timers.timers.values())


def test_reset_engine_timers_tolerates_engines_without_timers():
    reset_engine_timers(type("E", (), {})())  # must not raise


def test_handle_oom_skip_resets_timers():
    engine = type("E", (), {})()
    engine.optimizer = type("O", (), {})()
    engine.optimizer.zero_grad = lambda set_to_none=True: None
    engine.timers = _FakeTimerGroup()
    handle_oom_skip(OomSkipState(max_consecutive=3), engine, clear_cache=False)
    assert all(not t.started_ for t in engine.timers.timers.values())
