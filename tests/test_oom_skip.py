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


def test_is_cuda_oom_triton_message():
    # Fused optimizer kernels (kaon/adakaon) OOM through Triton with a different string than torch.
    assert is_cuda_oom(RuntimeError("Triton Error [CUDA]: out of memory"))


def test_is_cuda_oom_other_runtime_error():
    assert not is_cuda_oom(RuntimeError("something else"))
    assert not is_cuda_oom(RuntimeError("DefaultCPUAllocator: not enough memory"))


def test_oom_skip_at_limit_within_window():
    # 3 OOMs within the 10-step window -> at limit on the 3rd (the point to bump swap or abort).
    state = OomSkipState(max_in_window=3)
    state.record_skip(0)
    assert not state.at_limit(0)
    state.record_skip(4)
    assert not state.at_limit(4)
    state.record_skip(9)
    assert state.at_limit(9)  # 3 OOMs in steps 0..9


def test_oom_skip_window_forgets_aged_out_steps():
    # OOMs interleaved with good steps but spread past the 10-step window must NOT trip the limit.
    state = OomSkipState(max_in_window=3)
    state.record_skip(0)
    state.record_skip(5)
    # step 20 is >10 past step 0 and 5: only the step-20 OOM is in-window.
    state.record_skip(20)
    assert not state.at_limit(20)
    assert state.recent(20) == 1


def test_oom_skip_reset_window_clears_history():
    state = OomSkipState(max_in_window=3)
    state.record_skip(0)
    state.record_skip(1)
    state.reset_window()  # e.g. after a swap bump
    assert state.recent(1) == 0
    assert not state.at_limit(1)


def test_handle_oom_skip_zeros_grad():
    engine = type("E", (), {})()
    engine.optimizer = type("O", (), {"zeroed": False})()

    def zero_grad(set_to_none=True):
        engine.optimizer.zeroed = True

    engine.optimizer.zero_grad = zero_grad
    state = OomSkipState(max_in_window=3)
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
    handle_oom_skip(OomSkipState(max_in_window=3), engine, clear_cache=False)
    assert all(not t.started_ for t in engine.timers.timers.values())
