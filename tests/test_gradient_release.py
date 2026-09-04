"""Tests for GradientReleaseOptimizerWrapper."""

import torch

from rengu_flow.vendor.diffusion_pipe_optimizers.gradient_release import (
    GradientReleaseOptimizerWrapper,
)


class _LookaheadishOptimizer:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.param_groups = [{"params": []}]

    def eval(self) -> "_LookaheadishOptimizer":  # noqa: A003
        self.events.append("eval")
        return self

    def train(self) -> "_LookaheadishOptimizer":
        self.events.append("train")
        return self


class _PlainOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"params": []}]


def test_gradient_release_wrapper_state_dict_roundtrip():
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(3))
    o1 = torch.optim.SGD([p1], lr=0.1)
    o2 = torch.optim.SGD([p2], lr=0.2)
    wrapper = GradientReleaseOptimizerWrapper([o1, o2])
    assert len(wrapper.param_groups) == 2
    sd = wrapper.state_dict()
    wrapper.load_state_dict(sd)
    wrapper.step()
    wrapper.zero_grad()


def test_gradient_release_wrapper_forwards_eval_train():
    lookahead = _LookaheadishOptimizer()
    plain = _PlainOptimizer()
    wrapper = GradientReleaseOptimizerWrapper([lookahead, plain])

    assert callable(wrapper.eval) and callable(wrapper.train)
    assert wrapper.eval() is wrapper
    assert lookahead.events == ["eval"]

    assert wrapper.train() is wrapper
    assert lookahead.events == ["eval", "train"]


def test_gradient_release_wrapper_eval_train_noop_without_lookahead():
    wrapper = GradientReleaseOptimizerWrapper([_PlainOptimizer(), _PlainOptimizer()])
    assert wrapper.eval() is wrapper
    assert wrapper.train() is wrapper
