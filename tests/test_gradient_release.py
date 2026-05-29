"""Tests for GradientReleaseOptimizerWrapper."""

import torch

from rengu_flow.vendor.diffusion_pipe_optimizers.gradient_release import (
    GradientReleaseOptimizerWrapper,
)


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
