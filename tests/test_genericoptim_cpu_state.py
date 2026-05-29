"""GenericOptim CPU offload state roundtrip (kahan_buffer_offload)."""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("transformers")

from rengu_flow.optim.resolver import resolve_optimizer_class


@pytest.fixture
def generic_optim_cls():
    return resolve_optimizer_class("genericoptim")


def test_kahan_buffer_offload_state_on_cpu_after_step(generic_optim_cls):
    param = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    param.grad = torch.randn_like(param)
    opt = generic_optim_cls([param], lr=1e-3, kahan_buffer_offload=True)
    with patch("torch.cuda.synchronize"):
        opt.step()
    state = opt.state[param]
    assert "shift" in state
    assert state["shift"].device.type == "cpu"


def test_kahan_buffer_offload_load_state_dict_keeps_cpu(generic_optim_cls):
    param = torch.nn.Parameter(torch.randn(3, 5, dtype=torch.bfloat16))
    param.grad = torch.randn_like(param)
    opt1 = generic_optim_cls([param], lr=1e-3, kahan_buffer_offload=True)
    with patch("torch.cuda.synchronize"):
        opt1.step()
    sd = opt1.state_dict()

    param2 = torch.nn.Parameter(torch.randn(3, 5, dtype=torch.bfloat16))
    param2.grad = torch.randn_like(param2)
    opt2 = generic_optim_cls([param2], lr=1e-3, kahan_buffer_offload=True)
    opt2.load_state_dict(sd)
    assert opt2.state[param2]["shift"].device.type == "cpu"

    param2.grad = torch.randn_like(param2)
    with patch("torch.cuda.synchronize"):
        opt2.step()
