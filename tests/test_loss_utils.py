"""Tests for renga_flow.model.loss_utils."""

import torch

import importlib.util

_loss_utils = importlib.import_module("renga_flow.model.loss_utils")
compute_diffusion_loss_per_element = _loss_utils.compute_diffusion_loss_per_element


def test_mse_default():
    out = torch.tensor([1.0, 2.0])
    tgt = torch.tensor([1.0, 0.0])
    loss = compute_diffusion_loss_per_element(out, tgt, {})
    assert loss.shape == out.shape
    assert loss[1].item() == 4.0


def test_huber_delta():
    out = torch.zeros(2)
    tgt = torch.ones(2)
    loss = compute_diffusion_loss_per_element(out, tgt, {"huber_delta": 1.0})
    assert (loss >= 0).all()


def test_pseudo_huber_c_legacy():
    out = torch.zeros(1)
    tgt = torch.ones(1)
    loss = compute_diffusion_loss_per_element(out, tgt, {"pseudo_huber_c": 0.1})
    assert loss.shape == (1,)
