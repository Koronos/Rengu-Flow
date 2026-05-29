"""Tests for rengu_flow.optim.param_groups."""

import copy
import importlib

import torch

_param_groups = importlib.import_module("rengu_flow.optim.param_groups")
adjust_beta2_half_life = _param_groups.adjust_beta2_half_life
split_weight_decay_param_groups = _param_groups.split_weight_decay_param_groups


def test_adjust_beta2_half_life_recomputes_beta2():
    cfg = {"betas": [0.9, 0.999], "lr": 1e-4, "beta2_half_life": 100}
    out = adjust_beta2_half_life(copy.deepcopy(cfg), global_batch_size=8)
    assert "beta2_half_life" not in out
    assert out["betas"][0] == 0.9
    assert out["betas"][1] == 0.5 ** (8 / 100)


def test_adjust_beta2_half_life_no_key_unchanged():
    cfg = {"betas": [0.9, 0.999], "lr": 1e-4}
    out = adjust_beta2_half_life(copy.deepcopy(cfg), global_batch_size=8)
    assert out["betas"][1] == 0.999


def test_split_weight_decay_separates_1d_params():
    w = torch.nn.Parameter(torch.zeros(4, 4))
    b = torch.nn.Parameter(torch.zeros(4))
    groups = [{"params": [w, b], "lr": 1e-4, "weight_decay": 0.01}]
    result = split_weight_decay_param_groups(groups, "adamw")
    assert len(result) == 2
    wd_group = next(g for g in result if len(g["params"]) == 1 and g["params"][0] is w)
    no_wd = next(g for g in result if g["params"][0] is b)
    assert no_wd["weight_decay"] == 0


def test_split_weight_decay_genericoptim_disables_muon_on_no_wd():
    w = torch.nn.Parameter(torch.zeros(4, 4))
    b = torch.nn.Parameter(torch.zeros(4))
    groups = [{"params": [w, b], "muon": True, "weight_decay": 0.01}]
    result = split_weight_decay_param_groups(groups, "genericoptim")
    no_wd = next(g for g in result if g["params"][0] is b)
    assert no_wd["muon"] is False
    assert no_wd["adamuon"] is False
    assert no_wd["normuon"] is False
