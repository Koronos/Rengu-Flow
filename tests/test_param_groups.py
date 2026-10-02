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
    assert wd_group["weight_decay"] == 0.01
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


def test_split_weight_decay_exempts_dora_and_scalar_params():
    w = torch.nn.Parameter(torch.zeros(4, 4))
    dora = torch.nn.Parameter(torch.zeros(1, 4))
    dora.original_name = "blocks.0.attn.q_proj.lora_magnitude_vector.dora_scale"
    scalar = torch.nn.Parameter(torch.zeros(()))
    named = torch.nn.Parameter(torch.zeros(2, 2))
    named.original_name = "blocks.0.lokr_scalar"
    groups = [{"params": [w, dora, scalar, named], "weight_decay": 0.01}]
    result = split_weight_decay_param_groups(groups, "adamw")
    wd = [p for g in result if g["weight_decay"] == 0.01 for p in g["params"]]
    no_wd = [p for g in result if g["weight_decay"] == 0 for p in g["params"]]
    assert wd == [w]
    assert {id(p) for p in no_wd} == {id(dora), id(scalar), id(named)}


def test_split_genericoptim_proj_keys_reach_every_group():
    groups = [
        {"params": [torch.nn.Parameter(torch.zeros(4, 4))], "lr": 1e-4},
        {"params": [torch.nn.Parameter(torch.zeros(4, 4))], "lr": 2e-4},
    ]
    kwargs = {"rank": 8, "proj_type": "std", "update_proj_gap": 100, "other": 1}
    out = _param_groups.split_genericoptim_param_groups(groups, kwargs)
    twod = [g for g in out if g["params"] and g["params"][0].ndim == 2]
    assert len(twod) == 2
    assert all(g["rank"] == 8 and g["proj_type"] == "std" for g in twod)
    assert all(g["update_proj_gap"] == 100 for g in twod)
    assert kwargs == {"other": 1}
