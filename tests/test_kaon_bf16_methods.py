"""Kaon 0.7.15 bf16_method values through Rengu's optimizer resolution and form pre-fill.

Rengu forwards ``[optimizer]`` keys verbatim to the kaon constructor, so this pins that every
kaon alias pre-filling ``bf16_method`` accepts the compact Kahan methods on bf16 weights (and
allocates the residual Rengu's docs promise), and that ScheduleFree still rejects them.
"""

import pytest
import torch

kaon = pytest.importorskip("kaon")

from rengu_flow.registry.optimizers import OPTIMIZER_ALIASES, get_optimizer_class  # noqa: E402
from rengu_flow_ui.optim_kv_defaults import optimizer_extra_params_defaults  # noqa: E402

KAON_BF16_METHOD_TYPES = sorted(
    name
    for name, (module, _cls) in OPTIMIZER_ALIASES.items()
    if module == "kaon" and "bf16_method" in optimizer_extra_params_defaults(name)
)
RESIDUAL_DTYPE = {"kahan8": torch.uint8, "kahan16": torch.int16}


def _all_states(opt: torch.optim.Optimizer) -> list[dict]:
    states = list(opt.state.values())
    inner = getattr(opt, "inner", None)
    if isinstance(inner, torch.optim.Optimizer):
        states += list(inner.state.values())
    return states


def _two_steps(opt: torch.optim.Optimizer, p: torch.nn.Parameter, name: str) -> None:
    for _ in range(2):
        p.grad = torch.randn_like(p)
        if name == "sam":  # two-pass: the closure recomputes the gradient at the perturbed point

            def closure():
                p.grad = torch.randn_like(p)
                return torch.tensor(0.0)

            opt.step(closure)
        else:
            opt.step()


def test_kaon_bf16_method_types_cover_the_kaon_family() -> None:
    assert {"adakaon", "nekaon", "adapnm", "schedulefree", "lookahead"} <= set(
        KAON_BF16_METHOD_TYPES
    )
    assert all(
        optimizer_extra_params_defaults(n)["bf16_method"] == "stochastic_rounding"
        for n in KAON_BF16_METHOD_TYPES
    )


@pytest.mark.parametrize("method", ["kahan8", "kahan16"])
@pytest.mark.parametrize(
    "name", [n for n in KAON_BF16_METHOD_TYPES if n != "schedulefree"]
)
def test_compact_kahan_accepted_with_rengu_prefill(name: str, method: str) -> None:
    kwargs = {**optimizer_extra_params_defaults(name), "bf16_method": method}
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    opt = get_optimizer_class(name)([p], **kwargs)
    before = p.detach().clone()
    _two_steps(opt, p, name)

    residuals = [st["kahan_lo"] for st in _all_states(opt) if "kahan_lo" in st]
    assert residuals, f"{name}: no kahan_lo residual allocated for {method}"
    assert all(r.dtype == RESIDUAL_DTYPE[method] and r.shape == p.shape for r in residuals)
    assert torch.isfinite(p.float()).all()
    assert not torch.equal(p.detach(), before)


@pytest.mark.parametrize("method", ["kahan8", "kahan16"])
def test_schedulefree_rejects_compact_kahan(method: str) -> None:
    kwargs = {**optimizer_extra_params_defaults("schedulefree"), "bf16_method": method}
    p = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bf16_method"):
        get_optimizer_class("schedulefree")([p], **kwargs)
