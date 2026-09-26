"""Kaon 0.7.15/0.7.16 bf16_method values through Rengu's optimizer resolution and form pre-fill.

Rengu forwards ``[optimizer]`` keys verbatim to the kaon constructor, so this pins that every
kaon alias pre-filling ``bf16_method`` accepts the compact Kahan methods on bf16 weights (and
allocates the residual Rengu's docs promise), and that ScheduleFree still rejects them.

0.7.16 additions: Adakaon's kahan16 update (weight decay + gradient centralization + cautious)
is bit-exact vs. an equivalent fp32-weight run; Nekaon is fully usable with kahan8/kahan16
(weight decay and GC read the corrected full-precision value); and the streaming fp32 export
(``kaon.decode_weights`` / ``kaon.full_precision_state_dict``) round-trips through
``opt.eval()``/``opt.train()`` and a save/load cycle.

0.7.17: on the fused CUDA path, a param whose grad dtype doesn't match its weight dtype
(``p.grad_dtype = None`` plus an off-dtype grad, e.g. fp32 grad on a bf16 weight) used to
produce NaN/garbage; that param now falls back to the native path for that step instead.
"""

import io

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


def test_adakaon_kahan16_bitexact_with_wd_gc_cautious() -> None:
    """0.7.16: kahan16 keeps weight decay, gradient centralization and cautious masking
    reading the corrected full-precision value, matching an fp32-weight run bit for bit."""
    torch.manual_seed(0)
    # Start from a value already representable in bf16, so the bf16+residual pair encodes
    # it exactly — the "exact vs fp32" guarantee is about the update math, not recovering
    # precision lost before kaon ever sees the weight.
    base = torch.randn(16, 8).to(torch.bfloat16)
    p_fp32 = torch.nn.Parameter(base.float())
    p_bf16 = torch.nn.Parameter(base.clone())
    p_bf16.grad_dtype = None  # allow assigning the same fp32-valued grad as the fp32 run

    common = dict(
        lr=1e-3, weight_decay=0.05, cautious=True, gradient_centralization=True,
        momentum_dtype="bfloat16", fused=False, foreach=False,
    )
    opt_fp32 = kaon.Adakaon([p_fp32], bf16_method="stochastic_rounding", **common)
    opt_bf16 = kaon.Adakaon([p_bf16], bf16_method="kahan16", **common)

    for _ in range(5):
        grad = torch.randn_like(p_fp32)
        p_fp32.grad = grad.clone()
        p_bf16.grad = grad.clone()
        opt_fp32.step()
        opt_bf16.step()

    decoded = kaon.decode_weights(opt_bf16)[p_bf16]
    assert torch.equal(decoded, p_fp32.detach())


@pytest.mark.parametrize("method", ["kahan8", "kahan16"])
def test_nekaon_usable_with_compact_kahan_wd_and_gc(method: str) -> None:
    """0.7.16: Nekaon (wd=0.1, cautious=True by Rengu's own defaults) through the wrapper's
    lookahead/climb. Given the same bf16 gradients, kahan16 is bit-exact vs. an fp32-weight
    twin with the same kwargs; kahan8 stays much closer than a plain bf16 rounding would.

    Both runs are compared in the SAME view (eval): Nekaon's train-mode weight sits at the
    lookahead point, ~k*lr away from the true weight, so decoding the bf16 run in eval against
    the fp32 twin still in train measures that gap (1.5e-4 = k*lr with Rengu's defaults), not
    kahan's error."""
    kwargs = optimizer_extra_params_defaults("nekaon")
    assert kwargs["weight_decay"] > 0 and kwargs["cautious"] is True

    torch.manual_seed(0)
    base = torch.randn(16, 8).to(torch.bfloat16)
    p_fp32 = torch.nn.Parameter(base.float())
    p_bf16 = torch.nn.Parameter(base.clone())

    # Same kwargs on both (bf16_method is inert on fp32 weights).
    opt_fp32 = get_optimizer_class("nekaon")([p_fp32], **kwargs)
    opt_bf16 = get_optimizer_class("nekaon")([p_bf16], **{**kwargs, "bf16_method": method})

    for _ in range(5):
        grad = torch.randn_like(p_bf16)  # bf16, as a real backward yields on a bf16 weight
        p_fp32.grad = grad.float()
        p_bf16.grad = grad.clone()
        opt_fp32.step()
        opt_bf16.step()

    opt_fp32.eval()
    opt_bf16.eval()
    decoded = kaon.decode_weights(opt_bf16)[p_bf16]
    reference = p_fp32.detach().clone()
    opt_bf16.train()
    opt_fp32.train()
    assert torch.isfinite(decoded).all()
    if method == "kahan16":
        assert torch.equal(decoded, reference)
    else:
        # Much tighter than a plain bf16 round-to-nearest (~half a bf16 ulp, ~2e-3 at this scale).
        assert (decoded - reference).abs().max().item() < 5e-4


def test_nekaon_kahan16_fp32_export_survives_save_load_eval_train() -> None:
    """0.7.16 fp32 streaming export: kaon.full_precision_state_dict(model, opt, device="cpu")
    round-trips through opt.eval()/opt.train() and a torch.save/load cycle."""
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4, bias=False).to(torch.bfloat16)
    kwargs = {**optimizer_extra_params_defaults("nekaon"), "bf16_method": "kahan16"}
    opt = get_optimizer_class("nekaon")(model.parameters(), **kwargs)

    for _ in range(3):
        model.weight.grad = torch.randn_like(model.weight)
        opt.step()

    opt.eval()  # Nekaon's train-mode view sits at the lookahead point; decode refuses there
    fp32_sd = kaon.full_precision_state_dict(model, opt, device="cpu")
    opt.train()

    assert fp32_sd["weight"].dtype == torch.float32
    assert torch.isfinite(fp32_sd["weight"]).all()

    buf = io.BytesIO()
    torch.save(fp32_sd, buf)
    buf.seek(0)
    reloaded = torch.load(buf, weights_only=True)
    assert torch.equal(reloaded["weight"], fp32_sd["weight"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="0.7.17 fused-path fix is CUDA-only")
def test_nekaon_fused_mismatched_grad_dtype_stays_finite() -> None:
    """0.7.17: a bf16 weight with an off-dtype (fp32) grad on the fused CUDA path used to
    produce NaN/garbage; that param now falls back to the native path for that step."""
    torch.manual_seed(0)
    device = "cuda"
    kwargs = {**optimizer_extra_params_defaults("nekaon"), "fused": True, "bf16_method": "kahan16"}
    p = torch.nn.Parameter(torch.randn(16, 8, device=device, dtype=torch.bfloat16))
    p.grad_dtype = None  # allow an fp32 grad on this bf16 param
    opt = get_optimizer_class("nekaon")([p], **kwargs)

    for _ in range(3):
        p.grad = torch.randn(16, 8, device=device, dtype=torch.float32)
        opt.step()

    assert torch.isfinite(p.detach()).all()
