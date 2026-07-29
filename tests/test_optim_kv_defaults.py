"""Tests for optimizer/scheduler KV default maps."""

from rengu_flow_ui.optim_kv_defaults import (
    OPTIMIZER_REGISTRY_KV_DEFAULTS,
    SCHEDULER_RUNTIME_TOKEN_HINTS,
    SCHEDULER_RUNTIME_TOKENS,
    optimizer_extra_params_defaults,
    scheduler_kv_defaults,
)
from rengu_flow_ui.optimizer_form import defaults_for_optimizer_type_change
from rengu_flow_ui.scheduler_form import defaults_for_scheduler_type_change


def test_every_registry_optimizer_has_kv_defaults() -> None:
    """Each selectable optimizer must prefill non-empty defaults (no silent empty form)."""
    from rengu_flow.registry.optimizers import (
        OPTIMIZER_ALIASES,
        VENDOR_OPTIMIZER_ALIASES,
        optimizer_registry,
    )

    registry_names = set(optimizer_registry) | set(OPTIMIZER_ALIASES) | set(VENDOR_OPTIMIZER_ALIASES)
    missing = {n for n in registry_names if not OPTIMIZER_REGISTRY_KV_DEFAULTS.get(n)}
    assert not missing, f"optimizers in the registry without KV defaults: {sorted(missing)}"


def test_optimizer_kv_defaults_exposed_in_registries() -> None:
    """The schema must ship the KV defaults so the frontend prefills from the backend."""
    from rengu_flow_ui.config_schema import get_registries

    reg = get_registries()
    kv = reg["optimizer_kv_defaults"]
    assert set(reg["optimizers"]).issubset(set(kv))
    assert kv["nekaon"]["lr"] == 1e-4  # a kaon optimizer the old FE list missed


def test_every_registry_scheduler_has_kv_defaults() -> None:
    """Each built-in scheduler must have a KV-defaults entry (``none`` legitimately empty)."""
    from rengu_flow.optim.resolver import scheduler_registry
    from rengu_flow_ui.optim_kv_defaults import SCHEDULER_BUILTIN_KV_DEFAULTS

    missing = set(scheduler_registry) - set(SCHEDULER_BUILTIN_KV_DEFAULTS)
    assert not missing, f"schedulers in the registry without a KV-defaults entry: {sorted(missing)}"


def test_scheduler_kv_defaults_exposed_in_registries() -> None:
    """The schema must ship scheduler KV defaults (builtin + suggested FQNs)."""
    from rengu_flow_ui.config_schema import get_registries

    reg = get_registries()
    assert reg["scheduler_kv_defaults"]["cosine"] == {"lr_min": 0.0}
    fqn = "torch.optim.lr_scheduler.CosineAnnealingLR"
    assert reg["scheduler_fqn_kv_defaults"][fqn]["T_max"] == "effective_total_steps"


def test_scheduler_runtime_token_hints_cover_all_tokens() -> None:
    assert set(SCHEDULER_RUNTIME_TOKEN_HINTS) == set(SCHEDULER_RUNTIME_TOKENS)
    assert "min(total_steps" in SCHEDULER_RUNTIME_TOKEN_HINTS["effective_total_steps"]


def test_optimizer_adamw_defaults_include_lr_and_betas() -> None:
    kv = optimizer_extra_params_defaults("adamw")
    assert kv["lr"] == 1e-4
    assert kv["betas"] == [0.9, 0.999]


def test_optimizer_genericoptim_defaults() -> None:
    kv = optimizer_extra_params_defaults("genericoptim")
    assert kv["muon"] is False
    assert "adamuon" in kv
    assert kv["lr"] == 1e-4


def test_new_kaon_optimizer_defaults() -> None:
    """KProdigy, Adakaon, Lion and AdaPNM expose their supported Kaon controls."""
    kprodigy = optimizer_extra_params_defaults("kprodigy")
    assert kprodigy["lr"] == 1.0  # parameter-free: train at lr=1.0
    assert kprodigy["d_coef"] == 1.0

    adakaon = optimizer_extra_params_defaults("adakaon")
    assert adakaon["auto_lr"] is False

    lion = optimizer_extra_params_defaults("lion")
    assert lion["lr"] == 2e-4
    assert lion["betas"] == [0.95, 0.98]  # classic Lion betas

    adapnm = optimizer_extra_params_defaults("adapnm")
    assert adapnm["betas"] == [0.8, 0.999]
    assert adapnm["beta0"] == 0.5


def test_fused_flag_prefilled_for_fused_capable_kaon_optimizers() -> None:
    """Kaon's accelerated optimizers enable their native-Windows Triton path by default."""
    assert optimizer_extra_params_defaults("adakaon")["fused"] is True
    assert optimizer_extra_params_defaults("adapnm")["fused"] is True
    assert optimizer_extra_params_defaults("nekaon")["fused"] is True


def test_newer_kaon_optimizer_defaults() -> None:
    """adabelief/adamp/adopt/schedulefree/lookahead/sam pre-fill sensible kaon defaults."""
    assert optimizer_extra_params_defaults("adabelief")["betas"] == [0.9, 0.999]
    assert optimizer_extra_params_defaults("adamp")["cautious"] is True

    adopt = optimizer_extra_params_defaults("adopt")
    assert adopt["betas"] == [0.9, 0.9999]  # high beta2 is intentional for ADOPT

    schedulefree = optimizer_extra_params_defaults("schedulefree")
    assert schedulefree["lr"] == 2.5e-3
    assert schedulefree["warmup_steps"] == 0

    lookahead = optimizer_extra_params_defaults("lookahead")
    assert lookahead["k"] == 5
    assert lookahead["alpha"] == 0.5

    assert optimizer_extra_params_defaults("sam")["rho"] == 0.05


def test_newest_kaon_optimizer_defaults() -> None:
    """msam/nekaon (kaon 0.4.0) pre-fill sensible kaon defaults."""
    msam = optimizer_extra_params_defaults("msam")
    assert msam["rho"] == 0.3
    assert msam["lr"] == 1e-4

    nekaon = optimizer_extra_params_defaults("nekaon")
    assert nekaon["k"] == 1.5
    assert nekaon["betas"] == [0.5, 0.999]  # beta1 regime knob, must be > 0
    assert nekaon["weight_decay"] == 0.1
    assert nekaon["momentum_dtype"] == "4bit"  # Nekaon's deliberate default (not Adakaon's bf16)


def test_scheduler_cosine_fqn_defaults_use_tokens() -> None:
    kv = scheduler_kv_defaults("torch.optim.lr_scheduler.CosineAnnealingLR")
    assert kv["T_max"] == "effective_total_steps"
    assert kv["eta_min"] == 0.0
    assert "warmup_steps" not in kv


def test_scheduler_builtin_cosine_defaults() -> None:
    kv = scheduler_kv_defaults("cosine")
    assert kv["lr_min"] == 0.0
    assert "warmup_steps" not in kv


def test_scheduler_builtin_rex_defaults() -> None:
    kv = scheduler_kv_defaults("rex")
    assert kv == {"lr_min": 0.0, "rex_d": 0.5}


def test_scheduler_none_has_no_warmup() -> None:
    assert scheduler_kv_defaults("none") == {}


def test_type_change_includes_kv_defaults() -> None:
    out = defaults_for_optimizer_type_change("genericoptim")
    assert out["optimizer.extra_params"] == OPTIMIZER_REGISTRY_KV_DEFAULTS["genericoptim"]

    sched = defaults_for_scheduler_type_change("torch.optim.lr_scheduler.StepLR")
    assert sched["lr_scheduler_args.extra_params"] == scheduler_kv_defaults(
        "torch.optim.lr_scheduler.StepLR"
    )

    linear = defaults_for_scheduler_type_change("linear")
    assert linear["lr_scheduler_args.extra_params"] == scheduler_kv_defaults("linear")
