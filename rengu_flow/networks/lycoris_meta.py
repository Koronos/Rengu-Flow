"""Catalog of LyCORIS library algorithms exposed as rengu adapter types.

Torch-free on purpose: config validation/defaults and the web UI schema import this
module without pulling torch or the lycoris package. Everything here mirrors
``lycoris.wrapper`` (v3.4.0) — ``create_lycoris`` kwargs, per-module ``weight_list``
key families and ``custom_state_dict`` behavior. ``full`` is excluded (upstream
FullModule.apply_to deletes the weight its org_forward still needs) and ``ia3`` is
excluded (registered in the config SDK but absent from ``network_module_dict``).
"""

# rengu adapter.type -> lycoris algo name (create_lycoris ``algo=``)
ALGO_MAP = {
    "lycoris_locon": "locon",
    "lycoris_loha": "loha",
    "lycoris_lokr": "lokr",
    # DoRA = LoCon with weight decomposition; first-class type for discoverability.
    "lycoris_dora": "locon",
    "lycoris_dylora": "dylora",
    "lycoris_glora": "glora",
    "lycoris_diag_oft": "diag-oft",
    "lycoris_boft": "boft",
}

LYCORIS_ADAPTER_TYPES = tuple(ALGO_MAP)

# kwargs forced onto create_lycoris regardless of config (not user-settable).
FORCED_ALGO_KWARGS = {
    "lycoris_dora": {"dora_wd": True},
}

# Keys every adapter table may carry regardless of algorithm. ``alpha`` is accepted
# by validation but rejected by defaults with the alpha=rank message (same rule as
# lora/lokr); ``dim`` is the Kohya-style alias normalized to ``rank``.
COMMON_CONFIG_KEYS = frozenset(
    {
        "type",
        "rank",
        "dim",
        "alpha",
        "dtype",
        "train_conv",
        "init_from_existing",
        # Module targeting: fnmatch globs against the full dotted module path
        # (e.g. "unet.down_blocks.0...attn1.to_q"); absent include = all modules.
        "target_include",
        "target_exclude",
    }
)

# Config keys consumed by rengu's attach layer itself, never forwarded to
# create_lycoris (rs_lora is applied as a post-init override — create_lycoris
# does not forward it to the modules).
ATTACH_ONLY_KEYS = frozenset({"rs_lora"})

# Per-type tunables passed through to create_lycoris. Spelling follows the
# create_lycoris kwargs (DoRA toggle is ``dora_wd``, not ``weight_decompose``).
_DROPOUTS = ("dropout", "rank_dropout", "module_dropout")
# train_norm is network-level (NormModule on LayerNorm/GroupNorm), available on
# every algorithm; rs_lora only exists on LoConModule (locon/dora).
ALGO_CONFIG_KEYS = {
    "lycoris_locon": (
        *_DROPOUTS,
        "train_norm",
        "use_tucker",
        "use_scalar",
        "dora_wd",
        "wd_on_output",
        "rs_lora",
    ),
    "lycoris_loha": (*_DROPOUTS, "train_norm", "use_tucker", "use_scalar", "dora_wd", "wd_on_output"),
    "lycoris_lokr": (
        *_DROPOUTS,
        "train_norm",
        "use_tucker",
        "use_scalar",
        "dora_wd",
        "wd_on_output",
        "factor",
        "full_matrix",
        "decompose_both",
        "unbalanced_factorization",
    ),
    # dora_wd is implied by the type itself; offering it would invite dora_wd=false.
    "lycoris_dora": (*_DROPOUTS, "train_norm", "use_tucker", "use_scalar", "wd_on_output", "rs_lora"),
    "lycoris_dylora": (*_DROPOUTS, "train_norm", "block_size"),
    "lycoris_glora": (*_DROPOUTS, "train_norm"),
    "lycoris_diag_oft": (*_DROPOUTS, "train_norm", "constraint", "rescaled"),
    "lycoris_boft": (*_DROPOUTS, "train_norm", "constraint", "rescaled"),
}

# Defaults mirror create_lycoris (wrapper.py): dropouts 0.0, toggles off,
# wd_on_output True, lokr factor -1, dylora block_size 4, oft/boft constraint 0.0.
_SHARED_DEFAULTS = {
    "dropout": 0.0,
    "rank_dropout": 0.0,
    "module_dropout": 0.0,
    "train_norm": False,
}
ALGO_CONFIG_DEFAULTS = {
    "lycoris_locon": {
        **_SHARED_DEFAULTS,
        "use_tucker": False,
        "use_scalar": False,
        "dora_wd": False,
        "wd_on_output": True,
        "rs_lora": False,
    },
    "lycoris_loha": {
        **_SHARED_DEFAULTS,
        "use_tucker": False,
        "use_scalar": False,
        "dora_wd": False,
        "wd_on_output": True,
    },
    "lycoris_lokr": {
        **_SHARED_DEFAULTS,
        "use_tucker": False,
        "use_scalar": False,
        "dora_wd": False,
        "wd_on_output": True,
        "factor": -1,
        "full_matrix": False,
        "decompose_both": False,
        "unbalanced_factorization": False,
    },
    "lycoris_dora": {
        **_SHARED_DEFAULTS,
        "use_tucker": False,
        "use_scalar": False,
        "wd_on_output": True,
        "rs_lora": False,
    },
    "lycoris_dylora": {**_SHARED_DEFAULTS, "block_size": 4},
    "lycoris_glora": dict(_SHARED_DEFAULTS),
    "lycoris_diag_oft": {**_SHARED_DEFAULTS, "constraint": 0.0, "rescaled": False},
    "lycoris_boft": {**_SHARED_DEFAULTS, "constraint": 0.0, "rescaled": False},
}

# Expected per-module key suffixes in the exported kohya-flat file, mirroring each
# module's custom_state_dict / weight_list. ``delta`` is the tensor that must show a
# nonzero training delta (zero- or identity-initialized upstream, so any nonzero
# value proves the optimizer touched it). ``either`` lists groups where exactly one
# member family appears (LoKr's full-vs-factored W1/W2).
WEIGHT_KEY_FAMILIES = {
    "lycoris_locon": {
        "required": ("lora_up.weight", "lora_down.weight", "alpha"),
        "optional": ("lora_mid.weight", "dora_scale"),
        "either": (),
        "delta": "lora_up.weight",
    },
    "lycoris_dora": {
        "required": ("lora_up.weight", "lora_down.weight", "alpha", "dora_scale"),
        "optional": ("lora_mid.weight",),
        "either": (),
        "delta": "lora_up.weight",
    },
    "lycoris_loha": {
        "required": ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "alpha"),
        "optional": ("hada_t1", "hada_t2", "dora_scale"),
        "either": (),
        "delta": "hada_w1_a",
    },
    "lycoris_lokr": {
        "required": ("alpha",),
        "optional": ("lokr_t2", "dora_scale"),
        "either": (
            (("lokr_w1",), ("lokr_w1_a", "lokr_w1_b")),
            (("lokr_w2",), ("lokr_w2_a", "lokr_w2_b")),
        ),
        "delta": ("lokr_w2", "lokr_w2_b"),
    },
    "lycoris_dylora": {
        "required": ("lora_up.weight", "lora_down.weight", "alpha"),
        "optional": (),
        "either": (),
        "delta": "lora_up.weight",
    },
    "lycoris_glora": {
        "required": ("a1.weight", "a2.weight", "b1.weight", "b2.weight", "alpha"),
        "optional": ("bm.weight",),
        "either": (),
        "delta": "b2.weight",
    },
    "lycoris_diag_oft": {
        "required": ("oft_blocks", "alpha"),
        "optional": ("rescale",),
        "either": (),
        "delta": "oft_blocks",
    },
    "lycoris_boft": {
        "required": ("oft_blocks", "alpha"),
        "optional": ("rescale",),
        "either": (),
        "delta": "oft_blocks",
    },
}

# train_norm attaches NormModule to LayerNorm/GroupNorm layers regardless of the
# algorithm; its exported modules carry only these keys (no alpha entry).
NORM_MODULE_FAMILY = {
    "required": ("w_norm",),
    "optional": ("b_norm",),
    "either": (),
    "delta": "w_norm",
}

# When use_scalar=True the trained ``scalar`` is folded into these tensors at export
# (exactly what each module's custom_state_dict does); for LoKr only whichever W1
# family is present gets folded.
SCALAR_FOLD_TARGETS = {
    "locon": ("lora_up.weight",),
    "loha": ("hada_w1_a",),
    "lokr": ("lokr_w1", "lokr_w1_a"),
    "glora": ("a2.weight", "b2.weight"),
}

# Exported ``.alpha`` per module: OFT family stores the constraint there (upstream
# registers alpha = torch.tensor(constraint)); everything else stores rank.
ALPHA_IS_CONSTRAINT = ("lycoris_diag_oft", "lycoris_boft")


def is_lycoris_type(adapter_type: str) -> bool:
    return isinstance(adapter_type, str) and adapter_type.startswith("lycoris_")


def allowed_config_keys(adapter_type: str) -> frozenset[str]:
    """Full set of [adapter] keys valid for this lycoris type."""
    return COMMON_CONFIG_KEYS | set(ALGO_CONFIG_KEYS[adapter_type])


def collect_lycoris_adapter_issues(adapter: dict) -> list[str]:
    """Validation issues for a lycoris [adapter] table (type already known to be lycoris_*)."""
    adapter_type = adapter["type"]
    issues = []
    if adapter_type not in LYCORIS_ADAPTER_TYPES:
        known = ", ".join(f"`{t}`" for t in LYCORIS_ADAPTER_TYPES)
        return [f"adapter.type {adapter_type!r} is not a known LyCORIS type. Available: {known}."]
    unknown = set(adapter) - allowed_config_keys(adapter_type)
    if unknown:
        allowed = ", ".join(sorted(ALGO_CONFIG_KEYS[adapter_type]))
        issues.append(
            f"[adapter] keys not supported by {adapter_type}: "
            f"{', '.join(sorted(unknown))}. Tunables for this type: {allowed}."
        )
    if adapter_type == "lycoris_dylora" and adapter.get("train_conv"):
        # Upstream DyLoraModule allocates 2D block lists from (out, in) only; full
        # conv kernels don't round-trip through its state dict.
        issues.append("lycoris_dylora does not support train_conv = true.")
    for key in ("target_include", "target_exclude"):
        patterns = adapter.get(key)
        if patterns is None:
            continue
        if (
            not isinstance(patterns, list)
            or not patterns
            or not all(isinstance(p, str) and p.strip() for p in patterns)
        ):
            issues.append(
                f"adapter.{key} must be a non-empty list of glob patterns "
                f'(e.g. ["*attn*"]); omit it to match all modules.'
            )
    rank = adapter.get("rank", adapter.get("dim"))
    if adapter_type == "lycoris_dylora" and rank is not None:
        block_size = adapter.get("block_size", ALGO_CONFIG_DEFAULTS["lycoris_dylora"]["block_size"])
        if isinstance(rank, int) and isinstance(block_size, int) and block_size > 0 and rank % block_size:
            issues.append(
                f"lycoris_dylora needs adapter.rank divisible by block_size "
                f"(rank={rank}, block_size={block_size})."
            )
    return issues


def apply_lycoris_defaults(adapter_config: dict) -> None:
    """Fill per-algo defaults in-place (rank/alpha/dtype handled by the caller)."""
    for key, value in ALGO_CONFIG_DEFAULTS[adapter_config["type"]].items():
        adapter_config.setdefault(key, value)
    adapter_config.setdefault("train_conv", False)


def create_lycoris_kwargs(adapter_config: dict) -> tuple[str, dict]:
    """Map a defaulted adapter config to (algo, kwargs) for ``create_lycoris``."""
    adapter_type = adapter_config["type"]
    kwargs = {
        key: adapter_config[key]
        for key in ALGO_CONFIG_KEYS[adapter_type]
        if key in adapter_config and key not in ATTACH_ONLY_KEYS
    }
    kwargs.update(FORCED_ALGO_KWARGS.get(adapter_type, {}))
    return ALGO_MAP[adapter_type], kwargs
