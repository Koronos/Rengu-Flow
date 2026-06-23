# Adding optimizers and schedulers (developer guide)

This document describes where optimizer and scheduler resolution lives, how resolution works (registry + qualified path), and how to **register** a custom optimizer or scheduler so it can be referenced by name in TOML without editing the framework core.

## Code locations

- **Optimizer registry**: `rengu_flow/registry/optimizers.py`
  - `optimizer_registry`: dict mapping name (lowercase) → optimizer **class**
  - `register_optimizer(name)` decorator
  - `get_optimizer_class(optim_type)` — registry lookup first, then fully-qualified path
- **Scheduler registry and resolution**: `rengu_flow/optim/resolver.py`
  - `scheduler_registry`: dict mapping name (lowercase) → **factory** `(optimizer, config, total_steps, steps_per_epoch) -> LRScheduler | None`
  - `register_scheduler(name)` decorator
  - `resolve_scheduler(...)` — registry lookup first, then fully-qualified path
  - `build_scheduler_runtime_values(config, total_steps=…, steps_per_epoch=…)` and `substitute_runtime_tokens(kwargs, runtime_values)` for placeholders in `[lr_scheduler_args]` (`total_steps`, `effective_total_steps`, `steps_per_epoch`, `epochs`, `max_steps`, `gradient_accumulation_steps`)
  - `apply_warmup(optimizer, scheduler, warmup_steps)` — wraps scheduler with LinearLR warmup

The training entry point (`rengu_flow/main.py`) uses `resolve_optimizer_class(config["optimizer"]["type"])` and `resolve_scheduler(lr_scheduler, optimizer, config, ...)` from `rengu_flow.optim`; it does not call the registry modules directly.

## Resolution order

1. **Optimizer**: `get_optimizer_class(optim_type)` (used by `resolve_optimizer_class`)  
   - Look up `optim_type.lower()` in `optimizer_registry`.  
   - If not found and `"." in optim_type`, treat as `module.path.ClassName` and load via `importlib`.  
   - Otherwise raise `ValueError`.

2. **Scheduler**: `resolve_scheduler(scheduler_type, optimizer, config, total_steps, steps_per_epoch)`  
   - Look up `scheduler_type.lower()` in `scheduler_registry`; if found, call the factory.  
   - If not found and `"." in scheduler_type`, load the class from the path, read `config["lr_scheduler_args"]`, substitute runtime tokens, and instantiate with `(optimizer, **scheduler_kwargs)`.  
   - Otherwise raise `ValueError`.

So **qualified path always works** even if you never register a name: e.g. `type = "torch.optim.AdamW"` or `lr_scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"`.

## Registering a custom optimizer

1. Import the optimizer **class** (must be a subclass of `torch.optim.Optimizer`).
2. Use the decorator with the name you want in TOML (case-insensitive):

```python
from rengu_flow.registry.optimizers import register_optimizer
import torch

@register_optimizer("my_custom_optim")
class MyOptimizer(torch.optim.Optimizer):
    # ... implement step(), __init__, etc.
    pass
```

Or register an existing class:

```python
from rengu_flow.registry.optimizers import register_optimizer
import torch

register_optimizer("adamw_alt")(torch.optim.AdamW)
```

Then in TOML: `[optimizer]` / `type = "my_custom_optim"` (or `"adamw_alt"`). Ensure your package or module is imported before the training entry point runs (e.g. in `main.py` or via a plugin mechanism) so the registration happens.

## Registering a custom scheduler

The scheduler registry stores **factories**, not classes. Each factory has the signature:

```python
def factory(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_steps: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    ...
```

1. Implement a function with that signature that returns an `LRScheduler` (or `None` for “no scheduler”).
2. Register it with `register_scheduler("name")`:

```python
from rengu_flow.optim.resolver import register_scheduler
import torch

def _my_cosine(optimizer, config, total_steps, steps_per_epoch):
    lr_min = config.get("lr_scheduler_args", {}).get("lr_min", 0.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr_min
    )

register_scheduler("my_cosine")(_my_cosine)
```

Then in TOML: `lr_scheduler = "my_cosine"`. Runtime tokens (`total_steps`, etc.) are available in `config`; your factory can read `config["lr_scheduler_args"]` and use `substitute_runtime_tokens` from `rengu_flow.optim.resolver` if you need to pass token strings to a class constructor.

## Built-ins

- **Optimizers** (in `registry/optimizers.py`): `adamw`, `sgd`, `adam` — `torch.optim`; `genericoptim`, `automagic`, `adamw8bitkahan` — lazy-loaded from `rengu_flow/vendor/diffusion_pipe_optimizers/` (see NOTICE there).
- **Optional aliases** (lazy import, `[optim]` extra): `adamw8bit`, `adamw_optimi`, `stableadamw`, `offload`, `prodigy` — see `OPTIMIZER_ALIASES` in `registry/optimizers.py`.
- **kaon aliases** (lazy import, git-backed `kaon` profile): `adakaon`, `adamuon`, `kprodigy`, `autokaon`, `lion`, `adapnm`, `adabelief`, `adamp`, `adopt`, `schedulefree`, `lookahead`, `sam`, `msam`, `nekaon` — also in `OPTIMIZER_ALIASES` (module `"kaon"`). Selecting one routes to the `kaon` install profile (`install/manager.py` derives the trigger set from these aliases).
- **Fallback:** names without `.` are also looked up on the `pytorch_optimizer` package by class name when not registered as an alias.
- **Schedulers** (in `optim/resolver.py`): `constant`, `linear`, `cosine`, `rex` (REX reflected-exponential, custom `RexLR` class), `none`.

## Third-party optimizers (vendored)

Copied from [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) under `rengu_flow/vendor/diffusion_pipe_optimizers/`. See `README.md` and `NOTICE.md` in that folder for upstream credits and source commit.

Do not import that package at module load for all optimizers — `get_optimizer_class` lazy-imports vendor modules so tests and minimal installs avoid `bitsandbytes` / `optimum.quanto` until needed.

## Eval/train state (lookahead optimizers)

Optimizers that expose `eval()`/`train()` (kaon **Nekaon, MSAM, ScheduleFree, Lookahead**) keep the
between-step live weights displaced from the true iterate; only `eval()` restores the weights you
want to read. Every path that **reads weights for measurement or persistence** must bracket the read
with `optimizer.eval()` / `optimizer.train()`: previews (`utils/preview.py`), eval
(`utils/eval.py`), and both save paths (`Saver._persist_at_true_iterate`, see
`checkpoint-and-save.md`). A new optimizer with this property works automatically; a new
weight-reading code path must add the bracket or it will read/persist displaced weights.

## Training helpers

- **`rengu_flow/optim/param_groups.py`**: `adjust_beta2_half_life`, `split_weight_decay_param_groups`, `split_genericoptim_param_groups`.
- **`rengu_flow/utils/training_metrics.py`**: `log_training_step`, `get_prodigy_d`, `get_automagic_lrs` — used from `main.py`.

Adding more built-in names can be done via `register_optimizer` or by extending `VENDOR_OPTIMIZER_ALIASES` / `OPTIMIZER_ALIASES` in `registry/optimizers.py`.

## Config form (optimizer / scheduler KV)

The training config UI uses a single key-value list per section (parity between optimizer and scheduler):

- **Optimizer**: `optimizer.type` + `optimizer.extra_params` in the flat form; `rengu_flow_ui/optimizer_form.py` splits/merges KV ↔ flat `[optimizer]` keys via `split_optimizer_extras` / `merge_optimizer_extras` (same pattern as `scheduler_form.py`).
- **Scheduler**: `lr_scheduler` + `lr_scheduler_args.extra_params`.

Default KV rows when the user changes type are defined in `rengu_flow_ui/optim_kv_defaults.py` (`OPTIMIZER_REGISTRY_KV_DEFAULTS`, `SCHEDULER_BUILTIN_KV_DEFAULTS`, `SCHEDULER_FQN_KV_DEFAULTS`) and served to the frontend via `get_registries()` — no TypeScript copy.

When adding a registry optimizer with non-trivial constructor args:

1. Register the class in `registry/optimizers.py`.
2. Add a row to `OPTIMIZER_REGISTRY_KV_DEFAULTS` (include `lr` / `betas` when applicable).
3. Document parameters in `docs/user/optimizer-and-scheduler.md` (per-type table + link to upstream docs).
4. Extend `tests/test_optim_kv_defaults.py` and optimizer/scheduler form tests if behavior changes.
