# Adding optimizers and schedulers (developer guide)

This document describes where optimizer and scheduler resolution lives, how resolution works (registry + qualified path), and how to **register** a custom optimizer or scheduler so it can be referenced by name in TOML without editing the framework core.

## Code locations

- **Optimizer registry**: `renga_flow/registry/optimizers.py`
  - `optimizer_registry`: dict mapping name (lowercase) → optimizer **class**
  - `register_optimizer(name)` decorator
  - `get_optimizer_class(optim_type)` — registry lookup first, then fully-qualified path
- **Scheduler registry and resolution**: `renga_flow/optim/resolver.py`
  - `scheduler_registry`: dict mapping name (lowercase) → **factory** `(optimizer, config, total_steps, steps_per_epoch) -> LRScheduler | None`
  - `register_scheduler(name)` decorator
  - `resolve_scheduler(...)` — registry lookup first, then fully-qualified path
  - `substitute_runtime_tokens(kwargs, runtime_values)` for `total_steps` / `steps_per_epoch` / `epochs` in config
  - `apply_warmup(optimizer, scheduler, warmup_steps)` — wraps scheduler with LinearLR warmup

The training entry point (`renga_flow/main.py`) uses `resolve_optimizer_class(config["optimizer"]["type"])` and `resolve_scheduler(lr_scheduler, optimizer, config, ...)` from `renga_flow.optim`; it does not call the registry modules directly.

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
from renga_flow.registry.optimizers import register_optimizer
import torch

@register_optimizer("my_custom_optim")
class MyOptimizer(torch.optim.Optimizer):
    # ... implement step(), __init__, etc.
    pass
```

Or register an existing class:

```python
from renga_flow.registry.optimizers import register_optimizer
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
from renga_flow.optim.resolver import register_scheduler
import torch

def _my_cosine(optimizer, config, total_steps, steps_per_epoch):
    lr_min = config.get("lr_scheduler_args", {}).get("lr_min", 0.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr_min
    )

register_scheduler("my_cosine")(_my_cosine)
```

Then in TOML: `lr_scheduler = "my_cosine"`. Runtime tokens (`total_steps`, etc.) are available in `config`; your factory can read `config["lr_scheduler_args"]` and use `substitute_runtime_tokens` from `renga_flow.optim.resolver` if you need to pass token strings to a class constructor.

## Built-ins

- **Optimizers** (in `registry/optimizers.py`): `adamw`, `sgd`, `adam` — `torch.optim`; `genericoptim`, `automagic`, `adamw8bitkahan` — lazy-loaded from `renga_flow/vendor/diffusion_pipe_optimizers/` (see NOTICE there).
- **Optional aliases** (lazy import, `[optim]` extra): `adamw8bit`, `adamw_optimi`, `stableadamw`, `offload` — see `OPTIMIZER_ALIASES` in `registry/optimizers.py`.
- **Fallback:** names without `.` are also looked up on the `pytorch_optimizer` package (e.g. `Prodigy`).
- **Schedulers** (in `optim/resolver.py`): `constant`, `linear`, `cosine`, `none`.

## Third-party optimizers (vendored)

Copied from [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) under `renga_flow/vendor/diffusion_pipe_optimizers/`. See `README.md` and `NOTICE.md` in that folder for upstream credits and source commit.

Do not import that package at module load for all optimizers — `get_optimizer_class` lazy-imports vendor modules so tests and minimal installs avoid `bitsandbytes` / `optimum.quanto` until needed.

## Training helpers

- **`renga_flow/optim/param_groups.py`**: `adjust_beta2_half_life`, `split_weight_decay_param_groups`, `split_genericoptim_param_groups`.
- **`renga_flow/utils/training_metrics.py`**: `log_training_step`, `get_prodigy_d`, `get_automagic_lrs` — used from `main.py`.

Adding more built-in names can be done via `register_optimizer` or by extending `VENDOR_OPTIMIZER_ALIASES` / `OPTIMIZER_ALIASES` in `registry/optimizers.py`.
