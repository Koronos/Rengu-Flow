# Engine Backend Separation — Design

Date: 2026-06-23
Status: Proposed (pending implementation plan)

## Problem

Engine-specific behavior is scattered as `if backend == "deepspeed"/"accelerate"` (and `if distributed`)
across the codebase. There is no formal engine abstraction: the two engines are "kept in sync with
`deepspeed.runtime` by convention". Concretely it bifurcates in **four** places:

| # | Location | Bifurcates on | Today |
|---|---|---|---|
| 1 | `cli/train_launcher.py` `base_train_command` (+ `cli/train_cmd.py`) | launcher: `deepspeed --num_gpus` vs `python -m` | `if resolve_backend()=="deepspeed"` |
| 2 | `engine.py` `build_pipe` / `build_engine` | pipeline + engine build | if-ladder by string |
| 3 | `main.py` (~7 sites) | capability constraints (block_swap, gradient_release, pipeline_stages) | scattered `if backend==…: raise/patch` |
| 4 | `data/manager.py` `cache()` | cache strategy (multi-GPU distributed queue vs single-process thread) | `if distributed:` |

(The model is clean: `model/cosmos_predict2/dit.py`'s `backend` refs are the *attention* backend —
`transformer_engine` vs `torch` — unrelated to the training engine.)

Mental model the design commits to: **`deepspeed` ⟺ multi-GPU, `accelerate` ⟺ single-GPU.** No
single-GPU-on-deepspeed special cases.

## Goals / Non-goals

**Goals**
- One cohesive object per engine owns its four concerns; consumers are backend-agnostic (no `if
  backend==` in call sites).
- Minimal disruption to a modular codebase; green tests at every migration step.
- Config strings `engine = "accelerate" | "deepspeed"` unchanged (back-compat; clarity over renaming).
- `base.py` torch/deepspeed-free so CLI launch + config validation don't pay DeepSpeed's ~17s import
  (consistent with the existing DeepSpeed-import-boundary seam).

**Non-goals**
- No change to the per-step training loop or the runtime `Engine` methods.
- No change to the data caching *algorithm* (the ThreadPool work is separate; this only relocates the
  distributed-vs-thread *choice*).
- No multi-node work. `accelerate_deepspeed` is dropped (YAGNI; re-add as a class if needed).

## Architecture — two layers

**Layer 1 — `TrainingBackend` (Strategy/policy).** New. Lightweight, constructible from config alone.
Owns the four bifurcation points. Selected by a factory.

**Layer 2 — `Engine` (runtime).** Already exists as `TorchEngine` (accelerate) and the DeepSpeed engine
object. Unchanged behavior; `TrainingBackend.build_engine()` produces it. Formalized as a `typing.Protocol`
(documentation-only, no runtime cost) to replace the "in sync by convention" contract.

### File structure

`engine.py` (module) → `engine/` (package):

```
rengu_flow/engine/
  __init__.py        # select_backend(config) -> TrainingBackend ; back-compat re-exports
  base.py            # TrainingBackend (ABC) + Engine (Protocol)  — TORCH/DEEPSPEED-FREE
  single_device.py   # SingleDeviceBackend  (+ TorchEngine, SequentialPipe, _SingleGpuGrid moved here)
  deepspeed_pipe.py  # DeepSpeedPipeBackend (deepspeed.initialize, ManualPipelineModule, block_swap patch, deepspeed launcher)
```

`engine/__init__.py` re-exports `resolve_backend`, and thin `build_pipe(name, …)` / `build_engine(name, …)`
wrappers (delegating to `select_backend`) so existing imports keep working while call sites migrate.

## Interface

```python
# engine/base.py  — torch-free
class TrainingBackend(ABC):
    name: ClassVar[str]                      # "accelerate" | "deepspeed"

    # Phase 1 — CLI launch (classmethod: runs in the parent process, no engine yet, no torch import)
    @classmethod
    @abstractmethod
    def launch_argv(cls, config, *, num_gpus: int, master_port: int) -> list[str]: ...

    # Phase 2 — config validation (centralizes the ~7 main.py guards)
    @abstractmethod
    def validate(self, config) -> None: ...

    # Capability flags (read instead of `if backend==`)
    @property
    @abstractmethod
    def is_distributed(self) -> bool: ...
    @property
    @abstractmethod
    def supports_block_swap(self) -> bool: ...
    @property
    @abstractmethod
    def supports_gradient_release(self) -> bool: ...

    # Phase 3 — build (produces the Layer-2 Engine)
    @abstractmethod
    def build_pipe(self, *, layers, num_stages, loss_fn, **kw): ...
    @abstractmethod
    def build_engine(self, *, pipeline_model, ds_config, get_optimizer, parameters_to_train, **kw) -> "Engine": ...

    # Phase 4 — caching (returns the right worker + queue for this backend)
    @abstractmethod
    def make_cache_worker(self, cache_fn, args) -> tuple[object, object]: ...  # (worker, queue)


class Engine(Protocol):
    """Runtime surface the training loop / Saver depend on (was 'kept in sync with deepspeed.runtime')."""
    optimizer: object
    lr_scheduler: object
    micro_batches: int
    def train_batch(self, iterator) -> "torch.Tensor": ...
    def eval_batch(self, iterator, num_micro_batches: int | None = ...) -> "torch.Tensor": ...
    def save_checkpoint(self, *a, **k): ...
    def load_checkpoint(self, *a, **k): ...
    def get_global_grad_norm(self): ...
    def is_first_stage(self) -> bool: ...
    def is_last_stage(self) -> bool: ...
    # … (full surface enumerated from the current TorchEngine / DeepSpeed usage)
```

### Concrete backends

- **`SingleDeviceBackend`** (`name = "accelerate"`): `launch_argv` → `[python, -m, rengu_flow.main, …]`;
  `validate` rejects `gradient_release` and `pipeline_stages > 1`, allows `block_swap` only for adapters;
  `is_distributed=False`, `supports_block_swap=True (adapters)`, `supports_gradient_release=False`;
  `build_pipe` → `SequentialPipe`; `build_engine` → `TorchEngine`; `make_cache_worker` → `threading.Thread`
  + thread `queue.Queue` (the single-process ThreadPool cache path).
- **`DeepSpeedPipeBackend`** (`name = "deepspeed"`): `launch_argv` → `["deepspeed", "--num_gpus=N", "--module", …]`;
  `validate` allows the full feature set; `is_distributed=True`, `supports_gradient_release=True`;
  `build_pipe` → `ManualPipelineModule`; `build_engine` → `deepspeed.initialize` + `_configure_optimizer`
  (+ `patch_deepspeed_for_block_swap` when block_swap); `make_cache_worker` → `mp.Process` + broadcast
  `mp.Manager().Queue()` (multi-GPU rank fan-out). DeepSpeed imported only inside this module.

## Selection & lifecycle

Factory in `engine/__init__.py`:

```python
_BACKENDS = {b.name: b for b in (SingleDeviceBackend, DeepSpeedPipeBackend)}

def select_backend(config) -> TrainingBackend:
    name = (os.environ.get("RENGU_ENGINE") or config.get("engine") or "").strip().lower()
    name = name or PLATFORM.default_engine            # accelerate everywhere (deepspeed is opt-in)
    try:
        return _BACKENDS[name](config)
    except KeyError:
        raise SystemExit(f"unknown engine {name!r} (accelerate|deepspeed)")
```

- **CLI process**: `cli/train_launcher.build_train_command` does `select_backend(config).launch_argv(…)`.
  No torch/deepspeed import (base + single_device launcher path stays torch-free at module import; the
  deepspeed import lives inside `DeepSpeedPipeBackend.build_*`, not `launch_argv`).
- **Worker process**: `main.py` constructs `backend = select_backend(config)` once, then
  `backend.validate(config)` → `backend.build_pipe(…)` → `backend.build_engine(…)`. The instance is
  injected into `DatasetManager(... , backend=backend)`, which calls `backend.make_cache_worker(…)`.
  Re-constructing the backend per process is cheap and deterministic (config-only), mirroring how
  Accelerate re-derives state from env in each process.

Unchanged and relied upon: `rengu_flow.distributed`'s `is_main_process()` / `barrier()` already no-op on
single-GPU, so the cache drain loop and the training loop are identical for both backends — only the
injected objects differ (the "no-op collectives" principle).

## Migration plan (green at every step)

1. **Add the abstraction (additive).** Create `engine/` package; move `TorchEngine`/`SequentialPipe`/
   `_SingleGpuGrid` and the deepspeed build into the two backend classes. `engine/__init__.py` re-exports
   the old names as wrappers. No call site changes yet → behavior identical, tests green.
2. **Flip launcher.** `train_launcher`/`train_cmd` → `backend.launch_argv`. Verify: CLI builds the right
   command for both; `import` path stays torch-free.
3. **Flip validation + capabilities.** Replace `main.py`'s ~7 guards with `backend.validate(config)` and
   `backend.supports_*` reads.
4. **Flip build.** `main.py` build calls → `backend.build_pipe/build_engine`.
5. **Flip caching.** `manager.cache()`'s `if distributed` → `backend.make_cache_worker(…)`; thread the
   backend into `DatasetManager`.
6. **Remove dead code.** Drop the `accelerate_deepspeed` stub and the now-unused string if-ladders.

Each step runs: `pytest tests/test_engine.py` + the cache/data subset, plus a manual deepspeed smoke and an
accelerate smoke (the existing engine A/B path) to confirm both engines still build, cache, and step.

## Testing

- **Unit (CPU, no GPU):** `select_backend` resolution (env > config > OS default); `validate` raises the
  right errors per engine (gradient_release/pipeline_stages on accelerate); capability properties;
  `launch_argv` argv shape for both; a **torch/deepspeed-free import test** for `engine.base` (guards the
  import boundary — `import rengu_flow.engine.base` must not import deepspeed/torch).
- **Existing:** keep + extend `tests/test_engine.py` (6 tests, TorchEngine surface) and
  `tests/test_cache_handoff_queue.py`.
- **Integration:** deepspeed and accelerate each build + cache-regenerate + run a short smoke (the engine
  A/B harness) with no behavior change vs. pre-refactor.

## Risks

- **Launcher torch-free**: a stray top-level `import torch`/`deepspeed` in `base.py`/`single_device.py`
  would reintroduce the ~17s import on the CLI path. Mitigated by the import-boundary unit test.
- **DeepSpeed build path** is the riskiest move (deepspeed.initialize + block_swap patch); gated by a
  deepspeed smoke after step 4.
- **Caching worker move** (step 5): verify cache regenerate completes on both engines (the multi-GPU
  rank fan-out path is exercised only structurally on this single-GPU machine).
