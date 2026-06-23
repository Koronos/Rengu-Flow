# Engine Backend Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scattered `if backend==`/`if distributed` conditionals with a polymorphic `TrainingBackend` strategy (two concrete backends) selected by a dict factory, so launcher, build, capability constraints, and cache strategy live behind one interface.

**Architecture:** Two layers. Layer 1 `TrainingBackend` (ABC, torch-free) owns launcher argv, config validation, capability flags, engine/pipe build, and cache-worker creation. Layer 2 `Engine` (existing `TorchEngine` / DeepSpeed engine, unchanged behavior) is produced by `build_engine` and documented by a `typing.Protocol`. A `select_backend(config)` factory (dict literal) picks the concrete backend; consumers call methods instead of branching.

**Tech Stack:** Python 3.13, PyTorch 2.12 cu130, DeepSpeed 0.19, `multiprocess`/`multiprocessing`, pytest + pytest-xdist, uv.

## Global Constraints

- Config strings unchanged: `engine = "accelerate" | "deepspeed"` (these are the backend `name`s). Verbatim.
- `rengu_flow/engine/base.py` MUST NOT import `torch`, `deepspeed`, or any heavy ML lib at module top (CLI launch + config validation path). DeepSpeed import stays inside `deepspeed_pipe.py` build methods only ([[deepspeed-import-boundary]]).
- No version bump (`pyproject.toml`/`uv.lock` untouched).
- Author commits as `koronos`; sign `Co-Authored-By: Poet <noreply@anthropic.com>`. English only in code/docs.
- Run pytest from repo root with `PYTHONPATH="$PWD" uv run --extra dev pytest …` ([[pytest-in-worktree-needs-pythonpath]]).
- Built on top of branch `refactor/cache-threadpool` (ThreadPool cache refactor already present).
- Behavior-preserving: deepspeed and accelerate must still build + cache + step identically after every task.

---

## File structure

```
rengu_flow/engine/
  __init__.py        # select_backend(config) -> TrainingBackend ; back-compat: resolve_backend, build_pipe, build_engine
  base.py            # TrainingBackend(ABC) + Engine(Protocol)  — torch/deepspeed-free
  single_device.py   # SingleDeviceBackend + TorchEngine + SequentialPipe + _SingleGpuGrid (moved from engine.py)
  deepspeed_pipe.py  # DeepSpeedPipeBackend (deepspeed.initialize, ManualPipelineModule, block_swap patch, launcher)
```
`rengu_flow/engine.py` is deleted; its public symbols are re-exported from `engine/__init__.py`.

Consumers touched: `cli/train_launcher.py`, `cli/train_cmd.py`, `main.py`, `data/manager.py`.

---

## Task 1: Create the `engine/` package, move runtime code, keep behavior identical

**Files:**
- Create: `rengu_flow/engine/__init__.py`, `rengu_flow/engine/base.py`, `rengu_flow/engine/single_device.py`, `rengu_flow/engine/deepspeed_pipe.py`
- Delete: `rengu_flow/engine.py`
- Test: `tests/test_engine_backend.py` (new), existing `tests/test_engine.py` (must stay green)

**Interfaces:**
- Produces: `select_backend(config: dict) -> TrainingBackend`; `TrainingBackend` ABC (methods per spec); `Engine` Protocol; back-compat `resolve_backend(config=None) -> str`, `build_pipe(backend, *, layers, num_stages, partition_method, manual_partition_split, loss_fn, extra_kw)`, `build_engine(backend, *, pipeline_model, ds_config, args, get_optimizer, parameters_to_train)`.
- Consumes: existing `TorchEngine`, `SequentialPipe`, `_SingleGpuGrid` (current `engine.py`); `rengu_flow.platform_compat.PLATFORM.default_engine`; `rengu_flow.utils.pipeline.ManualPipelineModule`.

- [ ] **Step 1: Write the failing test** `tests/test_engine_backend.py`

```python
"""TrainingBackend factory + capability surface (CPU-only, no torch/deepspeed needed)."""
import importlib
import sys

import pytest


def test_select_backend_resolution(monkeypatch):
    from rengu_flow.engine import select_backend
    monkeypatch.delenv("RENGU_ENGINE", raising=False)
    assert select_backend({"engine": "accelerate"}).name == "accelerate"
    assert select_backend({"engine": "deepspeed"}).name == "deepspeed"
    monkeypatch.setenv("RENGU_ENGINE", "accelerate")
    assert select_backend({"engine": "deepspeed"}).name == "accelerate"  # env wins


def test_select_backend_unknown():
    from rengu_flow.engine import select_backend
    with pytest.raises(SystemExit):
        select_backend({"engine": "nope"})


def test_capabilities():
    from rengu_flow.engine import select_backend
    acc = select_backend({"engine": "accelerate"})
    ds = select_backend({"engine": "deepspeed"})
    assert acc.is_distributed is False and ds.is_distributed is True
    assert acc.supports_gradient_release is False and ds.supports_gradient_release is True


def test_base_is_torch_free():
    # base.py must import without torch/deepspeed (CLI launch + config validation path).
    for mod in ("torch", "deepspeed"):
        sys.modules.pop(mod, None)
    importlib.reload(importlib.import_module("rengu_flow.engine.base"))
    # base imported; torch/deepspeed must not have been pulled in by it.
    assert "deepspeed" not in sys.modules


def test_launch_argv_accelerate():
    from rengu_flow.engine import select_backend
    argv = select_backend({"engine": "accelerate"}).launch_argv(
        {"engine": "accelerate"}, config_path="cfg.toml", num_gpus=1, master_port=29500
    )
    assert argv[1:3] == ["-m", "rengu_flow.main"]
    assert "--config" in argv and "cfg.toml" in argv
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_engine_backend.py -q`
Expected: FAIL (`ModuleNotFoundError: rengu_flow.engine` is now a package not yet created, or import errors).

- [ ] **Step 3: Create `rengu_flow/engine/base.py`** (torch-free)

```python
"""TrainingBackend strategy + Engine runtime protocol. TORCH/DEEPSPEED-FREE on import."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    """Runtime surface the training loop / Saver / eval depend on (was kept in sync with
    deepspeed.runtime by convention). Documentation-only; both engines satisfy it structurally."""

    optimizer: Any
    lr_scheduler: Any
    communication_data_type: Any
    module: Any
    grid: Any
    is_pipe_parallel: bool
    num_stages: int
    micro_batches: int

    def train_batch(self, iterator) -> Any: ...
    def eval_batch(self, iterator, num_micro_batches: int | None = ...) -> Any: ...
    def reset_activation_shape(self) -> None: ...
    def zero_grad(self) -> None: ...
    def get_global_grad_norm(self) -> Any: ...
    def save_checkpoint(self, *args, **kwargs) -> Any: ...
    def load_checkpoint(self, *args, **kwargs) -> Any: ...
    def is_first_stage(self) -> bool: ...
    def is_last_stage(self) -> bool: ...


class TrainingBackend(ABC):
    """One object owns the engine-specific concerns: launcher, validation, capabilities, build,
    caching. Constructible from config alone so the CLI can introspect before any engine exists."""

    name: ClassVar[str]

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    # -- Phase 1: CLI launch (classmethod: parent process, no engine, no torch) --
    @classmethod
    @abstractmethod
    def launch_argv(cls, config: dict, *, config_path: str, num_gpus: int, master_port: int) -> list[str]:
        ...

    # -- Phase 2: config validation (centralizes the scattered guards) --
    @abstractmethod
    def validate(self, config: dict) -> None:
        ...

    # -- Capability flags --
    @property
    @abstractmethod
    def is_distributed(self) -> bool: ...

    @property
    @abstractmethod
    def supports_block_swap(self) -> bool: ...

    @property
    @abstractmethod
    def supports_gradient_release(self) -> bool: ...

    # -- Phase 3: build (returns the Layer-2 Engine) --
    @abstractmethod
    def build_pipe(self, *, layers, num_stages, partition_method, manual_partition_split, loss_fn, extra_kw): ...

    @abstractmethod
    def build_engine(self, *, pipeline_model, ds_config, args, get_optimizer, parameters_to_train) -> Engine: ...

    # -- Phase 4: caching (returns the right (worker, queue) for this backend) --
    @abstractmethod
    def make_cache_worker(self, cache_fn, args) -> tuple[Any, Any]: ...
```

- [ ] **Step 4: Create `rengu_flow/engine/single_device.py`**

Move VERBATIM from the current `rengu_flow/engine.py` into this file: `_SingleGpuGrid` (class), `SequentialPipe` (class), `TorchEngine` (class) — including their imports (`import os`, `from pathlib import Path`, `from typing import Any`, `import torch`). Then add the backend class below them:

```python
class SingleDeviceBackend(TrainingBackend):
    """Single-GPU plain-torch engine ("accelerate")."""

    name = "accelerate"

    @classmethod
    def launch_argv(cls, config, *, config_path, num_gpus, master_port):
        import sys
        return [sys.executable, "-m", "rengu_flow.main", "--config", str(config_path)]

    def validate(self, config):
        if config.get("optimizer", {}).get("gradient_release"):
            raise ValueError(
                "optimizer.gradient_release requires engine='deepspeed' (it patches the DeepSpeed "
                "pipeline engine); engine='accelerate' does not support it."
            )
        if config.get("pipeline_stages", 1) > 1:
            raise ValueError("pipeline_stages > 1 requires engine='deepspeed'.")
        if config.get("blocks_to_swap", 0) and not _is_adapter(config):
            raise ValueError(
                "engine='accelerate' block swap supports adapter (LoRA/LoKr) training only; "
                "full-model swap needs gradient_release — use engine='deepspeed'."
            )

    @property
    def is_distributed(self): return False

    @property
    def supports_block_swap(self): return True  # adapters only; validate() enforces

    @property
    def supports_gradient_release(self): return False

    def build_pipe(self, *, layers, num_stages, partition_method, manual_partition_split, loss_fn, extra_kw):
        if num_stages > 1:
            raise SystemExit("engine='accelerate' is single-stage; set pipeline_stages = 1.")
        return SequentialPipe(layers, loss_fn, **extra_kw)

    def build_engine(self, *, pipeline_model, ds_config, args, get_optimizer, parameters_to_train):
        return TorchEngine(
            pipeline_model, get_optimizer, parameters_to_train, ds_config,
            block_swap=bool((self.config or {}).get("blocks_to_swap", 0)),
        )

    def make_cache_worker(self, cache_fn, args):
        import threading
        import queue as _queue
        q = _queue.Queue()
        worker = threading.Thread(target=cache_fn, args=(args, q), daemon=True)
        return worker, q
```

Add a small local helper `_is_adapter(config)` mirroring `main.py`'s current adapter check (read it at execution and copy the exact predicate, typically `bool(config.get("adapter"))`).

- [ ] **Step 5: Create `rengu_flow/engine/deepspeed_pipe.py`**

Move the deepspeed branches of the current `build_pipe`/`build_engine` into methods here. DeepSpeed/`ManualPipelineModule` imported INSIDE methods only.

```python
"""Multi-GPU DeepSpeed pipeline engine ("deepspeed")."""
from __future__ import annotations

from rengu_flow.engine.base import TrainingBackend


class DeepSpeedPipeBackend(TrainingBackend):
    name = "deepspeed"

    @classmethod
    def launch_argv(cls, config, *, config_path, num_gpus, master_port):
        from shutil import which
        deepspeed = which("deepspeed")
        if not deepspeed:  # fall back like today when the launcher is absent
            import sys
            return [sys.executable, "-m", "rengu_flow.main", "--config", str(config_path)]
        cmd = [deepspeed, f"--num_gpus={num_gpus}"]
        if master_port is not None:
            cmd.append(f"--master_port={master_port}")
        cmd += ["--module", "rengu_flow.main", "--config", str(config_path)]
        return cmd

    def validate(self, config):
        # DeepSpeed supports the full feature set; only the data-parallel constraint for
        # gradient_release (handled where the optimizer is built) applies. No-op here for now.
        return None

    @property
    def is_distributed(self): return True

    @property
    def supports_block_swap(self): return True

    @property
    def supports_gradient_release(self): return True

    def build_pipe(self, *, layers, num_stages, partition_method, manual_partition_split, loss_fn, extra_kw):
        from rengu_flow.utils.pipeline import ManualPipelineModule
        return ManualPipelineModule(
            layers=layers, num_stages=num_stages, partition_method=partition_method,
            manual_partition_split=manual_partition_split, loss_fn=loss_fn, **extra_kw,
        )

    def build_engine(self, *, pipeline_model, ds_config, args, get_optimizer, parameters_to_train):
        import deepspeed
        engine, _, _, _ = deepspeed.initialize(args=args, model=pipeline_model, config=ds_config)
        engine._configure_optimizer(get_optimizer, parameters_to_train)
        return engine

    def make_cache_worker(self, cache_fn, args):
        # Multi-GPU: rank-0 process worker feeds a Manager queue broadcast to all ranks.
        try:
            import multiprocess as mp
        except ImportError:
            import multiprocessing as mp
        manager = mp.Manager()
        q = manager.Queue()
        worker = mp.Process(target=cache_fn, args=(args, q))
        return worker, q
```

NOTE at execution: the current deepspeed caching path broadcasts the queue across ranks inside `manager.cache()`. `make_cache_worker` returns the worker+queue for rank 0; the broadcast/rank coordination stays in `manager.cache()` (Task 5). Read the exact current `build_pipe`/`build_engine` deepspeed bodies and `ManualPipelineModule` kwargs and copy them faithfully.

- [ ] **Step 6: Create `rengu_flow/engine/__init__.py`** (factory + back-compat)

```python
"""Engine backend selection + back-compat shims."""
from __future__ import annotations

import os

from rengu_flow.engine.base import Engine, TrainingBackend
from rengu_flow.engine.single_device import SequentialPipe, SingleDeviceBackend, TorchEngine
from rengu_flow.engine.deepspeed_pipe import DeepSpeedPipeBackend

_BACKENDS = {b.name: b for b in (SingleDeviceBackend, DeepSpeedPipeBackend)}


def resolve_backend(config: dict | None = None) -> str:
    from rengu_flow.platform_compat import PLATFORM
    name = (os.environ.get("RENGU_ENGINE") or (config or {}).get("engine") or "").strip().lower()
    return name or PLATFORM.default_engine


def select_backend(config: dict | None = None) -> TrainingBackend:
    name = resolve_backend(config)
    try:
        return _BACKENDS[name](config)
    except KeyError:
        raise SystemExit(f"unknown engine {name!r} (accelerate|deepspeed)")


# Back-compat free functions (delegate to the selected backend); removed in Task 6 once call sites migrate.
def build_pipe(backend: str, *, layers, num_stages, partition_method, manual_partition_split, loss_fn, extra_kw):
    return _BACKENDS[backend]({}).build_pipe(
        layers=layers, num_stages=num_stages, partition_method=partition_method,
        manual_partition_split=manual_partition_split, loss_fn=loss_fn, extra_kw=extra_kw,
    )


def build_engine(backend: str, *, pipeline_model, ds_config, args, get_optimizer, parameters_to_train):
    return _BACKENDS[backend]({}).build_engine(
        pipeline_model=pipeline_model, ds_config=ds_config, args=args,
        get_optimizer=get_optimizer, parameters_to_train=parameters_to_train,
    )


__all__ = ["Engine", "TrainingBackend", "SingleDeviceBackend", "DeepSpeedPipeBackend",
           "select_backend", "resolve_backend", "build_pipe", "build_engine",
           "TorchEngine", "SequentialPipe"]
```

- [ ] **Step 7: Delete `rengu_flow/engine.py`** — `git rm rengu_flow/engine.py`.

- [ ] **Step 8: Run tests**

Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_engine_backend.py tests/test_engine.py -q`
Expected: PASS (new backend tests + the existing 6 TorchEngine tests, which import `TorchEngine` via the package re-export).

- [ ] **Step 9: Commit**

```bash
git add rengu_flow/engine tests/test_engine_backend.py
git rm rengu_flow/engine.py
git commit -m "refactor(engine): TrainingBackend strategy + factory package (additive, behavior-preserving)

Co-Authored-By: Poet <noreply@anthropic.com>"
```

---

## Task 2: Flip the launcher to `backend.launch_argv`

**Files:**
- Modify: `rengu_flow/cli/train_launcher.py` (`base_train_command`)
- Test: `tests/test_train_launcher_engine.py` (new)

**Interfaces:**
- Consumes: `select_backend(config).launch_argv(config, config_path=, num_gpus=, master_port=)`.

- [ ] **Step 1: Write the failing test** `tests/test_train_launcher_engine.py`

```python
from pathlib import Path
from rengu_flow.cli.train_launcher import base_train_command


def test_accelerate_uses_python_m(monkeypatch):
    monkeypatch.setenv("RENGU_ENGINE", "accelerate")
    cmd = base_train_command(Path("cfg.toml"), num_gpus=1, master_port=29500)
    assert cmd[1:3] == ["-m", "rengu_flow.main"]


def test_deepspeed_uses_launcher(monkeypatch):
    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")
    cmd = base_train_command(Path("cfg.toml"), num_gpus=2, master_port=29500)
    assert "rengu_flow.main" in cmd and ("--module" in cmd or cmd[1:3] == ["-m", "rengu_flow.main"])
```

- [ ] **Step 2: Run → FAIL.** Run: `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_train_launcher_engine.py -q`

- [ ] **Step 3: Edit `base_train_command`** — replace the current `if resolve_backend()=="deepspeed": …` body with:

```python
def base_train_command(config_path, *, num_gpus, master_port=None):
    from rengu_flow.engine import select_backend
    # Config is read elsewhere; pass what the launcher knows. resolve_backend reads RENGU_ENGINE/config.
    backend = select_backend({})  # env/OS default; --engine already set RENGU_ENGINE upstream
    return backend.launch_argv({}, config_path=str(config_path), num_gpus=num_gpus, master_port=master_port)
```

At execution: confirm `--engine` upstream sets `RENGU_ENGINE` before this runs (it does, `train_cmd.run_train`). If `base_train_command` has access to the loaded config, pass it instead of `{}`.

- [ ] **Step 4: Run → PASS.** Same command.

- [ ] **Step 5: Commit** `refactor(engine): route launcher through backend.launch_argv`.

---

## Task 3: Flip capability guards to `backend.validate` + properties

**Files:**
- Modify: `rengu_flow/main.py` (the ~7 `if backend==` guard sites: gradient_release guard ~611, block_swap setup ~476-498)
- Test: extend `tests/test_engine_backend.py`

- [ ] **Step 1: Add failing tests** for validation:

```python
def test_validate_accelerate_rejects_gradient_release():
    from rengu_flow.engine import select_backend
    import pytest
    b = select_backend({"engine": "accelerate"})
    with pytest.raises(ValueError):
        b.validate({"optimizer": {"gradient_release": True}})
    with pytest.raises(ValueError):
        b.validate({"pipeline_stages": 2})


def test_validate_deepspeed_allows():
    from rengu_flow.engine import select_backend
    select_backend({"engine": "deepspeed"}).validate(
        {"optimizer": {"gradient_release": True}, "pipeline_stages": 2}
    )  # no raise
```

- [ ] **Step 2: Run → PASS** (validate already implemented in Task 1; this just locks behavior). If a case is missing, add it to the backend `validate()`.

- [ ] **Step 3: Edit `main.py`** — after `backend = select_backend(config)`, call `backend.validate(config)` once near the top of setup (replacing the scattered `if backend==…: raise` for gradient_release/pipeline_stages/block-swap-adapter). Replace the deepspeed-only block-swap patch site with:

```python
if config.get("blocks_to_swap", 0) and backend.is_distributed:
    from rengu_flow.training.block_swap import patch_deepspeed_for_block_swap
    patch_deepspeed_for_block_swap()
    model.enable_block_swap(config["blocks_to_swap"])
elif config.get("blocks_to_swap", 0):
    model.enable_block_swap(config["blocks_to_swap"])
```

Read the exact current lines 462-498 and 610-619 at execution and replace the string-keyed checks with `backend.is_distributed` / capability properties; keep the `gradient_release` data-parallel-world-size check (it is a runtime guard, not a backend branch).

- [ ] **Step 4: Run** `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_engine_backend.py tests/test_engine.py -q` → PASS.

- [ ] **Step 5: Commit** `refactor(engine): centralize capability guards in backend.validate`.

---

## Task 4: Flip build to `backend.build_pipe` / `backend.build_engine`

**Files:**
- Modify: `rengu_flow/main.py` (build sites ~512 build_pipe, ~596 build_engine, and `backend = resolve_backend(config)` → `select_backend`)

- [ ] **Step 1:** In `main.py`, change `from rengu_flow.engine import build_engine, build_pipe, resolve_backend` to `from rengu_flow.engine import select_backend`, and `backend = resolve_backend(config)` to `backend = select_backend(config)`.
- [ ] **Step 2:** Replace `build_pipe(backend, …)` call with `backend.build_pipe(…)` and `build_engine(backend, …)` with `backend.build_engine(…)` (same kwargs).
- [ ] **Step 3: Run the engine smoke (both engines).** This is a GPU step — see Verification section. Expected: both build + step.
- [ ] **Step 4: Commit** `refactor(engine): build pipeline/engine via backend object`.

---

## Task 5: Flip caching to `backend.make_cache_worker`

**Files:**
- Modify: `rengu_flow/data/manager.py` (`DatasetManager.__init__` to accept `backend`; `cache()` worker/queue creation), `rengu_flow/main.py` (pass `backend=backend` into `DatasetManager(...)`)
- Test: `tests/test_cache_handoff_queue.py` (the single-process path still works)

- [ ] **Step 1:** Add `backend` param to `DatasetManager.__init__(..., backend=None)`, store `self.backend`.
- [ ] **Step 2:** In `cache()`, replace the `if distributed:` queue/worker creation with:

```python
worker, queue = self.backend.make_cache_worker(_run_cache_worker, cache_args) if is_main_process() else (None, _broadcast_queue())
```

At execution: the distributed path must still broadcast the manager queue to non-main ranks (current lines 228-237). Keep that broadcast for `backend.is_distributed`; for the single path `make_cache_worker` returns the thread queue directly. Concretely: if `backend.is_distributed`, build the queue via the backend on rank 0 then `broadcast_object_list`; else use the backend's thread queue. Preserve the existing drain loop, `dist.barrier()`, and `worker.join()` unchanged.

- [ ] **Step 3:** In `main.py`, pass `backend=backend` to the `DatasetManager(...)` constructor.
- [ ] **Step 4: Run** `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_cache_handoff_queue.py tests/test_cache_utils_config.py -q` → PASS.
- [ ] **Step 5: Run cache-regenerate smoke on BOTH engines** (Verification section). Expected: both complete, no deadlock.
- [ ] **Step 6: Commit** `refactor(engine): cache worker/queue owned by backend`.

---

## Task 6: Remove dead code + final sweep

**Files:**
- Modify: `rengu_flow/engine/__init__.py` (drop the back-compat `build_pipe`/`build_engine` free functions once no caller remains), any lingering `accelerate_deepspeed` references.

- [ ] **Step 1:** `grep -rn "accelerate_deepspeed\|build_pipe(\|build_engine(" rengu_flow/ --include=*.py | grep -v engine/` → expect no remaining call-site usages.
- [ ] **Step 2:** Remove the back-compat free functions from `engine/__init__.py` if unused (keep `resolve_backend`, `select_backend`, classes). Drop the `accelerate_deepspeed` stub/NotImplementedError everywhere.
- [ ] **Step 3:** `grep -rn "backend ==\|== .deepspeed.\|== .accelerate.\|if distributed" rengu_flow/ --include=*.py | grep -v test_` → expect only legitimate residue (e.g. the gradient_release world-size runtime guard), none re-introducing engine branching in consumers.
- [ ] **Step 4: Run full relevant suite** `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_engine.py tests/test_engine_backend.py tests/test_train_launcher_engine.py tests/test_cache_handoff_queue.py tests/test_cache_utils_config.py tests/test_data_loader.py -q` → PASS.
- [ ] **Step 5: Commit** `refactor(engine): drop string if-ladders and dead accelerate_deepspeed stub`.

---

## Verification (run before merge — both engines, GPU)

Cache is built already (`tmp/bench_nekaon.toml`). Smokes use a low step count.

1. **Unit/integration suite:** `PYTHONPATH="$PWD" uv run --extra dev pytest tests/test_engine.py tests/test_engine_backend.py tests/test_train_launcher_engine.py tests/test_cache_handoff_queue.py tests/test_cache_utils_config.py tests/test_data_loader.py tests/test_data_synthetic.py -q` → all pass.
2. **Accelerate smoke (single GPU):** `RENGU_BENCH_PEAK_PER_STEP=1 uv run rengu train --config tmp/bench_nekaon.toml --engine accelerate -- --trust_cache` → builds, caches/reads, steps to max_steps, "Training complete." (monitor with hang/zombie watchdog).
3. **DeepSpeed smoke (single GPU launcher):** same with `--engine deepspeed` → builds via deepspeed.initialize, steps, completes.
4. **Cache regenerate both engines:** `RENGU_ENGINE=accelerate uv run rengu cache --config tmp/bench_nekaon.toml -- --regenerate_cache` and the `deepspeed` variant → both reach "cache complete", no deadlock.

Every GPU run gets a Monitor + timeout ([[monitor-long-running-commands]]); confirm GPU exclusive first ([[engine-ab-accelerate-vs-deepspeed-singlegpu]]).

## Merge

Once tasks 1-6 land and Verification passes: merge the working branch (which carries the ThreadPool cache refactor + this separation) into `develop`. **No version bump** — do not touch `pyproject.toml`/`uv.lock`. Discard the superseded `fix/cache-handoff-queue-deadlock` branch.
