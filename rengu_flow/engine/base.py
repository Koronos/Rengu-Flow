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
