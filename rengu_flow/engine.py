"""Pluggable training engine: pick DeepSpeed (pipeline, multi-GPU) or a plain single-GPU
torch loop ("accelerate") at runtime.

Three backends, selected by ``resolve_backend`` (env ``RENGU_ENGINE`` > config ``engine`` >
per-OS default):

* ``deepspeed`` — the original pipeline-parallel path. Default on Linux. Multi-GPU,
  gradient_release, block_swap, pipeline_stages>1. Imports DeepSpeed only here.
* ``accelerate`` — single-GPU plain torch (this file's :class:`TorchEngine`). Default on
  Windows. No DeepSpeed import at all, so native Windows needs no DeepSpeed build.
* ``accelerate_deepspeed`` — Accelerate driving DeepSpeed ZeRO. Not implemented yet
  (raises); the knob exists so it can be filled when someone needs ZeRO offload on Windows.

The engine surface the training loop / Saver / eval depend on (kept in sync with
``deepspeed.runtime``): ``train_batch(iter)`` / ``eval_batch(iter, num_micro_batches=)`` →
scalar-loss tensor, ``reset_activation_shape()``, ``zero_grad()``, ``get_global_grad_norm()``,
``save_checkpoint(...)`` / ``load_checkpoint(...)``, attrs ``optimizer`` / ``lr_scheduler`` /
``communication_data_type`` / ``module`` / ``grid`` / ``is_pipe_parallel`` / ``num_stages`` /
``micro_batches`` / ``is_first_stage()`` / ``is_last_stage()``.

ponytail: ONE file, single-GPU only for the torch path. block_swap/gradient_release/
pipeline_stages>1 stay DeepSpeed-only — the SDXL-LoRA-on-8GB smoke (7.5 GB) proves they
aren't needed for the main Windows workload. Add the torch ports when a model needs swap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def resolve_backend(config: dict | None = None) -> str:
    """Pick the training backend. ``RENGU_ENGINE`` env wins, then ``config['engine']``,
    else the platform default (``accelerate`` on Windows, ``deepspeed`` elsewhere)."""
    from rengu_flow.platform_compat import PLATFORM

    eng = os.environ.get("RENGU_ENGINE") or (config or {}).get("engine") or ""
    eng = eng.strip().lower()
    return eng or PLATFORM.default_engine


# ------------------------------------------------------------------ single-GPU torch backend


class _SingleGpuGrid:
    """Stand-in for DeepSpeed's grid topology on one GPU: rank 0, world 1, one stage."""

    def get_data_parallel_rank(self) -> int:
        return 0

    def get_data_parallel_world_size(self) -> int:
        return 1

    def get_pipe_parallel_rank(self) -> int:
        return 0

    def get_pipe_parallel_world_size(self) -> int:
        return 1

    def stage_to_global(self, stage_id: int) -> int:
        return stage_id


class SequentialPipe(torch.nn.Module):
    """Hold ``model.to_layers()`` as a plain sequential module (no DeepSpeed PipelineModule).

    Layers follow the pipeline tuple-in/tuple-out convention; the final tuple goes to
    ``loss_fn``. Params keep the model-assigned ``original_name`` (export reads it).

    Supports per-layer activation checkpointing (``activation_checkpointing=true``): the layer
    sequence is walked in fixed windows of ``activation_checkpoint_interval`` and each window is
    wrapped in ``activation_checkpoint_func`` (``torch.utils.checkpoint``) — but only when EVERY
    layer in the window is a checkpointable type. This mirrors DeepSpeed's PipelineModule so the
    single-GPU path gets the same recompute-for-VRAM trade. ``use_reentrant`` is baked into the
    func by the caller; with ``use_reentrant=False`` it works even when the inputs don't require
    grad (LoRA: only adapter params inside the window do)."""

    def __init__(self, layers: list, loss_fn: Any, *, activation_checkpoint_interval: int = 0,
                 checkpointable_layers: list[str] | None = None, activation_checkpoint_func=None):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.loss_fn = loss_fn
        self._ac_interval = int(activation_checkpoint_interval or 0)
        self._ac_func = activation_checkpoint_func
        self._checkpointable = set(checkpointable_layers or [])

    def _group_checkpointable(self, group: list) -> bool:
        # An empty filter means everything is checkpointable; otherwise a window is checkpointed
        # only if ALL its layers are of a listed type (matches DeepSpeed PipelineModule).
        if not self._checkpointable:
            return True
        return all(type(layer).__name__ in self._checkpointable for layer in group)

    def forward(self, x):
        if self._ac_interval <= 0 or self._ac_func is None:
            for layer in self.layers:
                x = layer(x)
            return x
        layers = list(self.layers)
        for start in range(0, len(layers), self._ac_interval):
            group = layers[start:start + self._ac_interval]
            if self._group_checkpointable(group):
                def run(inp, _group=group):
                    for layer in _group:
                        inp = layer(inp)
                    return inp

                x = self._ac_func(run, x)
            else:
                for layer in group:
                    x = layer(x)
        return x


class TorchEngine:
    """Minimal single-GPU training engine matching the DeepSpeed surface the loop uses."""

    def __init__(self, module: SequentialPipe, get_optimizer, parameters_to_train, ds_config: dict):
        self.module = module
        self.grid = _SingleGpuGrid()
        self.is_pipe_parallel = False
        self.num_stages = 1
        self.micro_batches = int(ds_config.get("gradient_accumulation_steps", 1))
        self._grad_clip = float(ds_config.get("gradient_clipping", 1.0) or 0.0)
        self.optimizer = get_optimizer(parameters_to_train)
        self.lr_scheduler = None
        self.communication_data_type = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(self.device)
        self._trainable = [p for p in self.module.parameters() if p.requires_grad]
        self._last_grad_norm: float | None = None

    # -- pipeline-stage queries (single GPU == both ends) --
    def is_first_stage(self) -> bool:
        return True

    def is_last_stage(self) -> bool:
        return True

    def reset_activation_shape(self) -> None:
        pass  # DeepSpeed caches pipeline activation shapes; nothing to reset here.

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def get_global_grad_norm(self):
        return self._last_grad_norm

    def _to_device(self, items):
        return tuple(
            t.to(self.device) if isinstance(t, torch.Tensor) else t for t in items
        )

    def _forward_loss(self, micro_batch):
        features, label = micro_batch
        x = self._to_device(features)
        label = self._to_device(label)
        for layer in self.module.layers:
            x = layer(x)
        return self.module.loss_fn(x, label)

    def train_batch(self, iterator) -> torch.Tensor:
        """Run ``micro_batches`` forward/backward, then one optimizer step. Returns mean loss."""
        self.module.train()
        total = 0.0
        for micro_batch in iterator:
            loss = self._forward_loss(micro_batch) / self.micro_batches
            loss.backward()
            total += loss.item()
        if self._grad_clip:
            self._last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self._trainable, self._grad_clip)
            )
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return torch.tensor(total)

    def eval_batch(self, iterator, num_micro_batches: int | None = None) -> torch.Tensor:
        self.module.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for micro_batch in iterator:
                total += self._forward_loss(micro_batch).item()
                n += 1
        return torch.tensor(total / max(n, 1))

    # -- checkpoint (resume state): trainable params + optimizer + scheduler + client_state --
    def save_checkpoint(
        self,
        save_dir,
        client_state: dict | None = None,
        save_latest: bool = True,
        exclude_frozen_parameters: bool = True,
        tag: str | None = None,
    ) -> None:
        client_state = client_state or {}
        tag = tag or f"global_step{client_state.get('step', 0)}"
        out = Path(save_dir) / tag
        out.mkdir(parents=True, exist_ok=True)
        module_sd = {
            name: p.detach().cpu()
            for name, p in self.module.named_parameters()
            if p.requires_grad or not exclude_frozen_parameters
        }
        torch.save(
            {
                "module": module_sd,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "client_state": client_state,
            },
            out / "torch_engine.pt",
        )
        if save_latest:
            (Path(save_dir) / "latest").write_text(tag)

    def load_checkpoint(
        self,
        load_dir,
        tag: str | None = None,
        load_module_strict: bool = False,
        load_lr_scheduler_states: bool = True,
        load_optimizer_states: bool = True,
        **_: Any,
    ):
        root = Path(load_dir)
        if tag is None:
            latest = root / "latest"
            if not latest.is_file():
                return None, None
            tag = latest.read_text().strip()
        ckpt_file = root / tag / "torch_engine.pt"
        if not ckpt_file.is_file():
            return None, None
        ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        own = dict(self.module.named_parameters())
        for name, tensor in ckpt["module"].items():
            if name in own:
                own[name].data.copy_(tensor.to(self.device))
        if load_optimizer_states and ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if load_lr_scheduler_states and self.lr_scheduler and ckpt.get("lr_scheduler"):
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        return str(ckpt_file), ckpt["client_state"]


# ------------------------------------------------------------------------------ construction


def build_pipe(backend: str, *, layers, num_stages, partition_method, manual_partition_split,
               loss_fn, extra_kw: dict):
    """Build the layer-holding module for *backend*. DeepSpeed import stays in its branch."""
    if backend == "deepspeed":
        from rengu_flow.utils.pipeline import ManualPipelineModule

        return ManualPipelineModule(
            layers=layers,
            num_stages=num_stages,
            partition_method=partition_method,
            manual_partition_split=manual_partition_split,
            loss_fn=loss_fn,
            **extra_kw,
        )
    if backend == "accelerate":
        if num_stages and num_stages > 1:
            raise SystemExit(
                "engine='accelerate' is single-GPU: pipeline_stages must be 1 "
                "(use engine='deepspeed' on Linux for multi-GPU)."
            )
        extra_kw = extra_kw or {}
        return SequentialPipe(
            layers,
            loss_fn,
            activation_checkpoint_interval=extra_kw.get("activation_checkpoint_interval", 0),
            checkpointable_layers=extra_kw.get("checkpointable_layers"),
            activation_checkpoint_func=extra_kw.get("activation_checkpoint_func"),
        )
    if backend == "accelerate_deepspeed":
        raise NotImplementedError(
            "engine='accelerate_deepspeed' (Accelerate+DeepSpeed ZeRO) is not implemented yet; "
            "use engine='deepspeed' (Linux multi-GPU) or engine='accelerate' (single-GPU)."
        )
    raise SystemExit(f"unknown engine backend {backend!r} (deepspeed|accelerate|accelerate_deepspeed)")


def build_engine(backend: str, *, pipeline_model, ds_config, args, get_optimizer,
                 parameters_to_train):
    """Construct the engine for *backend* and return it ready to train."""
    if backend == "deepspeed":
        import deepspeed

        engine, _, _, _ = deepspeed.initialize(args=args, model=pipeline_model, config=ds_config)
        engine._support_torch_style_backward = True
        engine._configure_optimizer(get_optimizer, parameters_to_train)
        return engine
    if backend == "accelerate":
        return TorchEngine(pipeline_model, get_optimizer, parameters_to_train, ds_config)
    if backend == "accelerate_deepspeed":
        raise NotImplementedError(
            "engine='accelerate_deepspeed' is not implemented yet; use 'deepspeed' or 'accelerate'."
        )
    raise SystemExit(f"unknown engine backend {backend!r}")
