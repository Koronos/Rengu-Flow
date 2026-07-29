"""Single-GPU plain-torch engine ("accelerate")."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from rengu_flow.engine.base import Engine, TrainingBackend


def _is_adapter(config: dict) -> bool:
    return bool(config.get("adapter"))


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
                # Whether the state is a tuple must be captured HERE, not inferred from
                # len(inp) inside run(): a bare tensor and a 1-tuple both arrive as a
                # single positional arg, and guessing "1 arg means bare tensor" hands a
                # raw tensor to layers that expect the (x,) pipeline convention.
                def run(*inp, _group=group, _tupled=isinstance(x, tuple)):
                    state = inp if _tupled else inp[0]
                    for layer in _group:
                        state = layer(state)
                    return state

                # Unpack the inter-layer tuple into positional tensor args. Under
                # use_reentrant=True (auto-enabled for block-swap + quantized base),
                # checkpoint only inspects top-level tensor args for requires_grad; a
                # tuple hides InitialLayer's requires_grad_(True) boundary tensors, so the
                # recomputed segment is severed from autograd and every adapter param
                # inside gets no gradient (silent training stall). The inter-layer state is
                # tensors-only by design, so *x is safe for both reentrant modes.
                args = x if isinstance(x, tuple) else (x,)
                x = self._ac_func(run, *args)
            else:
                for layer in group:
                    x = layer(x)
        return x


class TorchEngine:
    """Minimal single-GPU training engine matching the DeepSpeed surface the loop uses."""

    # DeepSpeed's pipeline engine needs all micro-batches materialized before pipeline IPC.
    # The single-device engine consumes them in order, so keeping them lazy removes a redundant
    # list allocation and avoids holding a full accumulation step of batch references at once.
    preload_micro_batches = False

    def __init__(self, module: SequentialPipe, get_optimizer, parameters_to_train, ds_config: dict,
                 *, block_swap: bool = False):
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
        # With block swap the model is deliberately kept mostly on CPU; placement is owned by
        # model.prepare_block_swap_training() (called right after build_engine) and the offloader's
        # hooks stream blocks on demand. Moving the whole module to the GPU here would defeat the swap
        # (a transient full-model spike → OOM on exactly the cards block swap exists for).
        if not block_swap:
            self.module.to(self.device)
        self._trainable = [p for p in self.module.parameters() if p.requires_grad]
        self._last_grad_norm: torch.Tensor | float | None = None

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
            t.to(self.device, non_blocking=self.device.type == "cuda")
            if isinstance(t, torch.Tensor)
            else t
            for t in items
        )

    def _forward_loss(self, micro_batch):
        features, label = micro_batch
        x = self._to_device(features)
        label = self._to_device(label)
        # Route through SequentialPipe.forward (NOT a manual layer loop) so activation checkpointing
        # actually runs: without it the full-model autograd graph pins every base weight for the
        # backward, which defeats block swap (evicted weights can't be freed). Then apply loss_fn.
        output = self.module(x)
        return self.module.loss_fn(output, label)

    def train_batch(self, iterator) -> torch.Tensor:
        """Run ``micro_batches`` forward/backward, then one optimizer step. Returns mean loss."""
        # Module.train() recursively visits every child. Large diffusion pipelines can contain
        # hundreds of modules, so only walk the tree when eval actually changed its mode.
        if not self.module.training:
            self.module.train()
        # Accumulate the loss on-GPU (loss.detach(), not .item()) so we don't force a
        # cudaStreamSynchronize every micro-batch. That per-micro-batch sync stalls the CPU
        # behind the GPU and defeats the CPU-runs-ahead overlap torch.compile relies on; the
        # single sync happens later when the caller reads the returned scalar.
        total = None
        for micro_batch in iterator:
            loss = self._forward_loss(micro_batch) / self.micro_batches
            loss.backward()
            total = loss.detach() if total is None else total + loss.detach()
        if self._grad_clip:
            # Keep the scalar on-device. Converting it to float here synchronizes CUDA before the
            # optimizer step; metrics resolve it later, after the caller's unavoidable loss sync.
            self._last_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._trainable, self._grad_clip
            ).detach()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return total if total is not None else torch.tensor(0.0)

    def eval_batch(self, iterator, num_micro_batches: int | None = None) -> torch.Tensor:
        if self.module.training:
            self.module.eval()
        total, n = None, 0
        with torch.no_grad():
            for micro_batch in iterator:
                loss = self._forward_loss(micro_batch).detach()
                total = loss if total is None else total + loss
                n += 1
        return (total / n) if total is not None else torch.tensor(0.0)

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
        if num_stages and num_stages > 1:
            raise SystemExit("engine='accelerate' is single-stage; set pipeline_stages = 1.")
        extra_kw = extra_kw or {}
        return SequentialPipe(
            layers,
            loss_fn,
            activation_checkpoint_interval=extra_kw.get("activation_checkpoint_interval", 0),
            checkpointable_layers=extra_kw.get("checkpointable_layers"),
            activation_checkpoint_func=extra_kw.get("activation_checkpoint_func"),
        )

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
