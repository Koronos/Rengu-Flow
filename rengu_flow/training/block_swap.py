"""CPU block swap during training and inference (shared by SDXL UNet and Cosmos DiT)."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Protocol, runtime_checkable

import torch
from torch import nn


def patch_deepspeed_for_block_swap() -> None:
    """Make DeepSpeed leave block placement to the offloader.

    Block swap keeps the model mostly on CPU and streams blocks on demand, but DeepSpeed wants to
    (1) move the whole pipeline module onto the GPU at init — a transient multi-GB spike that spills
    to shared RAM on small cards — and (2) broadcast every parameter across the data-parallel group,
    which NCCL can't do for CPU-resident blocks. We neutralize both: ``PipelineModule.to`` becomes a
    no-op (the model's ``prepare_block_swap_training`` does placement), and the per-parameter
    ``_broadcast_model`` is skipped on a single rank (where it is a no-op anyway). Keeping this
    DeepSpeed-internals knowledge here, next to the offloader, rather than inline in the orchestrator.
    """
    import deepspeed.pipe as ds_pipe

    ds_pipe.PipelineModule.to = lambda self, *args, **kwargs: self
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        import deepspeed.runtime.engine as ds_engine

        ds_engine.DeepSpeedEngine._broadcast_model = lambda self: None


@runtime_checkable
class BlockSwapHandler(Protocol):
    def wait_for_block(self, block_idx: int) -> None: ...

    def submit_move_blocks_forward(self, block_idx: int) -> None: ...


class NoopOffloader:
    """Placeholder when block swap is disabled."""

    @property
    def enabled(self) -> bool:
        return False

    def wait_for_block(self, block_idx: int) -> None:
        pass

    def submit_move_blocks_forward(self, block_idx: int) -> None:
        pass

    def teardown(self) -> None:
        pass

    def apply_training_layout(self) -> None:
        pass


class BlockSwapOffloader:
    """Keep swapable blocks on CPU between forward steps; move one block to GPU per step."""

    def __init__(
        self,
        blocks: list[nn.Module] | nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device | str = "cuda",
    ):
        self.blocks: list[nn.Module] = list(blocks)
        self.num_blocks = len(self.blocks)
        self.blocks_to_swap = min(max(int(blocks_to_swap), 0), self.num_blocks)
        self.device = torch.device(device)
        self._enabled = self.blocks_to_swap > 0
        if self._enabled:
            self.apply_training_layout()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _move_block(block: nn.Module, device, *, non_blocking: bool = False) -> None:
        """Move a block by reassigning each param's ``.data`` and buffers — NOT ``nn.Module.to()``.

        For a bitsandbytes ``Params4bit`` weight, ``module.to()`` routes through bnb's ``.to``
        override, which mishandles the round-trip and corrupts the ``quant_state`` of the resident
        model (illegal memory access on the next use — the failure this preview offloader hit on a
        4-bit base). Moving ``.data`` shifts only the packed uint8 weight; the (tiny) ``quant_state``
        tensors stay put on the GPU, exactly as the training HookBlockSwapOffloader relies on."""
        for p in block.parameters():
            p.data = p.data.to(device, non_blocking=non_blocking)
        for b in block.buffers():
            b.data = b.data.to(device, non_blocking=non_blocking)

    def apply_training_layout(self) -> None:
        if not self._enabled:
            return
        for block in self.blocks:
            self._move_block(block, "cpu")

    def wait_for_block(self, block_idx: int) -> None:
        if not self._enabled:
            return
        self._move_block(self.blocks[block_idx], self.device, non_blocking=True)

    def submit_move_blocks_forward(self, block_idx: int) -> None:
        if not self._enabled:
            return
        self._move_block(self.blocks[block_idx], "cpu", non_blocking=True)

    def teardown(self) -> None:
        if not self._enabled:
            return
        for block in self.blocks:
            self._move_block(block, self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()


class HookBlockSwapOffloader:
    """On-demand, backward-aware CPU<->GPU block swap driven by module hooks.

    Unlike ``BlockSwapOffloader`` (which relies on a model's pipeline-layer ``forward`` calling
    ``wait_for_block``/``submit_move_blocks_forward``), this offloader registers hooks directly on
    the leaf modules of each swappable block, so it works even when ``to_layers`` flattens a block
    into several pipeline layers (as SDXL's UNet does). It keeps at most
    ``num_blocks - blocks_to_swap`` blocks resident on the GPU; a forward-pre and full-backward-pre
    hook pulls a block onto the GPU before it runs (covering activation-checkpointing recompute),
    and an LRU policy evicts the least-recently-used resident block to CPU when the cap is exceeded.

    Intended for **full fine-tuning with gradient_release**: each parameter's optimizer step runs
    inside the backward (``register_post_accumulate_grad_hook``) while its block is resident, so the
    weights are updated on-GPU before the block is evicted. A monolithic ``optimizer.step()`` would
    instead need every trainable block resident at once, which defeats the swap — hence the gate.
    Initial CPU layout is applied by ``apply_training_layout`` (call it *after* DeepSpeed has placed
    the model on the GPU, e.g. via ``prepare_block_swap_training``).
    """

    def __init__(
        self,
        blocks: list[nn.Module] | nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device | str = "cuda",
        prefetch: bool = False,
        swap_trainable: bool = True,
    ):
        self.blocks: list[nn.Module] = list(blocks)
        self.num_blocks = len(self.blocks)
        self.blocks_to_swap = min(max(int(blocks_to_swap), 0), self.num_blocks)
        self.resident_cap = max(1, self.num_blocks - self.blocks_to_swap)
        self.device = torch.device(device)
        self._enabled = self.blocks_to_swap > 0
        # When can a block's *trainable* params be evicted to CPU? Only when gradient_release runs
        # each parameter's optimizer step inside the backward (then no end-of-step grad reduction
        # touches them). Otherwise — adapter training, where the small trainable params live *inside*
        # the swapped blocks — DeepSpeed flattens all gradients at step end and would hit cpu+cuda
        # tensors. So with swap_trainable=False we keep requires_grad params GPU-resident and swap
        # only the frozen base weights (this is what makes block swap safe for LoRA/LoKr).
        self._swap_trainable = bool(swap_trainable)
        # block_swap_prefetch bundles two independent speedups; decouple them:
        #  * _pin  — pin the swapped weights' CPU storage so H2D copies run as full-bandwidth DMA
        #    (pageable copies stage through a hidden pinned buffer at ~half the bandwidth). Costs host
        #    RAM (page-locked), NOT VRAM, so it applies at ANY cap — even cap=1 / maximal swap.
        #  * _prefetch — additionally overlap the next block's H2D with the current block's compute on
        #    a side stream. Double-buffering means the running block AND the pulled-ahead block are
        #    co-resident, so it costs +1 resident block of VRAM and needs resident_cap >= 2.
        # Correctness never depends on either (ensure_resident always falls back to a blocking pull).
        self._pin = bool(prefetch) and self.device.type == "cuda"
        self._prefetch = self._pin and self.resident_cap >= 2
        self._resident: "OrderedDict[int, None]" = OrderedDict()
        self._block_of_module: dict[int, int] = {}
        self._handles: list = []
        self._stream = None
        self._cpu: dict[int, torch.Tensor] = {}      # id(param) -> pinned CPU storage
        self._gpu: dict[int, torch.Tensor] = {}      # id(param) -> GPU tensor while resident
        self._pull_event: dict[int, torch.cuda.Event] = {}  # block_idx -> pull-complete event
        self._params: dict[int, list] = {}           # block_idx -> [params]
        self._last_block: int | None = None
        if self._enabled:
            self._register_hooks()
            if self._prefetch:
                self._stream = torch.cuda.Stream(device=self.device)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _register_hooks(self) -> None:
        # The hooks move parameters between CPU and GPU — pure side effects that dynamo
        # cannot trace (fake tensors see two devices on the same op). compiler.disable makes
        # a compiled forward run them eagerly with a graph break at the block boundary; the
        # block interiors still compile, so blocks_to_swap composes with compile = true.
        fwd_hook = torch.compiler.disable(self._forward_pre_hook)
        bwd_hook = torch.compiler.disable(self._backward_pre_hook)
        for idx, block in enumerate(self.blocks):
            for module in block.modules():
                # The block root is hooked too when it directly owns parameters: krea2's
                # blocks keep their modulation table as a raw Parameter on the block and use
                # it BEFORE any leaf submodule runs, so a leaf-only hook set leaves it on
                # CPU at first use. Roots without direct params (cosmos, SDXL) are skipped
                # by the ownership check below, as before.
                if next(module.parameters(recurse=False), None) is None:
                    continue  # only modules that directly own parameters actually run/transfer
                self._block_of_module[id(module)] = idx
                self._handles.append(module.register_forward_pre_hook(fwd_hook))
                self._handles.append(module.register_full_backward_pre_hook(bwd_hook))

    def _swap_params(self, idx: int) -> list:
        """Params that physically move CPU<->GPU for this block. Full-model mode swaps everything;
        adapter mode swaps only the frozen base weights (trainable adapters stay GPU-resident)."""
        cached = self._params.get(idx)
        if cached is None:
            params = list(self.blocks[idx].parameters())
            if not self._swap_trainable:
                params = [p for p in params if not p.requires_grad]
            cached = params
            self._params[idx] = cached
        return cached

    def _offload_block_to_cpu(self, block: nn.Module) -> None:
        """Move a block to CPU. With ``swap_trainable`` move the whole block; otherwise keep the
        (small) trainable params GPU-resident and offload only the frozen weights + buffers, so an
        end-of-step gradient reduction never sees a trainable grad on CPU (adapter training)."""
        if self._swap_trainable:
            block.to("cpu")
            return
        for p in block.parameters():
            p.data = p.data.to(self.device if p.requires_grad else "cpu")
        for buf in block.buffers():
            buf.data = buf.data.to("cpu")

    # --------------------------------------------------------------- simple (synchronous) path
    def _ensure_resident_sync(self, block_idx: int) -> None:
        if block_idx in self._resident:
            self._resident.move_to_end(block_idx)
            return
        if self._pin:
            # DMA pull from the pinned CPU master on the default stream: faster H2D than a pageable
            # copy, but single-buffered (no compute overlap), so it fits at cap=1 / minimal VRAM.
            # Same-stream ordering guarantees the copy lands before the consuming kernel runs.
            for p in self._swap_params(block_idx):
                gpu = torch.empty_like(self._cpu[id(p)], device=self.device)
                gpu.copy_(self._cpu[id(p)], non_blocking=True)
                self._gpu[id(p)] = gpu
                p.data = gpu
        else:
            self.blocks[block_idx].to(self.device)
        self._resident[block_idx] = None
        while len(self._resident) > self.resident_cap:
            evict_idx, _ = self._resident.popitem(last=False)
            if self._pin:
                for p in self._swap_params(evict_idx):
                    gpu = self._gpu.pop(id(p), None)
                    if gpu is None:
                        continue
                    if self._swap_trainable:  # weights updated on GPU -> copy back; frozen are immutable
                        self._cpu[id(p)].copy_(gpu, non_blocking=True)
                    p.data = self._cpu[id(p)]
                continue
            self._offload_block_to_cpu(self.blocks[evict_idx])

    # --------------------------------------------------------------- overlapped (prefetch) path
    def _pull_async(self, idx: int) -> None:
        """Issue an H2D copy of block ``idx`` on the side stream (pinned -> fresh GPU tensors)."""
        if idx in self._resident:
            return
        event = torch.cuda.Event()
        with torch.cuda.stream(self._stream):
            for p in self._swap_params(idx):
                gpu = torch.empty_like(self._cpu[id(p)], device=self.device)
                gpu.copy_(self._cpu[id(p)], non_blocking=True)
                self._gpu[id(p)] = gpu
                p.data = gpu
            event.record(self._stream)
        self._pull_event[idx] = event
        self._resident[idx] = None

    def _evict_async(self, idx: int) -> None:
        """Release block ``idx`` from the GPU. In full-model mode the weights may have been updated
        on the GPU (gradient_release), so copy them back to the pinned CPU buffer on the side stream.
        In adapter mode the swapped weights are frozen (immutable), so skip the copy-back entirely —
        just drop the GPU tensor and point ``p.data`` back to the pinned CPU master. ``record_stream``
        defers the allocator's reuse of the freed GPU memory until in-flight compute finishes."""
        if self._swap_trainable:
            self._stream.wait_stream(torch.cuda.current_stream())  # block's compute must finish first
            with torch.cuda.stream(self._stream):
                for p in self._swap_params(idx):
                    gpu = self._gpu.get(id(p))
                    if gpu is None:
                        continue
                    self._cpu[id(p)].copy_(gpu, non_blocking=True)
                    p.data = self._cpu[id(p)]
                    gpu.record_stream(self._stream)
                    self._gpu.pop(id(p), None)
        else:
            for p in self._swap_params(idx):
                gpu = self._gpu.pop(id(p), None)
                if gpu is None:
                    continue
                p.data = self._cpu[id(p)]  # immutable frozen master — no D2H copy needed
                gpu.record_stream(torch.cuda.current_stream())
        self._resident.pop(idx, None)
        self._pull_event.pop(idx, None)

    def _ensure_resident_prefetch(self, block_idx: int, ahead: int) -> None:
        if block_idx == self._last_block:
            return  # still inside the same block (another leaf module) — nothing to do
        self._last_block = block_idx
        if block_idx not in self._resident:
            self._pull_async(block_idx)  # cold (mis-speculated): blocking-ish pull, still correct
        event = self._pull_event.pop(block_idx, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)  # compute waits for this block's H2D
        self._resident.move_to_end(block_idx)
        # Speculatively prefetch the next block in this direction.
        nxt = block_idx + ahead
        if 0 <= nxt < self.num_blocks and nxt not in self._resident:
            self._pull_async(nxt)
        # Evict oldest residents beyond the cap (never the block we just entered).
        while len(self._resident) > self.resident_cap:
            oldest = next(iter(self._resident))
            if oldest == block_idx:
                break
            self._evict_async(oldest)

    def _forward_pre_hook(self, module: nn.Module, args) -> None:
        idx = self._block_of_module[id(module)]
        if self._prefetch:
            self._ensure_resident_prefetch(idx, ahead=1)
        else:
            self._ensure_resident_sync(idx)

    def _backward_pre_hook(self, module: nn.Module, grad_output) -> None:
        idx = self._block_of_module[id(module)]
        if self._prefetch:
            self._ensure_resident_prefetch(idx, ahead=-1)
        else:
            self._ensure_resident_sync(idx)

    def apply_training_layout(self) -> None:
        """Push every swappable block to CPU; blocks are pulled back on demand by the hooks. When
        pinning is on (block_swap_prefetch) the swapped weights' CPU storage is page-locked so the
        on-demand H2D copies run as DMA — and, at cap>=2, can also overlap compute (prefetch)."""
        if not self._enabled:
            return
        for block in self.blocks:
            self._offload_block_to_cpu(block)
        if self._pin:
            # Pin only the params that actually swap (frozen base weights in adapter mode); the
            # trainable adapters stay GPU-resident and are never streamed, so they aren't pinned.
            self._cpu.clear()
            self._gpu.clear()
            for idx in range(self.num_blocks):
                for p in self._swap_params(idx):
                    p.data = p.data.pin_memory()
                    self._cpu[id(p)] = p.data
        self._resident.clear()
        self._pull_event.clear()
        self._last_block = None

    def teardown(self) -> None:
        """Remove hooks and move all blocks back to the GPU (for full-model eval / save)."""
        if self._stream is not None:
            torch.cuda.synchronize()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for block in self.blocks:
            block.to(self.device)
        self._resident.clear()
        self._gpu.clear()
        self._cpu.clear()
        self._pull_event.clear()
        self._last_block = None
        self._enabled = False

    # diffusion-pipe-style API compatibility: this offloader uses hooks, so the layer-driven
    # wait/submit calls (used by Cosmos's TransformerLayer.forward) are no-ops here.
    def wait_for_block(self, block_idx: int) -> None:
        pass

    def submit_move_blocks_forward(self, block_idx: int) -> None:
        pass
