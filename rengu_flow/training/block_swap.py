"""CPU block swap during training and inference (shared by SDXL UNet and Cosmos DiT)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol, runtime_checkable

import torch
from torch import nn


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

    def apply_training_layout(self) -> None:
        if not self._enabled:
            return
        for block in self.blocks:
            block.to("cpu")

    def wait_for_block(self, block_idx: int) -> None:
        if not self._enabled:
            return
        self.blocks[block_idx].to(self.device, non_blocking=True)

    def submit_move_blocks_forward(self, block_idx: int) -> None:
        if not self._enabled:
            return
        self.blocks[block_idx].to("cpu", non_blocking=True)

    def teardown(self) -> None:
        if not self._enabled:
            return
        for block in self.blocks:
            block.to(self.device, non_blocking=True)
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
    ):
        self.blocks: list[nn.Module] = list(blocks)
        self.num_blocks = len(self.blocks)
        self.blocks_to_swap = min(max(int(blocks_to_swap), 0), self.num_blocks)
        self.resident_cap = max(1, self.num_blocks - self.blocks_to_swap)
        self.device = torch.device(device)
        self._enabled = self.blocks_to_swap > 0
        # Prefetch needs room to hold the running block plus the one being pulled ahead, and a CUDA
        # device for the side stream. It overlaps the next block's H2D copy with the current block's
        # compute; correctness never depends on it (ensure_resident always falls back to a blocking
        # pull), so a mis-speculated prefetch under activation-checkpointing recompute only costs a
        # little speed, never correctness.
        self._prefetch = bool(prefetch) and self.resident_cap >= 2 and self.device.type == "cuda"
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
        for idx, block in enumerate(self.blocks):
            for module in block.modules():
                if module is block:
                    continue
                if next(module.parameters(recurse=False), None) is None:
                    continue  # only modules that directly own parameters actually run/transfer
                self._block_of_module[id(module)] = idx
                self._handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
                self._handles.append(module.register_full_backward_pre_hook(self._backward_pre_hook))

    def _block_params(self, idx: int) -> list:
        cached = self._params.get(idx)
        if cached is None:
            cached = list(self.blocks[idx].parameters())
            self._params[idx] = cached
        return cached

    # --------------------------------------------------------------- simple (synchronous) path
    def _ensure_resident_sync(self, block_idx: int) -> None:
        if block_idx in self._resident:
            self._resident.move_to_end(block_idx)
            return
        self.blocks[block_idx].to(self.device)
        self._resident[block_idx] = None
        while len(self._resident) > self.resident_cap:
            evict_idx, _ = self._resident.popitem(last=False)
            self.blocks[evict_idx].to("cpu")

    # --------------------------------------------------------------- overlapped (prefetch) path
    def _pull_async(self, idx: int) -> None:
        """Issue an H2D copy of block ``idx`` on the side stream (pinned -> fresh GPU tensors)."""
        if idx in self._resident:
            return
        event = torch.cuda.Event()
        with torch.cuda.stream(self._stream):
            for p in self._block_params(idx):
                gpu = torch.empty_like(self._cpu[id(p)], device=self.device)
                gpu.copy_(self._cpu[id(p)], non_blocking=True)
                self._gpu[id(p)] = gpu
                p.data = gpu
            event.record(self._stream)
        self._pull_event[idx] = event
        self._resident[idx] = None

    def _evict_async(self, idx: int) -> None:
        """Copy block ``idx`` back to its pinned CPU buffer on the side stream and free the GPU
        tensors safely (``record_stream`` defers reuse until the copy finishes)."""
        self._stream.wait_stream(torch.cuda.current_stream())  # block's compute must finish first
        with torch.cuda.stream(self._stream):
            for p in self._block_params(idx):
                gpu = self._gpu.get(id(p))
                if gpu is None:
                    continue
                self._cpu[id(p)].copy_(gpu, non_blocking=True)
                p.data = self._cpu[id(p)]
                gpu.record_stream(self._stream)
                self._gpu[id(p)] = None
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
        """Push every swappable block to CPU; blocks are pulled back on demand by the hooks. In
        prefetch mode the CPU storage is pinned so the on-demand H2D copies can overlap compute."""
        if not self._enabled:
            return
        for block in self.blocks:
            block.to("cpu")
        if self._prefetch:
            self._cpu.clear()
            self._gpu.clear()
            for block in self.blocks:
                for p in block.parameters():
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
