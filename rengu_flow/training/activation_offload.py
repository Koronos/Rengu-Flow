"""Stream saved activations to pinned CPU RAM instead of keeping them in VRAM.

``activation_offload = true`` intercepts autograd's saved tensors
(``torch.autograd.graph.saved_tensors_hooks``): during the forward each large
contiguous saved activation is copied to a pinned CPU buffer on a side CUDA
stream and its GPU memory is released as soon as the copy lands; during the
backward the tensors are copied back on a second side stream, prefetched in
reverse order ahead of where the backward is running. Both directions overlap
GPU compute, so the cost is PCIe bandwidth — not the recompute FLOPs that
activation checkpointing pays.

This composes with ``activation_memory_budget`` (and is the reason it can be
raised): the budget decides which activations are *saved* vs recomputed, the
offloader decides where the saved ones *live*. Works in eager and under
torch.compile (AOTAutograd routes the compiled graph's saved tensors through
the same hooks). Not compatible with CUDA-graph capture
(``compile_mode = "reduce-overhead"``), which cannot record the side-stream
copies.

Design notes (the parts that took measurement to get right):

- A saved tensor's GPU storage is released on the *GPU's* timeline, not
  Python's: the reference is dropped at pack time with
  ``record_stream(d2h)``, so the allocator can reuse the block the moment the
  copy lands. Python races a whole forward ahead of the GPU (every pack hook
  has fired before the first block finishes computing), so any scheme that
  frees from Python callbacks holds the entire saved set until the backward —
  measured: zero peak reduction.
- All side-stream staging tensors are allocated *inside* their stream's
  context. The caching allocator orders block reuse per stream; allocating on
  the current stream and writing on a side stream lets the copy clobber a
  block an in-flight kernel is still reading (measured: NaN losses).
- Only contiguous tensors are offloaded: the compiled backward asserts each
  saved tensor's exact sizes/strides, which a round-trip through a compact CPU
  buffer cannot reproduce for non-contiguous views.
- Pinned buffers are pooled and recycled within and across steps; reuse is
  gated on the CUDA event of the last copy touching the buffer. A *cold*
  buffer (first step on a shape) instead drains synchronously: pinned
  allocation is slow, and the backlog would otherwise push the cold step —
  which already pays compile transients — to a no-offload peak.

Each saved tensor may be unpacked once per backward; double backward /
``retain_graph`` is not supported and raises.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterable

import torch


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class _PinnedPool:
    """Reusable pinned 1-D CPU buffers keyed by (numel, dtype).

    Each free entry carries the CUDA event of the last copy that touched it;
    ``take`` makes the next writing stream wait on that event so a recycled
    buffer is never overwritten while a copy involving it is still in flight.
    """

    def __init__(self, pin: bool) -> None:
        self._pin = pin
        self._free: dict[tuple[int, torch.dtype], list[tuple[torch.Tensor, Any]]] = {}
        self.allocated_bytes = 0

    def take(self, numel: int, dtype: torch.dtype, wait_stream: Any = None) -> torch.Tensor:
        entry = self._free.get((numel, dtype))
        if entry:
            buf, event = entry.pop()
            if event is not None and wait_stream is not None:
                wait_stream.wait_event(event)
            return buf
        buf = torch.empty(numel, dtype=dtype, pin_memory=self._pin)
        self.allocated_bytes += numel * dtype.itemsize
        return buf

    def give(self, buf: torch.Tensor, event: Any = None) -> None:
        self._free.setdefault((buf.numel(), buf.dtype), []).append((buf, event))


class _Record:
    """One offloaded saved tensor."""

    __slots__ = ("cpu", "shape", "dtype", "numel", "d2h_event", "gpu", "h2d_event", "idx")

    def __init__(self, shape, dtype, numel, idx):
        self.shape = shape
        self.dtype = dtype
        self.numel = numel
        self.idx = idx
        self.cpu = None
        self.d2h_event = None
        self.gpu = None
        self.h2d_event = None


class ActivationOffloader:
    """Saved-tensor offload to pinned CPU RAM with stream-overlapped transfers.

    ``sync=True`` replaces the side-stream transport with plain blocking copies
    and no CUDA requirement — used by tests and as a debugging mode; the policy
    (which tensors offload, RAM cap, stats) is identical.
    """

    def __init__(
        self,
        *,
        min_tensor_mb: float = 4.0,
        max_ram_gb: float | None = None,
        prefetch_mb: float = 512.0,
        params_provider: Callable[[], Iterable[torch.nn.Parameter]] | None = None,
        sync: bool = False,
        verbose: bool = True,
    ) -> None:
        self.min_tensor_bytes = int(min_tensor_mb * (1 << 20))
        self.max_ram_bytes = None if max_ram_gb is None else int(max_ram_gb * (1 << 30))
        self.prefetch_bytes = int(prefetch_mb * (1 << 20))
        self._params_provider = params_provider
        self._sync = sync
        self._verbose = verbose
        self._pool = _PinnedPool(pin=not sync)
        self._records: list[_Record] = []
        self._param_ptrs: set[int] = set()
        self._d2h = None
        self._h2d = None
        if not sync:
            self._d2h = torch.cuda.Stream()
            self._h2d = torch.cuda.Stream()
        # stats (per wrapped step; *_total accumulate over the run)
        self.packed_count = 0
        self.packed_bytes = 0
        self.kept_count = 0
        self.kept_bytes = 0
        self.steps = 0
        self.packed_bytes_total = 0
        self._cap_hit_logged = False

    # ------------------------------------------------------------------ policy
    def _should_offload(self, t: torch.Tensor) -> bool:
        if not self._sync and t.device.type != "cuda":
            return False
        if _tensor_bytes(t) < self.min_tensor_bytes:
            return False
        if t.data_ptr() in self._param_ptrs:
            return False  # weights are saved tensors too; offloading them would thrash
        if not t.is_contiguous():
            # The compiled backward asserts each saved tensor's exact
            # sizes/strides; a round-trip through a compact CPU buffer cannot
            # reproduce a non-contiguous view's strides.
            return False
        if (
            self.max_ram_bytes is not None
            and self._pool.allocated_bytes + _tensor_bytes(t) > self.max_ram_bytes
        ):
            if self._verbose and not self._cap_hit_logged:
                self._cap_hit_logged = True
                print(
                    f"[act-offload] pinned RAM cap ({self.max_ram_bytes / 1e9:.1f} GB) "
                    "reached; further activations stay in VRAM. Raise "
                    "activation_offload_max_ram_gb for more VRAM savings.",
                    flush=True,
                )
            return False
        return True

    # --------------------------------------------------------------- transport
    def _pack(self, t: torch.Tensor):
        if not self._should_offload(t):
            self.kept_count += 1
            self.kept_bytes += _tensor_bytes(t) if t.device.type != "cpu" else 0
            return ("keep", t)
        self.packed_count += 1
        self.packed_bytes += _tensor_bytes(t)
        rec = _Record(t.shape, t.dtype, t.numel(), len(self._records))
        if self._sync:
            rec.cpu = self._pool.take(rec.numel, t.dtype)
            rec.cpu.copy_(t.reshape(-1))
        else:
            pool_bytes_before = self._pool.allocated_bytes
            rec.cpu = self._pool.take(rec.numel, t.dtype, wait_stream=self._d2h)
            self._d2h.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._d2h):
                rec.cpu.copy_(t.reshape(-1), non_blocking=True)
                rec.d2h_event = torch.cuda.Event()
                rec.d2h_event.record(self._d2h)
            # Release the GPU storage on the GPU's own timeline, not Python's:
            # record_stream defers the block's reuse until the d2h stream
            # passes the copy, and the allocator re-checks those events on
            # every allocation — so the VRAM becomes reusable the moment the
            # copy lands, even though Python (which races ~one full forward
            # ahead of the GPU) won't run any of our code again until the
            # backward starts. Holding the reference and freeing from pack
            # hooks instead was measured to keep the WHOLE saved set resident
            # through the forward (no Python runs between the last pack and
            # the first unpack).
            t.record_stream(self._d2h)
            if self._pool.allocated_bytes > pool_bytes_before:
                # Cold buffer (first step on this shape): pinned allocation is
                # slow, so the queued copies lag far behind and the not-yet-
                # reusable blocks pile up to a no-offload peak — exactly when
                # the step also pays compile transients. Draining synchronously
                # keeps the cold step's peak at the steady-state level; it only
                # costs time on that one (already slow, compiling) step.
                rec.d2h_event.synchronize()
        self._records.append(rec)
        return ("cpu", rec)

    def _fetch(self, rec: _Record) -> None:
        """Issue the H2D copy of ``rec`` on the side stream (idempotent)."""
        if rec.gpu is not None or rec.cpu is None:
            return
        # Allocate INSIDE the h2d stream context: the caching allocator orders
        # block reuse per stream, so a buffer allocated (and written) on the
        # current stream's timeline could be clobbered by this copy while an
        # in-flight kernel still reads the block's previous tensor. The
        # consumer on the current stream calls record_stream at handoff.
        self._h2d.wait_event(rec.d2h_event)  # its D2H write must land first
        with torch.cuda.stream(self._h2d):
            gpu = torch.empty(rec.shape, dtype=rec.dtype, device="cuda")
            gpu.reshape(-1).copy_(rec.cpu, non_blocking=True)
            rec.h2d_event = torch.cuda.Event()
            rec.h2d_event.record(self._h2d)
        rec.gpu = gpu

    def _unpack(self, packed):
        kind, payload = packed
        if kind == "keep":
            return payload
        rec: _Record = payload
        if rec.cpu is None and rec.gpu is None:
            raise RuntimeError(
                "activation_offload: saved tensor unpacked twice (double backward / "
                "retain_graph is not supported with activation_offload)."
            )
        if self._sync:
            out = rec.cpu[: rec.numel].reshape(rec.shape).clone()
            self._pool.give(rec.cpu)
            rec.cpu = None
            return out
        # Backward consumes saved tensors roughly in reverse forward order:
        # pull older records back while this one's consumer computes.
        budget = self.prefetch_bytes
        i = rec.idx - 1
        while i >= 0 and budget > 0:
            older = self._records[i]
            if older.cpu is not None and older.gpu is None:
                self._fetch(older)
                budget -= older.numel * older.dtype.itemsize
            i -= 1
        self._fetch(rec)
        torch.cuda.current_stream().wait_event(rec.h2d_event)
        out = rec.gpu
        # The tensor was allocated on the h2d stream; its free happens on the
        # current stream's timeline, so defer the block's reuse until current-
        # stream work has passed this point.
        out.record_stream(torch.cuda.current_stream())
        rec.gpu = None
        self._pool.give(rec.cpu, event=rec.h2d_event)
        rec.cpu = None
        return out

    # ------------------------------------------------------------------- step
    @contextlib.contextmanager
    def step(self):
        """Wrap one optimizer step (forward+backward, all micro-batches)."""
        if self._params_provider is not None:
            # Re-snapshot every step: block swap moves params between CPU and
            # GPU, changing data_ptrs.
            self._param_ptrs = {p.data_ptr() for p in self._params_provider()}
        self.packed_count = 0
        self.packed_bytes = 0
        self.kept_count = 0
        self.kept_bytes = 0
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            yield
        # Anything never unpacked (e.g. saved for a grad that was not needed)
        # goes back to the pool; its D2H may still be in flight, so gate reuse
        # on that event.
        for rec in self._records:
            if rec.cpu is not None:
                self._pool.give(rec.cpu, event=rec.d2h_event)
                rec.cpu = None
            rec.gpu = None
        self._records.clear()
        self.steps += 1
        self.packed_bytes_total += self.packed_bytes
        if self.steps == 1 and self._verbose:
            if self.packed_count == 0:
                print(
                    "[act-offload] WARNING: no saved activations were intercepted "
                    "this step - activation_offload is doing nothing. Causes: every "
                    "saved tensor is below activation_offload_min_tensor_mb / "
                    "non-contiguous, or this torch build does not route compiled "
                    "saved tensors through saved_tensors_hooks.",
                    flush=True,
                )
            else:
                print(f"[act-offload] {self.describe_step()}", flush=True)

    def describe_step(self) -> str:
        return (
            f"offloaded {self.packed_count} activations, "
            f"{self.packed_bytes / 1e9:.2f} GB/step to pinned RAM "
            f"(pool {self._pool.allocated_bytes / 1e9:.2f} GB, "
            f"{self.kept_count} tensors / {self.kept_bytes / 1e9:.2f} GB kept in VRAM)"
        )

    # ------------------------------------------------------------------ config
    @classmethod
    def from_config(
        cls,
        config: dict,
        params_provider: Callable[[], Iterable[torch.nn.Parameter]] | None = None,
    ) -> "ActivationOffloader | None":
        """Build from the training config; returns None when disabled."""
        if not config.get("activation_offload", False):
            return None
        if not torch.cuda.is_available():
            raise ValueError(
                "activation_offload = true requires CUDA (it streams activations "
                "between GPU and pinned RAM over side streams)."
            )
        if config.get("compile_mode") == "reduce-overhead":
            raise ValueError(
                "activation_offload is incompatible with compile_mode = "
                "'reduce-overhead' (CUDA-graph capture cannot record the "
                "side-stream CPU copies)."
            )
        min_tensor_mb = float(config.get("activation_offload_min_tensor_mb", 4.0))
        if min_tensor_mb < 0:
            raise ValueError("activation_offload_min_tensor_mb must be >= 0.")
        max_ram_gb = config.get("activation_offload_max_ram_gb")
        if max_ram_gb is not None:
            max_ram_gb = float(max_ram_gb)
            if max_ram_gb <= 0:
                raise ValueError("activation_offload_max_ram_gb must be > 0 (or unset).")
        prefetch_mb = float(config.get("activation_offload_prefetch_mb", 512.0))
        if prefetch_mb < 0:
            raise ValueError("activation_offload_prefetch_mb must be >= 0.")
        return cls(
            min_tensor_mb=min_tensor_mb,
            max_ram_gb=max_ram_gb,
            prefetch_mb=prefetch_mb,
            params_provider=params_provider,
        )
