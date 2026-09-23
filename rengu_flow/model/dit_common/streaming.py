"""Lazily-loaded, layer-streamed frozen encoder (for text encoders larger than the GPU).

``LazyStreamedEncoder`` wraps a frozen inference-only encoder behind the plain ``nn.Module``
placement API the data layer and the preview lifecycle already use (``.to("cuda")``,
``.to("cpu")``, ``.to("meta")``, ``next(parameters()).device``):

- **Lazy**: the weights are read from disk on the first ``.to(<cuda>)`` — a run whose text
  embeddings are already cached never loads the encoder. ``.to("cpu")`` on an unloaded
  encoder is a no-op (the data layer parks idle submodels on CPU); ``.to("meta")`` unloads.
  While unloaded, a 0-size ``meta`` placeholder is the only parameter, so
  ``next(parameters()).device.type == "meta"`` (what ``DatasetManager`` / ``DiTPipeline``
  check) triggers the load exactly like a freed module.
- **Streamed**: when the encoder does not fit in free VRAM (``offload="auto"``) or always
  (``offload="stream"``), everything except the repeated decoder layers moves to the GPU
  and each layer's weights are copied from pinned host RAM right before it runs and dropped
  right after (forward hooks). All copies run on the current stream, so the allocator's
  stream ordering keeps them safe without events; the pinned masters are immutable (frozen),
  so nothing is copied back. Peak VRAM is the resident parts plus one layer.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

OFFLOAD_MODES = ("auto", "stream", "none")
# Headroom kept free next to a fully resident encoder before "auto" picks streaming:
# activations of a caching batch plus the allocator's fragmentation slack.
_AUTO_HEADROOM_BYTES = 2 * 1024**3


class LazyStreamedEncoder(nn.Module):
    def __init__(
        self,
        loader: Callable[[], nn.Module],
        layers_of: Callable[[nn.Module], Sequence[nn.Module]],
        offload: str = "auto",
        name: str = "text encoder",
    ):
        super().__init__()
        if offload not in OFFLOAD_MODES:
            raise ValueError(f"offload must be one of {OFFLOAD_MODES}, got {offload!r}")
        self._loader = loader
        self._layers_of = layers_of
        self.offload = offload
        self._name = name
        self.module: nn.Module | None = None
        self._placeholder = nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
        self._stream_device: torch.device | None = None
        self._masters: dict[int, torch.Tensor] = {}
        self._handles: list = []

    # ---- state ------------------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self.module is not None

    @property
    def is_streaming(self) -> bool:
        return self._stream_device is not None

    def load(self) -> nn.Module:
        if self.module is None:
            from rengu_flow.utils.common import is_main_process

            if is_main_process():
                print(f"rengu_flow: loading {self._name} weights...", flush=True)
            module = self._loader()
            module.eval().requires_grad_(False)
            self.module = module
            self._placeholder = None
        return self.module

    def unload(self) -> None:
        # Read before _stop_streaming(), which clears the masters dict.
        had_pinned = bool(self._masters)
        self._stop_streaming()
        self.module = None
        self._masters.clear()
        self._placeholder = nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
        if had_pinned and torch.cuda.is_available():
            # Freed pinned blocks stay in torch's caching host allocator (page-locked, still
            # counted against host RAM) until released: return the ~16 GB to the OS.
            import gc

            gc.collect()
            empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
            if empty_host_cache is not None:
                empty_host_cache()

    def forward(self, *args, **kwargs):
        return self.load()(*args, **kwargs)

    # ---- placement --------------------------------------------------------------------------

    def to(self, *args, **kwargs):
        device, dtype, _non_blocking, _fmt = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            raise TypeError(f"{type(self).__name__} is frozen at its load dtype; .to(dtype) is unsupported")
        if device is None:
            return self
        if device.type == "meta":
            self.unload()
        elif device.type == "cuda":
            module = self.load()
            if self._should_stream(module, device):
                self._start_streaming(module, device)
            else:
                self._stop_streaming()
                module.to(device)
        elif self.module is not None:
            self._stop_streaming()
            self.module.to(device)
        return self

    def _should_stream(self, module: nn.Module, device: torch.device) -> bool:
        if self.offload != "auto":
            return self.offload == "stream"
        if self._stream_device == device:
            return True  # already streaming; don't flip on a transient free-memory reading
        need = sum(t.numel() * t.element_size() for t in module.parameters() if t.device != device)
        free, _total = torch.cuda.mem_get_info(device)
        return need + _AUTO_HEADROOM_BYTES > free

    def _layer_tensor_ids(self, module: nn.Module) -> set[int]:
        return {id(t) for layer in self._layers_of(module) for t in layer.parameters()}

    def _start_streaming(self, module: nn.Module, device: torch.device) -> None:
        if self._stream_device == device:
            return
        self._stop_streaming()
        layer_ids = self._layer_tensor_ids(module)
        for p in module.parameters():
            if id(p) not in layer_ids:
                p.data = p.data.to(device)
        for b in module.buffers():  # rotary tables & co. are tiny: always resident
            b.data = b.data.to(device)
        for layer in self._layers_of(module):
            for p in layer.parameters():
                master = self._masters.get(id(p))
                if master is None:
                    master = p.data if p.data.device.type == "cpu" else p.data.to("cpu")
                    try:
                        master = master.pin_memory()  # DMA-speed H2D; host RAM, not VRAM
                    except RuntimeError:
                        pass  # pinning refused (locked-memory limits): pageable copies still work
                    self._masters[id(p)] = master
                p.data = master
            self._handles.append(layer.register_forward_pre_hook(self._pull_hook(layer, device)))
            self._handles.append(layer.register_forward_hook(self._drop_hook(layer)))
        self._stream_device = device
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _pull_hook(self, layer: nn.Module, device: torch.device):
        def hook(_module, _args):
            for p in layer.parameters():
                p.data = self._masters[id(p)].to(device, non_blocking=True)

        return hook

    def _drop_hook(self, layer: nn.Module):
        def hook(_module, _args, _output):
            for p in layer.parameters():
                p.data = self._masters[id(p)]

        return hook

    def _stop_streaming(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        # Params keep pointing at the masters; dropping the dict lets a later .to() free the pinned copies.
        self._masters.clear()
        self._stream_device = None


class LazyStreamedEncoderWithCompanion(LazyStreamedEncoder):
    """A ``LazyStreamedEncoder`` plus a small frozen *companion* module (e.g. a vision tower)
    that is only needed for some inputs.

    The companion is loaded on the first :meth:`load_companion` call — never by ``.to()`` or
    ``load()`` — so runs that never need it never read its weights. Once loaded it is always
    resident (never streamed) and follows the encoder's placement: ``.to(<cuda>)`` /
    ``.to("cpu")`` move it along, ``.to("meta")`` (unload) drops it. It is kept out of
    ``parameters()`` so the placement probes (``next(parameters()).device``) see only the encoder.
    """

    def __init__(
        self,
        loader: Callable[[], nn.Module],
        layers_of: Callable[[nn.Module], Sequence[nn.Module]],
        companion_loader: Callable[[], nn.Module],
        offload: str = "auto",
        name: str = "text encoder",
        companion_name: str = "companion",
    ):
        super().__init__(loader, layers_of, offload=offload, name=name)
        self._companion_loader = companion_loader
        self._companion_name = companion_name
        # A plain list keeps the companion out of the registered submodules / parameters().
        self._companion: list[nn.Module] = []

    @property
    def companion(self) -> nn.Module | None:
        return self._companion[0] if self._companion else None

    def _resident_device(self) -> torch.device:
        if self._stream_device is not None:
            return self._stream_device
        return next(self.load().parameters()).device

    def load_companion(self) -> nn.Module:
        """Load the companion (once) and place it next to the encoder's resident parts."""
        if not self._companion:
            from rengu_flow.utils.common import is_main_process

            if is_main_process():
                print(f"rengu_flow: loading {self._companion_name} weights...", flush=True)
            module = self._companion_loader()
            module.eval().requires_grad_(False)
            self._companion.append(module)
        module = self._companion[0]
        device = self._resident_device()
        if next(module.parameters()).device != device:
            module.to(device)
        return module

    def unload(self) -> None:
        super().unload()
        self._companion.clear()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device, _dtype, _non_blocking, _fmt = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type != "meta" and self._companion:
            self._companion[0].to(device)
        return self
