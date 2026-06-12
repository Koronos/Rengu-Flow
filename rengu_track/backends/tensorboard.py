"""TensorBoard backend: scalars / histograms / images to event files (default ON).

The heavy ``torch.utils.tensorboard`` import is deferred to ``__init__`` so merely importing
this module (e.g. from the UI reader) never drags in torch. Ignores metadata — config/hparams
live in the manifest, not TB's rigid hparams plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rengu_track.backends.base import Backend


class TensorBoardBackend(Backend):
    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def histogram(self, tag: str, values: Any, step: int) -> None:
        self._writer.add_histogram(tag, values, step)

    def image(self, tag: str, image: Any, step: int) -> None:
        self._writer.add_image(tag, image, step)

    def close(self, *, status: str | None = None) -> None:
        try:
            self._writer.flush()
            self._writer.close()
        except Exception:
            pass
