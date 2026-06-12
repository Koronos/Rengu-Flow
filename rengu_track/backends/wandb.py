"""WandB backend: optional mirror (default OFF).

Keeps the only ``import wandb`` in the codebase here, behind a backend, instead of inline in the
training loop. The import is deferred to ``__init__`` so the module stays light; ``build_sink``
catches ImportError and skips this backend when wandb isn't installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rengu_track.backends.base import Backend


class WandbBackend(Backend):
    def __init__(
        self,
        *,
        project: str,
        name: str,
        config: dict[str, Any],
        dir: str | Path,
        api_key: str | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        if api_key:
            wandb.login(key=api_key)
        wandb.init(project=project, name=name, config=config, dir=str(dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._wandb.log({tag: value, "step": step})

    def histogram(self, tag: str, values: Any, step: int) -> None:
        try:
            self._wandb.log({tag: self._wandb.Histogram(values), "step": step})
        except Exception:
            pass

    def image(self, tag: str, image: Any, step: int) -> None:
        try:
            self._wandb.log({tag: self._wandb.Image(image), "step": step})
        except Exception:
            pass

    def summary(self, metrics: dict[str, Any]) -> None:
        self._wandb.log(metrics)

    def close(self, *, status: str | None = None) -> None:
        try:
            self._wandb.finish()
        except Exception:
            pass
