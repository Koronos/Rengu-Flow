"""Backend base class: the sink fans every metric call out to each backend.

Every method is a no-op by default; a backend overrides only what it handles (the TB backend
ignores ``set_metadata``; the manifest backend ignores per-step scalars/images). This keeps the
sink's fan-out uniform — it never needs to know which backend cares about what.
"""

from __future__ import annotations

from typing import Any


class Backend:
    """No-op metric backend. Subclasses override the calls they persist."""

    def scalar(self, tag: str, value: float, step: int) -> None: ...

    def histogram(self, tag: str, values: Any, step: int) -> None: ...

    def image(self, tag: str, image: Any, step: int) -> None: ...

    def set_metadata(
        self,
        *,
        config: dict[str, Any] | None = None,
        lineage: dict[str, Any] | None = None,
        hardware: dict[str, Any] | None = None,
    ) -> None: ...

    def summary(self, metrics: dict[str, Any]) -> None: ...

    def close(self, *, status: str | None = None) -> None: ...
