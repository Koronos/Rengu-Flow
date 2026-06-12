"""Tracking backends. Backend classes defer their heavy imports (torch, wandb) to ``__init__``,
so importing these modules stays cheap."""

from rengu_track.backends.base import Backend

__all__ = ["Backend"]
