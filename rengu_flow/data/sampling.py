"""Reusable, framework-free sampling primitives for the data layer.

Kept deliberately small and decoupled: each type owns exactly one concern and knows nothing
about datasets, DeepSpeed, resolutions, or the schedule. The dataset/loader compose these.
"""

from __future__ import annotations

import random


class RoundRobinCursor:
    """Cyclic, no-skip traversal of ``size`` items.

    Hands out indices ``0..size-1`` in a shuffled order, advancing a persistent position. When a
    full cycle completes it reshuffles (deterministically, from the next cycle's seed) so coverage
    stays even and the leftover of a partial cycle is spread across items instead of always
    falling on the same tail.

    This is the single piece behind the resolution-schedule fix: the dataset keeps one cursor per
    resolution bucket and *keeps it across stage changes*, so a resolution resumes where it left
    off instead of restarting from zero (which under-exposed the same images every time). It is
    pure and deterministic given ``(seed)`` — all data-parallel ranks build identical cursors, and
    ``state``/``set_state`` make resume exact.
    """

    def __init__(self, size: int, seed: int = 0):
        self._size = max(0, int(size))
        self._seed = int(seed)
        self._cycle = 0
        self._pos = 0
        self.total_drawn = 0
        self._order = self._shuffled(self._cycle)

    def _shuffled(self, cycle: int) -> list[int]:
        order = list(range(self._size))
        # Deterministic per (seed, cycle): same on every rank, and a resume reproduces it.
        random.Random(self._seed * 1_000_003 + int(cycle)).shuffle(order)
        return order

    def take(self, k: int) -> list[int]:
        """Return the next ``k`` item indices, cycling and reshuffling on each wrap."""
        out: list[int] = []
        if self._size <= 0:
            return out
        for _ in range(max(0, int(k))):
            if self._pos >= self._size:
                self._cycle += 1
                self._pos = 0
                self._order = self._shuffled(self._cycle)
            out.append(self._order[self._pos])
            self._pos += 1
            self.total_drawn += 1
        return out

    @property
    def size(self) -> int:
        return self._size

    def state(self) -> dict:
        """Serializable position (for checkpoint/resume)."""
        return {"seed": self._seed, "cycle": self._cycle, "pos": self._pos, "total": self.total_drawn}

    def set_state(self, state: dict) -> None:
        self._seed = int(state.get("seed", self._seed))
        self._cycle = int(state.get("cycle", 0))
        self._pos = int(state.get("pos", 0))
        self.total_drawn = int(state.get("total", 0))
        self._order = self._shuffled(self._cycle)
