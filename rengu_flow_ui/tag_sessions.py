"""In-memory tag-editing sessions for the dataset-prep tag editor.

A session loads a folder's captions once, stages ops without touching disk, and only
writes on commit — after taking a full snapshot backup. This is the safety model that
motivated the prep module: a bad bulk edit is one restore away, and uncommitted staging
costs nothing (a server restart loses only the staged ops, never data).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from rengu_flow.prep.caption_store import CaptionSet, CaptionStore
from rengu_flow.prep.tag_ops import (
    TagEditOp,
    TagFilter,
    apply_ops,
    diff_captions,
    select_images,
    tag_frequencies,
)

SESSION_TTL_SECONDS = 6 * 3600


@dataclass
class TagSession:
    id: str
    captions: CaptionSet
    staged_ops: list[TagEditOp] = field(default_factory=list)
    last_access: float = field(default_factory=time.monotonic)
    _current_cache: tuple[int, dict] | None = None
    _sizes_cache: dict[str, tuple[int, int]] | None = None

    def sizes(self) -> dict[str, tuple[int, int]]:
        """Pixel dimensions per image (header-only reads; cached for the session)."""
        if self._sizes_cache is None:
            from PIL import Image

            sizes: dict[str, tuple[int, int]] = {}
            for key, path in self.captions.images.items():
                try:
                    with Image.open(path) as img:
                        sizes[key] = (int(img.width), int(img.height))
                except OSError:
                    sizes[key] = (0, 0)
            self._sizes_cache = sizes
        return self._sizes_cache

    @property
    def base(self) -> dict[str, list[str]]:
        return self.captions.captions

    def current(self) -> dict[str, list[str]]:
        """Captions with all staged ops applied (cached per ops-stack length)."""
        if self._current_cache and self._current_cache[0] == len(self.staged_ops):
            return self._current_cache[1]
        result = apply_ops(self.base, self.staged_ops)
        self._current_cache = (len(self.staged_ops), result.captions)
        return result.captions

    def quarantine_pending(self) -> list[str]:
        return sorted(set(self.base) - set(self.current()))

    def summary(self) -> dict:
        current = self.current()
        return {
            "session_id": self.id,
            "path": str(self.captions.folder),
            "format": self.captions.fmt,
            "ext": self.captions.ext,
            "image_count": len(current),
            "staged_ops": [op.to_dict() for op in self.staged_ops],
            "changed_count": len(diff_captions(self.base, current)),
            "quarantine_pending": self.quarantine_pending(),
        }


class TagSessionStore:
    def __init__(self, ttl_seconds: float = SESSION_TTL_SECONDS) -> None:
        self._sessions: dict[str, TagSession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def _evict_expired(self) -> None:
        now = time.monotonic()
        for sid in [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_access > self._ttl
        ]:
            del self._sessions[sid]

    def open(self, path: str, fmt: str = "sidecar", ext: str = ".txt") -> TagSession:
        captions = CaptionStore.open(path, fmt=fmt, ext=ext)
        session = TagSession(id=uuid.uuid4().hex, captions=captions)
        with self._lock:
            self._evict_expired()
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> TagSession:
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.last_access = time.monotonic()
            return session

    def close(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    # -- operations (all stage-only except commit) --------------------------------

    def stage_ops(self, session_id: str, ops: list[dict]) -> dict:
        session = self.get(session_id)
        parsed = [TagEditOp.from_dict(op) for op in ops]
        session.staged_ops.extend(parsed)
        return session.summary()

    def undo(self, session_id: str) -> dict:
        session = self.get(session_id)
        if session.staged_ops:
            session.staged_ops.pop()
        return session.summary()

    def stats(self, session_id: str, scope: str = "line1") -> dict:
        session = self.get(session_id)
        freqs = tag_frequencies(session.current(), scope=scope)
        return {
            "tags": [
                {"tag": tag, "count": count}
                for tag, count in sorted(
                    freqs.items(), key=lambda kv: (-kv[1], kv[0].lower())
                )
            ],
            "image_count": len(session.current()),
        }

    def query(
        self,
        session_id: str,
        tag_filter: dict,
        scope: str = "tag_lines",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict:
        session = self.get(session_id)
        current = session.current()
        all_keys = select_images(current, TagFilter.from_dict(tag_filter), scope=scope)
        # Paginate: a filter can match hundreds of thousands of images — only the requested
        # page's keys/captions (and the route's previews) are returned. ``total`` is the count.
        keys = all_keys[offset : offset + limit] if limit is not None else all_keys
        return {
            "total": len(all_keys),
            "offset": offset,
            "limit": limit if limit is not None else len(all_keys),
            "keys": keys,
            "captions": {key: current[key] for key in keys},
        }

    def size_query(
        self,
        session_id: str,
        *,
        below: int | None = None,
        above: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict:
        """Images whose SHORT side is < ``below`` and/or whose LONG side is > ``above``.

        The trainer's bucketing resizes later, so this is the prep-time filter for
        thumbnails that caption/train badly and oversized originals worth flagging. Paginated
        like ``query`` so it scales to very large datasets.
        """
        session = self.get(session_id)
        current = session.current()
        sizes = session.sizes()
        all_keys = []
        for key in current:
            w, h = sizes.get(key, (0, 0))
            if below is not None and min(w, h) >= below:
                continue
            if above is not None and max(w, h) <= above:
                continue
            if below is None and above is None:
                continue
            all_keys.append(key)
        all_keys.sort(key=lambda k: min(sizes.get(k, (0, 0))))
        keys = all_keys[offset : offset + limit] if limit is not None else all_keys
        return {
            "total": len(all_keys),
            "offset": offset,
            "limit": limit if limit is not None else len(all_keys),
            "keys": keys,
            "captions": {key: current[key] for key in keys},
            "sizes": {key: list(sizes.get(key, (0, 0))) for key in keys},
        }

    def diff(self, session_id: str, limit: int | None = None) -> dict:
        session = self.get(session_id)
        entries = diff_captions(session.base, session.current())
        total = len(entries)
        if limit is not None:
            entries = entries[:limit]
        return {"total": total, "entries": entries}

    def commit(self, session_id: str) -> dict:
        """Snapshot, then write staged state to disk; the session resets on the new base."""
        session = self.get(session_id)
        if not session.staged_ops:
            raise ValueError("Nothing staged to commit")
        captions = session.captions
        current = session.current()
        backup_dir = captions.snapshot()

        quarantined = session.quarantine_pending()
        if quarantined:
            captions.quarantine(quarantined)
        for key, lines in current.items():
            if captions.get_lines(key) != lines:
                captions.set_lines(key, lines)
        written = captions.save()

        session.staged_ops.clear()
        session._current_cache = None
        return {
            "backup": backup_dir.name,
            "backup_path": str(backup_dir),
            "files_written": written,
            "quarantined": quarantined,
        }
