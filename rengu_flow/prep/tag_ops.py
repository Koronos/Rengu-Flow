"""Pure tag operations over caption sets (no I/O).

Captions are ``{image_key: [line, ...]}`` where each line is a caption variant. Tag lines
are comma-separated tag lists; natural-language caption lines are left alone unless the op
explicitly targets them. Every op carries a line *scope* so edits can hit the canonical
first line, every tag line (propagating to pre-generated dropout variants), every line, or
one specific line — the caller chooses, nothing is forced to be regenerated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# WD-tagger convention: these tags ARE underscores — never turn them into spaces.
KAOMOJIS = frozenset(
    {
        "0_0",
        "(o)_(o)",
        "+_+",
        "+_-",
        "._.",
        "<o>_<o>",
        "<|>_<|>",
        "=_=",
        ">_<",
        "3_3",
        "6_9",
        ">_o",
        "@_@",
        "^_^",
        "o_o",
        "u_u",
        "x_x",
        "|_|",
        "||_||",
    }
)

SCOPE_LINE1 = "line1"
SCOPE_TAG_LINES = "tag_lines"
SCOPE_ALL_LINES = "all_lines"
SCOPE_LINE_N = "line_n"
SCOPES = (SCOPE_LINE1, SCOPE_TAG_LINES, SCOPE_ALL_LINES, SCOPE_LINE_N)


def replace_underscores(tag: str) -> str:
    """``long_hair`` -> ``long hair`` but kaomojis like ``^_^`` stay intact."""
    return tag if tag in KAOMOJIS else tag.replace("_", " ")


def parse_tags(line: str) -> list[str]:
    return [t.strip() for t in line.split(",") if t.strip()]


def join_tags(tags: list[str]) -> str:
    return ", ".join(tags)


def is_tag_line(line: str) -> bool:
    """Heuristic: comma-separated short tokens = tags; prose (sentences) = caption."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.endswith((".", "!", "?")) or ". " in stripped:
        return False
    parts = parse_tags(stripped)
    if not parts:
        return False
    if len(parts) == 1:
        return len(parts[0].split()) <= 4
    return sum(len(p.split()) for p in parts) / len(parts) <= 3.5


def line_indices_for_scope(
    lines: list[str], scope: str, line_index: int | None = None
) -> list[int]:
    if scope == SCOPE_LINE1:
        return [0] if lines else []
    if scope == SCOPE_ALL_LINES:
        return list(range(len(lines)))
    if scope == SCOPE_LINE_N:
        if line_index is None or line_index >= len(lines):
            return []
        return [line_index]
    if scope == SCOPE_TAG_LINES:
        return [i for i, line in enumerate(lines) if is_tag_line(line)]
    raise ValueError(f"Unknown scope: {scope} (expected one of {SCOPES})")


@dataclass(frozen=True)
class TagFilter:
    """Image-level tag filter: has all of, has any of, has none of."""

    all: tuple[str, ...] = ()
    any: tuple[str, ...] = ()
    none: tuple[str, ...] = ()

    @staticmethod
    def from_dict(data: dict | None) -> "TagFilter":
        data = data or {}
        return TagFilter(
            all=tuple(t.strip() for t in data.get("all", []) if t.strip()),
            any=tuple(t.strip() for t in data.get("any", []) if t.strip()),
            none=tuple(t.strip() for t in data.get("none", []) if t.strip()),
        )

    def is_empty(self) -> bool:
        return not (self.all or self.any or self.none)

    def matches(self, tags: set[str]) -> bool:
        lowered = {t.lower() for t in tags}
        if any(t.lower() not in lowered for t in self.all):
            return False
        if self.any and not any(t.lower() in lowered for t in self.any):
            return False
        if any(t.lower() in lowered for t in self.none):
            return False
        return True


def image_tags(
    lines: list[str], scope: str = SCOPE_TAG_LINES, line_index: int | None = None
) -> set[str]:
    """Union of tags across the scoped lines (used for filter matching and stats)."""
    tags: set[str] = set()
    for i in line_indices_for_scope(lines, scope, line_index):
        tags.update(parse_tags(lines[i]))
    return tags


def select_images(
    captions: dict[str, list[str]],
    tag_filter: TagFilter,
    scope: str = SCOPE_TAG_LINES,
    line_index: int | None = None,
) -> list[str]:
    return [
        key
        for key, lines in captions.items()
        if tag_filter.matches(image_tags(lines, scope, line_index))
    ]


def tag_frequencies(
    captions: dict[str, list[str]],
    scope: str = SCOPE_LINE1,
    line_index: int | None = None,
) -> Counter:
    """Count images containing each tag (canonical-case of first occurrence wins)."""
    counts: Counter = Counter()
    canonical: dict[str, str] = {}
    for lines in captions.values():
        for tag in image_tags(lines, scope, line_index):
            low = tag.lower()
            canonical.setdefault(low, tag)
            counts[low] += 1
    return Counter({canonical[low]: n for low, n in counts.items()})


OP_ADD = "add"
OP_REMOVE = "remove"
OP_RENAME = "rename"
OP_PRUNE = "prune"
OP_QUARANTINE = "quarantine"
OPS = (OP_ADD, OP_REMOVE, OP_RENAME, OP_PRUNE, OP_QUARANTINE)


@dataclass(frozen=True)
class TagEditOp:
    op: str
    tags: tuple[str, ...] = ()
    rename_to: str | None = None
    filter: TagFilter = field(default_factory=TagFilter)
    # Explicit image selection (e.g. from a size query). When set, the op applies to
    # these keys (further narrowed by `filter` if that is non-empty too).
    keys: tuple[str, ...] = ()
    scope: str = SCOPE_TAG_LINES
    line_index: int | None = None
    min_count: int | None = None
    position: str = "end"  # add: "start" | "end"

    @staticmethod
    def from_dict(data: dict) -> "TagEditOp":
        op = data.get("op", "")
        if op not in OPS:
            raise ValueError(f"Unknown op: {op!r} (expected one of {OPS})")
        scope = data.get("scope", SCOPE_TAG_LINES)
        if scope not in SCOPES:
            raise ValueError(f"Unknown scope: {scope!r} (expected one of {SCOPES})")
        tags = tuple(t.strip() for t in data.get("tags", []) if str(t).strip())
        if op in (OP_ADD, OP_REMOVE, OP_RENAME) and not tags:
            raise ValueError(f"Op {op!r} requires non-empty tags")
        if op == OP_RENAME and not (data.get("rename_to") or "").strip():
            raise ValueError("Op 'rename' requires rename_to")
        if op == OP_PRUNE and not isinstance(data.get("min_count"), int):
            raise ValueError("Op 'prune' requires integer min_count")
        keys = tuple(str(k) for k in data.get("keys", []) if str(k).strip())
        if (
            op == OP_QUARANTINE
            and TagFilter.from_dict(data.get("filter")).is_empty()
            and not keys
        ):
            raise ValueError("Op 'quarantine' requires a non-empty filter or explicit keys")
        return TagEditOp(
            op=op,
            tags=tags,
            rename_to=(data.get("rename_to") or "").strip() or None,
            filter=TagFilter.from_dict(data.get("filter")),
            keys=keys,
            scope=scope,
            line_index=data.get("line_index"),
            min_count=data.get("min_count"),
            position=data.get("position", "end"),
        )

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "tags": list(self.tags),
            "rename_to": self.rename_to,
            "filter": {
                "all": list(self.filter.all),
                "any": list(self.filter.any),
                "none": list(self.filter.none),
            },
            "keys": list(self.keys),
            "scope": self.scope,
            "line_index": self.line_index,
            "min_count": self.min_count,
            "position": self.position,
        }


@dataclass
class ApplyResult:
    captions: dict[str, list[str]]
    quarantined: list[str] = field(default_factory=list)
    changed_keys: list[str] = field(default_factory=list)


def _edit_line_tags(line: str, edit) -> str:
    return join_tags(edit(parse_tags(line)))


def _select_for_op(captions: dict[str, list[str]], op: TagEditOp) -> list[str]:
    """Images an op applies to: explicit keys (optionally narrowed by filter), or filter."""
    if op.keys:
        selected = [k for k in op.keys if k in captions]
        if not op.filter.is_empty():
            selected = [
                k
                for k in selected
                if op.filter.matches(image_tags(captions[k], op.scope, op.line_index))
            ]
        return selected
    if op.filter.is_empty():
        return list(captions)
    return select_images(captions, op.filter, op.scope, op.line_index)


def _apply_one(
    captions: dict[str, list[str]], op: TagEditOp
) -> tuple[dict[str, list[str]], list[str], set[str]]:
    new_captions = {key: list(lines) for key, lines in captions.items()}
    changed: set[str] = set()
    quarantined: list[str] = []

    if op.op == OP_QUARANTINE:
        quarantined = _select_for_op(new_captions, op)
        for key in quarantined:
            new_captions.pop(key)
        return new_captions, quarantined, set(quarantined)

    if op.op == OP_PRUNE:
        freqs = tag_frequencies(new_captions, op.scope, op.line_index)
        doomed = {tag.lower() for tag, n in freqs.items() if n < op.min_count}
        if op.tags:  # optional whitelist: never prune these
            doomed -= {t.lower() for t in op.tags}
        for key, lines in new_captions.items():
            for i in line_indices_for_scope(lines, op.scope, op.line_index):
                edited = _edit_line_tags(
                    lines[i], lambda tags: [t for t in tags if t.lower() not in doomed]
                )
                if edited != lines[i]:
                    lines[i] = edited
                    changed.add(key)
        return new_captions, quarantined, changed

    selected = _select_for_op(new_captions, op)
    wanted = {t.lower() for t in op.tags}
    for key in selected:
        lines = new_captions[key]
        indices = line_indices_for_scope(lines, op.scope, op.line_index)
        if op.op == OP_ADD and not indices:
            # No tag line to edit (empty caption, or prose-only with tag_lines scope):
            # insert a fresh tag line at the top instead of touching a caption line.
            if op.scope == SCOPE_LINE_N:
                continue
            lines.insert(0, join_tags(list(op.tags)))
            changed.add(key)
            continue
        for i in indices:
            tags = parse_tags(lines[i])
            present = {t.lower() for t in tags}
            if op.op == OP_ADD:
                missing = [t for t in op.tags if t.lower() not in present]
                tags = missing + tags if op.position == "start" else tags + missing
            elif op.op == OP_REMOVE:
                tags = [t for t in tags if t.lower() not in wanted]
            elif op.op == OP_RENAME:
                seen_new = op.rename_to.lower() in present
                renamed = []
                for t in tags:
                    if t.lower() in wanted:
                        if not seen_new:
                            renamed.append(op.rename_to)
                            seen_new = True
                    else:
                        renamed.append(t)
                tags = renamed
            edited = join_tags(tags)
            if edited != lines[i]:
                lines[i] = edited
                changed.add(key)
    return new_captions, quarantined, changed


def apply_ops(
    captions: dict[str, list[str]], ops: list[TagEditOp]
) -> ApplyResult:
    """Apply ops in order to a copy of ``captions``. Deterministic and side-effect free."""
    current = {key: list(lines) for key, lines in captions.items()}
    all_quarantined: list[str] = []
    all_changed: set[str] = set()
    for op in ops:
        current, quarantined, changed = _apply_one(current, op)
        all_quarantined.extend(quarantined)
        all_changed.update(changed)
    return ApplyResult(
        captions=current,
        quarantined=all_quarantined,
        changed_keys=sorted(all_changed),
    )


def diff_captions(
    before: dict[str, list[str]], after: dict[str, list[str]]
) -> list[dict]:
    """Per-image before/after lines for every image whose captions changed or vanished."""
    entries = []
    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        if old != new:
            entries.append({"key": key, "before": old, "after": new})
    return entries
