"""Parquet-backed image sources for DirectoryDataset.

Parquet files in a dataset directory are enumerated like tars: each row becomes an
``image_spec = (parquet_path, "pq/{row_index}")``. Image bytes are read on demand
through a per-file row-group cache (columnar, near-sequential access during caching),
captions come from a caption column at enumeration time (columnar scan, no image
bytes touched), and optional width/height columns let the metadata stage skip
decoding image headers entirely.

Supported column shapes (auto-detected, overridable via directory config):
  image:   binary, or struct with a ``bytes`` field (HF datasets image format)
  caption: string | list<string> | list<struct{from,value}> (LLaVA chat: first
           non-human turn wins)
  width/height: any integer type (optional fast path)

Config keys on a ``[[directory]]`` entry (all optional):
  parquet_image_column   (default "image")
  parquet_caption_column (default: "caption" if present, else "conversations")
  parquet_width_column / parquet_height_column (default "width"/"height")
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

MEMBER_PREFIX = "pq/"


def is_parquet_spec(spec) -> bool:
    return spec[0] is not None and str(spec[0]).endswith(".parquet")


def spec_row(spec) -> int:
    return int(str(spec[1]).removeprefix(MEMBER_PREFIX))


def _caption_from_cell(cell) -> list[str]:
    """Normalize a caption cell to rengu's list-of-captions-per-image."""
    if cell is None:
        return [""]
    if isinstance(cell, str):
        return [cell]
    if isinstance(cell, list):
        if not cell:
            return [""]
        if isinstance(cell[0], str):
            return [str(c) for c in cell]
        if isinstance(cell[0], dict) and "value" in cell[0]:
            # LLaVA/ShareGPT chat rows: first non-human turn is the caption.
            for turn in cell:
                if str(turn.get("from", "")).lower() not in ("human", "user"):
                    return [str(turn.get("value") or "")]
            return [""]
    return [str(cell)]


def _image_bytes_from_cell(cell) -> bytes | None:
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):  # HF image struct {bytes, path}
        b = cell.get("bytes")
        return bytes(b) if b is not None else None
    return None


class ParquetSource:
    """Lazy reader for one or more parquet files (open handles + row-group cache).

    Mirrors the tarfile_map pattern: instances are created per consumer (enumeration,
    metadata map, media preprocessor) and open pyarrow handles lazily, so forked
    dataloader/map workers each get their own file handles on first use. Only the
    image column of ONE row group per file is kept decoded (near-sequential access
    during caching means each group is decoded ~once).
    """

    def __init__(self, directory_config: dict | None = None):
        cfg = directory_config or {}
        self.image_col = cfg.get("parquet_image_column", "image")
        self.caption_col = cfg.get("parquet_caption_column")  # None = auto
        self.width_col = cfg.get("parquet_width_column", "width")
        self.height_col = cfg.get("parquet_height_column", "height")
        self._files: dict[str, object] = {}          # path -> pyarrow.parquet.ParquetFile
        self._rg_starts: dict[str, list[int]] = {}   # path -> cumulative row-group starts
        self._rg_cache: dict[str, tuple[int, object]] = {}  # path -> (rg_idx, image chunk)

    def close(self) -> None:
        self._files.clear()
        self._rg_starts.clear()
        self._rg_cache.clear()

    def _open(self, path: str):
        import pyarrow.parquet as pq

        pf = self._files.get(path)
        if pf is None:
            pf = pq.ParquetFile(path)
            starts, total = [], 0
            for i in range(pf.num_row_groups):
                starts.append(total)
                total += pf.metadata.row_group(i).num_rows
            self._files[path] = pf
            self._rg_starts[path] = starts
        return pf

    def _resolve_caption_col(self, schema_names: list[str]) -> str | None:
        if self.caption_col is not None:
            return self.caption_col if self.caption_col in schema_names else None
        for cand in ("caption", "conversations"):
            if cand in schema_names:
                return cand
        return None

    def num_rows(self, path: str) -> int:
        return self._open(path).metadata.num_rows

    def enumerate_columns(self, path: str) -> dict:
        """Columnar scan of caption (+ optional dims) for every row — no image bytes.

        Returns {"caption": list[list[str]]} plus "width"/"height" (list[int|None])
        when those columns exist.
        """
        pf = self._open(path)
        names = pf.schema_arrow.names
        cols = []
        cap_col = self._resolve_caption_col(names)
        if cap_col:
            cols.append(cap_col)
        has_dims = self.width_col in names and self.height_col in names
        if has_dims:
            cols.extend([self.width_col, self.height_col])
        out: dict = {}
        n = pf.metadata.num_rows
        if not cols:
            logger.warning(
                "Parquet %s has no caption column (tried %s); captions will be empty.",
                path, self.caption_col or "caption/conversations",
            )
            out["caption"] = [[""] for _ in range(n)]
            return out
        table = pf.read(columns=cols)
        if cap_col:
            out["caption"] = [_caption_from_cell(c) for c in table.column(cap_col).to_pylist()]
        else:
            out["caption"] = [[""] for _ in range(n)]
        if has_dims:
            out["width"] = table.column(self.width_col).to_pylist()
            out["height"] = table.column(self.height_col).to_pylist()
        return out

    def read_image(self, path: str, row: int) -> io.BytesIO:
        """Image bytes for one row, via the single-row-group cache."""
        pf = self._open(path)
        starts = self._rg_starts[path]
        rg = 0
        for i, s in enumerate(starts):
            if row >= s:
                rg = i
            else:
                break
        cached = self._rg_cache.get(path)
        if cached is None or cached[0] != rg:
            if self.image_col not in pf.schema_arrow.names:
                raise ValueError(
                    f"Parquet {path} has no image column {self.image_col!r} "
                    f"(set parquet_image_column in the directory config)"
                )
            chunk = pf.read_row_group(rg, columns=[self.image_col]).column(self.image_col)
            self._rg_cache[path] = (rg, chunk)
            cached = (rg, chunk)
        cell = cached[1][row - starts[rg]].as_py()
        data = _image_bytes_from_cell(cell)
        if data is None:
            raise ValueError(f"Parquet {path} row {row}: empty/unsupported image cell")
        return io.BytesIO(data)
