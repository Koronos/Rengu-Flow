"""Disk cache with fingerprint and sharding (ported from diffusion-pipe utils/cache.py)."""

import io
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import torch


class Cache:
    """Persistent cache storing items in sharded binary files with SQLite metadata.

    Fingerprint is stored; if it changes, cache is cleared. Items are torch.save'd
    into shard files (default 10 GB per shard).
    """

    def __init__(self, path: str | Path, fingerprint: str, shard_size_gb: float = 10.0) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.metadata_db = self.path / "metadata.db"
        self.shard_size_gb = shard_size_gb
        os.makedirs(self.path, exist_ok=True)
        self.init()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            raise TypeError("Cache index must be int")
        shard_id, shard_index = self.items[idx]
        offset, size = self.shard_metadata[shard_id][shard_index]
        if shard_id not in self.open_files:
            self.open_files[shard_id] = open(  # noqa: SIM115
                self.path / f"shard_{shard_id}.bin", "rb"
            )
        f = self.open_files[shard_id]
        f.seek(offset)
        byte_string = f.read(size)
        buffer = io.BytesIO(byte_string)
        return torch.load(buffer, map_location="cpu", weights_only=False)

    def init(self) -> None:
        print("[CACHE] Initializing")
        self.con = sqlite3.connect(self.metadata_db)

        self.con.execute("CREATE TABLE IF NOT EXISTS fingerprint(value)")
        existing_fingerprint = self.con.execute("SELECT value FROM fingerprint").fetchone()
        if existing_fingerprint is not None:
            existing_fingerprint = existing_fingerprint[0]
            print(f"[CACHE] Existing cache has fingerprint {existing_fingerprint}")
            if self.fingerprint != existing_fingerprint:
                print("[CACHE] Fingerprint changed, deleting existing cache files")
                self.clear()
                return
        else:
            print(f"[CACHE] Storing new fingerprint: {self.fingerprint}")
            self.con.execute("INSERT INTO fingerprint VALUES(?)", (self.fingerprint,))

        self.con.execute("CREATE TABLE IF NOT EXISTS items(shard, shard_index)")
        self.items = self.con.execute("SELECT shard, shard_index FROM items").fetchall() or []
        max_existing_shard = -1
        for shard, _ in self.items:
            max_existing_shard = max(max_existing_shard, shard)
        self.shard = max_existing_shard + 1
        self.shard_file = None
        print(f"[CACHE] Existing cache length: {len(self)}")

        self.shard_metadata = defaultdict(list)
        for (table_name,) in self.con.execute(
            "SELECT name FROM sqlite_master"
        ).fetchall():
            if table_name.startswith("shard_"):
                shard_id = int(table_name.split("_")[-1])
                for entry in self.con.execute(
                    f"SELECT offset, size FROM {table_name}"
                ).fetchall():
                    self.shard_metadata[shard_id].append(entry)
        self.open_files = {}
        self.con.commit()

    def clear(self) -> None:
        """Delete all cache files from disk and re-run init()."""
        self.con.close()
        os.remove(self.metadata_db)
        for bin_path in self.path.glob("*.bin"):
            os.remove(bin_path)
        self.init()

    def create_new_shard(self) -> None:
        self.shard_file = open(  # noqa: SIM115
            self.path / f"shard_{self.shard}.bin", "wb"
        )
        self.shard_table = f"shard_{self.shard}"
        print(f"[CACHE] Creating new shard: {self.shard_table}")
        self.con.execute(f"CREATE TABLE {self.shard_table}(offset, size)")
        self.shard_index = 0
        self.offset = 0

    def finalize_current_shard(self) -> None:
        if self.shard_file is None:
            return
        self.shard_file.close()
        self.shard_file = None
        self.shard += 1
        self.con.commit()

    def add(self, item) -> None:
        if self.shard_file is None:
            self.create_new_shard()
        buffer = io.BytesIO()
        torch.save(item, buffer)
        bytes_view = buffer.getbuffer()
        self.shard_file.write(bytes_view)

        meta = (self.shard, self.shard_index)
        self.items.append(meta)
        self.con.execute("INSERT INTO items VALUES(?, ?)", meta)
        self.shard_index += 1

        size = len(bytes_view)
        entry = (self.offset, size)
        self.shard_metadata[self.shard].append(entry)
        self.con.execute(
            f"INSERT INTO {self.shard_table} VALUES (?, ?)", entry
        )
        self.offset += size

        current_size_gb = self.shard_file.tell() / 1_000_000_000
        if current_size_gb >= self.shard_size_gb:
            self.finalize_current_shard()
