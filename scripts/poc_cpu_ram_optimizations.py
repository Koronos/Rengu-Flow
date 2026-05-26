#!/usr/bin/env python3
"""Proof-of-concept benchmarks for CPU/RAM/disk optimization ideas.

Run from repo root (no GPU required for most tests):
  python scripts/poc_cpu_ram_optimizations.py
  python scripts/poc_cpu_ram_optimizations.py --json tmp/poc_results.json

Uses synthetic SDXL-like tensors (4x64x64 latents, TE-like 1x77x2048) in a temp dir.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sqlite3
import statistics
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = REPO / "tests/fixtures/smoke_cc0/manifest.json"

# SDXL-ish cached training tensors
LATENT_SHAPE = (4, 64, 64)
TE_SHAPE = (1, 77, 2048)
N_ITEMS = 512
BATCH_SIZE = 4
WARMUP = 20
ITERS = 200


@dataclass
class PocResult:
    name: str
    verdict: str  # adopt | opt_in | skip | defer
    default_on: bool
    notes: str
    metrics: dict = field(default_factory=dict)


def _timer(fn, *, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def _make_latent() -> torch.Tensor:
    return torch.randn(LATENT_SHAPE, dtype=torch.float32)


def _make_te() -> torch.Tensor:
    return torch.randn(TE_SHAPE, dtype=torch.float32)


# --- 1. Storage format: pickle vs safetensors vs raw mmap bf16 ---


def poc_storage_formats(tmp: Path) -> PocResult:
    import safetensors.torch

    items = [_make_latent() for _ in range(N_ITEMS)]
    indices = [random.randint(0, N_ITEMS - 1) for _ in range(ITERS)]

    # Pickle blobs (like current Cache)
    pickle_dir = tmp / "pickle"
    pickle_dir.mkdir()
    pickle_meta: list[tuple[int, int]] = []
    off = 0
    bin_path = pickle_dir / "shard_0.bin"
    with bin_path.open("wb") as f:
        for t in items:
            buf = io.BytesIO()
            torch.save({"latents": t}, buf)
            b = buf.getbuffer()
            f.write(b)
            pickle_meta.append((off, len(b)))
            off += len(b)

    def read_pickle():
        i = random.choice(indices)
        o, sz = pickle_meta[i]
        with bin_path.open("rb") as f:
            f.seek(o)
            return torch.load(io.BytesIO(f.read(sz)), map_location="cpu", weights_only=False)

    # Safetensors per item (one file) — worst case many files; also one combined
    st_dir = tmp / "safetensors"
    st_dir.mkdir()
    for i, t in enumerate(items):
        safetensors.torch.save_file({"latents": t}, st_dir / f"{i}.safetensors")

    def read_safetensors():
        i = random.choice(indices)
        return safetensors.torch.load_file(st_dir / f"{i}.safetensors")["latents"]

    # Raw bf16 mmap single file
    raw_path = tmp / "latents_bf16.bin"
    flat = torch.stack(items).to(torch.bfloat16)
    raw_path.write_bytes(flat.view(torch.uint16).numpy().tobytes())
    shape = flat.shape

    def read_mmap_bf16():
        i = random.choice(indices)
        arr = np.memmap(raw_path, dtype=np.uint16, mode="r", shape=shape)
        t = torch.from_numpy(np.asarray(arr[i]).view(np.uint16)).view(torch.bfloat16).float()
        return t

    t_pickle = _timer(read_pickle)
    t_st = _timer(read_safetensors)
    t_mmap = _timer(read_mmap_bf16)

    pickle_bytes = bin_path.stat().st_size
    st_bytes = sum(f.stat().st_size for f in st_dir.glob("*.safetensors"))
    raw_bytes = raw_path.stat().st_size

    best = min(t_pickle, t_st, t_mmap)
    speedup_mmap = (t_pickle / t_mmap - 1) * 100 if t_mmap else 0
    disk_ratio = pickle_bytes / raw_bytes if raw_bytes else 1

    verdict = "adopt" if speedup_mmap >= 15 and disk_ratio >= 1.4 else "opt_in"
    default_on = verdict == "adopt" and speedup_mmap >= 25
    return PocResult(
        name="storage_mmap_bf16_vs_pickle",
        verdict=verdict,
        default_on=default_on,
        notes="mmap+bf16 raw layout vs torch.save shards",
        metrics={
            "sec_per_read_pickle": round(t_pickle, 6),
            "sec_per_read_safetensors_file": round(t_st, 6),
            "sec_per_read_mmap_bf16": round(t_mmap, 6),
            "speedup_mmap_vs_pickle_pct": round(speedup_mmap, 1),
            "disk_pickle_mb": round(pickle_bytes / 1e6, 2),
            "disk_safetensors_mb": round(st_bytes / 1e6, 2),
            "disk_raw_bf16_mb": round(raw_bytes / 1e6, 2),
            "disk_savings_bf16_pct": round((1 - raw_bytes / pickle_bytes) * 100, 1),
        },
    )


# --- 2. Batch read vs sequential single reads ---


def poc_batch_read(tmp: Path) -> PocResult:
    items = [_make_latent() for _ in range(N_ITEMS)]
    meta = []
    off = 0
    path = tmp / "batch_shard.bin"
    with path.open("wb") as f:
        for t in items:
            buf = io.BytesIO()
            torch.save({"latents": t}, buf)
            b = buf.getbuffer()
            f.write(b)
            meta.append((off, len(b)))
            off += len(b)

    def read_one(idx: int):
        o, sz = meta[idx]
        with path.open("rb") as f:
            f.seek(o)
            return torch.load(io.BytesIO(f.read(sz)), map_location="cpu", weights_only=False)

    def sequential_batch():
        start = random.randint(0, N_ITEMS - BATCH_SIZE)
        for j in range(BATCH_SIZE):
            read_one(start + j)

    def batched_single_open():
        start = random.randint(0, N_ITEMS - BATCH_SIZE)
        with path.open("rb") as f:
            for j in range(BATCH_SIZE):
                o, sz = meta[start + j]
                f.seek(o)
                torch.load(io.BytesIO(f.read(sz)), map_location="cpu", weights_only=False)

    t_seq = _timer(sequential_batch, iters=ITERS // 2)
    t_batch = _timer(batched_single_open, iters=ITERS // 2)
    gain = (t_seq / t_batch - 1) * 100 if t_batch else 0
    verdict = "adopt" if gain >= 8 else "opt_in" if gain >= 3 else "skip"
    return PocResult(
        name="batch_read_single_file_handle",
        verdict=verdict,
        default_on=verdict == "adopt",
        notes="One open()+N seeks vs N open() per micro-batch",
        metrics={
            "sec_microbatch_sequential_opens": round(t_seq, 6),
            "sec_microbatch_single_open": round(t_batch, 6),
            "speedup_pct": round(gain, 1),
        },
    )


# --- 3. TE dedup by caption hash ---


def poc_te_dedup() -> PocResult:
    captions: list[str] = []
    if SMOKE_MANIFEST.is_file():
        data = json.loads(SMOKE_MANIFEST.read_text())
        captions = [e["caption"] for e in data]
    # Simulate tag-heavy dataset: 1000 images, 50 unique captions
    synthetic = [f"tag{i % 50}, character, style" for i in range(1000)]
    for pool, label in ((captions, "smoke_cc0"), (synthetic, "synthetic_1k")):
        pass

    def dedup_stats(pool: list[str]):
        keys = [hashlib.md5(c.encode()).hexdigest() for c in pool]
        unique = len(set(keys))
        return {
            "n": len(pool),
            "unique": unique,
            "dup_ratio": round(1 - unique / len(pool), 3),
            "te_entries_saved_pct": round((1 - unique / len(pool)) * 100, 1),
        }

    smoke = dedup_stats(captions) if captions else {"n": 0, "unique": 0, "dup_ratio": 0}
    syn = dedup_stats(synthetic)
    worth = syn["dup_ratio"] >= 0.3 or syn["te_entries_saved_pct"] >= 30
    verdict = "opt_in" if worth else "skip"
    return PocResult(
        name="te_dedup_caption_hash",
        verdict=verdict,
        default_on=False,
        notes="smoke_cc0 all unique; synthetic tag sets show big TE savings",
        metrics={"smoke_cc0": smoke, "synthetic_tag_heavy": syn},
    )


# --- 5. Page cache warm (second pass faster) ---


def poc_page_cache_warm(tmp: Path) -> PocResult:
    path = tmp / "warm_big.bin"
    # ~64MB file
    data = os.urandom(64 * 1024 * 1024)
    path.write_bytes(data)

    def read_chunks():
        with path.open("rb") as f:
            while f.read(1024 * 1024):
                pass

    # drop cache best-effort (linux)
    try:
        os.system(f"sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1")
    except Exception:
        pass

    t_cold = time.perf_counter()
    read_chunks()
    cold = time.perf_counter() - t_cold
    t_warm = time.perf_counter()
    read_chunks()
    warm = time.perf_counter() - t_warm
    gain = (cold / warm - 1) * 100 if warm > 0 else 0
    verdict = "opt_in" if gain >= 50 else "skip"
    return PocResult(
        name="page_cache_warm_second_read",
        verdict=verdict,
        default_on=False,
        notes="Optional warm pass; cold drop_caches may need sudo",
        metrics={
            "read_64mb_cold_sec": round(cold, 4),
            "read_64mb_warm_sec": round(warm, 4),
            "warm_speedup_pct": round(gain, 1),
        },
    )


# --- 6. NumPy index vs list-of-tuples iteration ---


def poc_numpy_iteration() -> PocResult:
    n = 50_000
    latent_idx = list(range(n))
    cap_idx = [i % 5 for i in range(n)]
    order_list = [(i, latent_idx[i], f"caption_{cap_idx[i]}", cap_idx[i]) for i in range(n)]
    lat_arr = np.array(latent_idx, dtype=np.int32)
    cap_arr = np.array(cap_idx, dtype=np.int32)

    def walk_list():
        s = 0
        for i in range(0, n, BATCH_SIZE):
            for j in range(BATCH_SIZE):
                k = i + j
                entry = order_list[k]
                s += entry[1] + entry[3]
        return s

    def walk_numpy():
        s = 0
        for i in range(0, n, BATCH_SIZE):
            sl = slice(i, i + BATCH_SIZE)
            s += int(lat_arr[sl].sum()) + int(cap_arr[sl].sum())
        return s

    t_list = _timer(walk_list, warmup=5, iters=30)
    t_np = _timer(walk_numpy, warmup=5, iters=30)
    gain = (t_list / t_np - 1) * 100 if t_np else 0
    verdict = "adopt" if gain >= 10 else "opt_in" if gain >= 5 else "skip"
    return PocResult(
        name="numpy_iteration_order_indices",
        verdict=verdict,
        default_on=verdict == "adopt",
        notes="Internal epoch index arrays vs list of tuples",
        metrics={
            "sec_per_epoch_pass_list": round(t_list, 6),
            "sec_per_epoch_pass_numpy": round(t_np, 6),
            "speedup_pct": round(gain, 1),
        },
    )


# --- 7. CUDA H2D pin_memory + non_blocking ---


def poc_cuda_h2d() -> PocResult:
    if not torch.cuda.is_available():
        return PocResult(
            name="cuda_h2d_non_blocking",
            verdict="defer",
            default_on=False,
            notes="No CUDA in this environment",
            metrics={},
        )
    device = torch.device("cuda")
    t_cpu = _make_latent().unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)

    def blocking():
        x = t_cpu.clone()
        x.to(device)

    def pinned_nonblocking():
        x = t_cpu.clone().pin_memory()
        x.to(device, non_blocking=True)
        torch.cuda.synchronize()

    t_block = _timer(blocking, warmup=10, iters=100)
    t_nb = _timer(pinned_nonblocking, warmup=10, iters=100)
    gain = (t_block / t_nb - 1) * 100 if t_nb else 0
    verdict = "opt_in" if gain >= 5 else "skip"
    return PocResult(
        name="cuda_h2d_non_blocking",
        verdict=verdict,
        default_on=False,
        notes="Micro-benchmark only; real win needs overlap with GPU compute",
        metrics={
            "sec_transfer_blocking": round(t_block, 6),
            "sec_transfer_pinned_nb": round(t_nb, 6),
            "speedup_pct": round(gain, 1),
        },
    )


# --- 8. bf16 on disk size + read + numeric error ---


def poc_bf16_disk(tmp: Path) -> PocResult:
    t = _make_latent()
    buf32 = io.BytesIO()
    torch.save({"latents": t}, buf32)
    sz32 = len(buf32.getvalue())
    flat16 = t.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
    sz16 = len(flat16)
    path16 = tmp / "one_bf16.raw"
    path16.write_bytes(flat16)
    path32 = tmp / "one_fp32.raw"
    t.numpy().tofile(path32)

    def read32():
        return torch.from_numpy(np.fromfile(path32, dtype=np.float32).reshape(LATENT_SHAPE))

    def read16():
        u = np.fromfile(path16, dtype=np.uint16)
        return torch.from_numpy(u).view(torch.bfloat16).reshape(LATENT_SHAPE).float()

    t_r32 = _timer(read32, iters=500)
    t_r16 = _timer(read16, iters=500)
    rec = read16()
    max_err = (rec - t).abs().max().item()
    disk_ratio = sz32 / sz16 if sz16 else 1
    verdict = "opt_in"
    default_on = disk_ratio >= 1.8 and max_err < 0.01 and t_r16 <= t_r32 * 1.1
    if default_on:
        verdict = "adopt"
    return PocResult(
        name="cache_storage_bf16",
        verdict=verdict,
        default_on=default_on,
        notes="Default on only if ~2x disk savings, neutral read, tiny numeric drift",
        metrics={
            "pickle_fp32_bytes": sz32,
            "raw_bf16_bytes": sz16,
            "disk_ratio_fp32_to_bf16": round(disk_ratio, 2),
            "max_abs_error_vs_fp32": max_err,
            "sec_read_fp32": round(t_r32, 6),
            "sec_read_bf16": round(t_r16, 6),
        },
    )


# --- 9. zstd on TE tensor ---


def poc_te_zstd(tmp: Path) -> PocResult:
    try:
        import zstandard as zstd
    except ImportError:
        te = _make_te()
        raw = te.numpy().tobytes()
        comp = zlib.compress(raw, level=3)
        ratio = len(raw) / len(comp)
        t_raw = _timer(lambda: te.numpy().tobytes(), iters=100)
        t_de = _timer(lambda: np.frombuffer(zlib.decompress(comp), dtype=np.float32).reshape(TE_SHAPE), iters=100)
        verdict = "opt_in" if ratio >= 1.5 else "skip"
        return PocResult(
            name="te_compression_zlib",
            verdict=verdict,
            default_on=False,
            notes="zstandard not installed; used zlib",
            metrics={
                "compress_ratio": round(ratio, 2),
                "sec_read_raw": round(t_raw, 6),
                "sec_decompress": round(t_de, 6),
            },
        )

    te = _make_te()
    raw = te.numpy().tobytes()
    cctx = zstd.ZstdCompressor(level=3)
    comp = cctx.compress(raw)
    dctx = zstd.ZstdDecompressor()
    ratio = len(raw) / len(comp)
    sparse = te.clone()
    sparse[sparse.abs() < 0.5] = 0
    raw_s = sparse.numpy().tobytes()
    comp_s = cctx.compress(raw_s)

    t_de = _timer(lambda: np.frombuffer(dctx.decompress(comp), dtype=np.float32).reshape(TE_SHAPE), iters=100)
    disk_save_pct = (1 - len(comp) / len(raw)) * 100
    verdict = "skip"
    default_on = False
    if disk_save_pct >= 40 and t_de < 0.002:
        verdict = "opt_in"
    if disk_save_pct >= 60 and ratio >= 2:
        default_on = disk_save_pct >= 50  # only if huge TE savings
    return PocResult(
        name="te_compression_zstd",
        verdict=verdict,
        default_on=default_on,
        notes="Default only if multi-GB TE cache; zlib/zstd on 1 tensor",
        metrics={
            "dense_compress_ratio": round(len(raw) / len(comp), 2),
            "sparse_compress_ratio": round(len(raw_s) / len(comp_s), 2),
            "disk_save_pct_dense": round(disk_save_pct, 1),
            "sec_decompress": round(t_de, 6),
            "te_shape": list(TE_SHAPE),
        },
    )


# --- 4 & 10: documented skip ---


def poc_skip(name: str, notes: str) -> PocResult:
    return PocResult(name=name, verdict="skip", default_on=False, notes=notes, metrics={})


def run_all(tmp: Path) -> list[PocResult]:
    random.seed(42)
    results = [
        poc_storage_formats(tmp),
        poc_batch_read(tmp),
        poc_te_dedup(),
        poc_skip("cache_dir_separate_volume", "Config/docs only; no code path"),
        poc_page_cache_warm(tmp),
        poc_numpy_iteration(),
        poc_cuda_h2d(),
        poc_bf16_disk(tmp),
        poc_te_zstd(tmp),
        poc_skip("incremental_cache_trust", "Already implemented: --trust_cache + fingerprint"),
    ]
    return results


def print_report(results: list[PocResult]) -> None:
    print("\n=== POC CPU/RAM optimizations ===\n")
    for r in results:
        flag = "DEFAULT ON" if r.default_on else "default off"
        print(f"[{r.verdict.upper():7}] {r.name} ({flag})")
        print(f"         {r.notes}")
        for k, v in r.metrics.items():
            print(f"         {k}: {v}")
        print()
    adopt_defaults = [r.name for r in results if r.default_on]
    print("Recommended defaults to enable:", adopt_defaults or "(none from POC alone)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="Write JSON results")
    args = parser.parse_args()
    import shutil

    tmp = Path(os.environ.get("POC_TMP", "/tmp/renga_poc"))
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    results = run_all(tmp)
    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "name": r.name,
                "verdict": r.verdict,
                "default_on": r.default_on,
                "notes": r.notes,
                "metrics": r.metrics,
            }
            for r in results
        ]
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.json}")
    # cleanup large files
    for p in tmp.glob("*"):
        try:
            if p.is_file() and p.stat().st_size > 1_000_000:
                p.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
