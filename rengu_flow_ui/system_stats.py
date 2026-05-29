"""Host CPU/RAM/GPU metrics for the control-plane UI (AI-toolkit style)."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

_GB = 1024**3


def _gb(n_bytes: float | int | None) -> float | None:
    if n_bytes is None:
        return None
    return round(float(n_bytes) / _GB, 2)


def _read_linux_thermal_zones() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base = Path("/sys/class/thermal")
    if not base.is_dir():
        return out
    for zone in sorted(base.glob("thermal_zone*")):
        typ = zone / "type"
        temp = zone / "temp"
        if not temp.is_file():
            continue
        try:
            millideg = int(temp.read_text().strip())
        except (OSError, ValueError):
            continue
        label = typ.read_text().strip() if typ.is_file() else zone.name
        out.append(
            {
                "label": label,
                "current_c": round(millideg / 1000.0, 1),
            }
        )
    return out


def _cpu_temperatures() -> list[dict[str, Any]]:
    temps: list[dict[str, Any]] = []
    try:
        import psutil

        if hasattr(psutil, "sensors_temperatures"):
            raw = psutil.sensors_temperatures() or {}
            for chip, entries in raw.items():
                for ent in entries:
                    if ent.current is None:
                        continue
                    label = ent.label or chip
                    temps.append(
                        {
                            "label": f"{chip}:{label}" if ent.label else chip,
                            "current_c": round(float(ent.current), 1),
                            "high_c": round(float(ent.high), 1) if ent.high else None,
                        }
                    )
    except Exception:
        pass
    if not temps:
        for z in _read_linux_thermal_zones():
            temps.append({"label": z["label"], "current_c": z["current_c"], "high_c": None})
    return temps


def _primary_cpu_temp_c(temps: list[dict[str, Any]]) -> float | None:
    if not temps:
        return None
    preferred = ("coretemp", "k10temp", "zenpower", "cpu", "x86_pkg_temp")
    for key in preferred:
        for t in temps:
            if key in t.get("label", "").lower():
                return t.get("current_c")
    return max((t["current_c"] for t in temps if t.get("current_c") is not None), default=None)


def _collect_cpu(*, sample: bool) -> dict[str, Any]:
    cpu: dict[str, Any] = {"available": False}
    try:
        import psutil

        cpu["available"] = True
        if sample:
            per_core = psutil.cpu_percent(interval=0.15, percpu=True)
            cpu["per_core"] = per_core
            cpu["percent"] = (
                round(sum(per_core) / len(per_core), 1) if per_core else psutil.cpu_percent(interval=0.15)
            )
        else:
            cpu["percent"] = psutil.cpu_percent(interval=None)
            cpu["per_core"] = psutil.cpu_percent(interval=None, percpu=True)
        cpu["logical_count"] = psutil.cpu_count(logical=True)
        cpu["physical_count"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq:
            cpu["freq_mhz"] = {
                "current": round(freq.current, 0) if freq.current else None,
                "min": round(freq.min, 0) if freq.min else None,
                "max": round(freq.max, 0) if freq.max else None,
            }
        temps = _cpu_temperatures()
        cpu["temperatures"] = temps
        cpu["temp_c"] = _primary_cpu_temp_c(temps)
    except ImportError:
        cpu["error"] = "psutil not installed"
    except Exception as e:
        cpu["error"] = str(e)
    return cpu


def _collect_ram() -> dict[str, Any]:
    ram: dict[str, Any] = {"available": False}
    try:
        import psutil

        v = psutil.virtual_memory()
        s = psutil.swap_memory()
        ram["available"] = True
        ram["total_bytes"] = v.total
        ram["used_bytes"] = v.used
        ram["available_bytes"] = v.available
        ram["percent"] = v.percent
        ram["total_gb"] = _gb(v.total)
        ram["used_gb"] = _gb(v.used)
        ram["available_gb"] = _gb(v.available)
        ram["swap"] = {
            "total_gb": _gb(s.total),
            "used_gb": _gb(s.used),
            "percent": s.percent,
        }
    except ImportError:
        ram["error"] = "psutil not installed"
    except Exception as e:
        ram["error"] = str(e)
    return ram


def _parse_nvidia_smi_line(line: str) -> dict[str, Any] | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None
    try:
        idx = int(parts[0])
        name = parts[1]
        temp = float(parts[2]) if parts[2] not in ("[N/A]", "N/A", "") else None
        util = float(parts[3]) if parts[3] not in ("[N/A]", "N/A", "") else None
        mem_used_mib = float(parts[4])
        mem_total_mib = float(parts[5])
        mem_util = float(parts[6]) if parts[6] not in ("[N/A]", "N/A", "") else None
    except ValueError:
        return None
    mem_used = mem_used_mib * 1024**2
    mem_total = mem_total_mib * 1024**2
    return {
        "index": idx,
        "name": name,
        "temp_c": temp,
        "util_percent": util,
        "memory_util_percent": mem_util,
        "vram_used_bytes": int(mem_used),
        "vram_total_bytes": int(mem_total),
        "vram_used_gb": _gb(mem_used),
        "vram_total_gb": _gb(mem_total),
        "vram_percent": round(100.0 * mem_used / mem_total, 1) if mem_total else None,
    }


def _collect_gpus() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False, "devices": [], "backend": None}
    smi = shutil.which("nvidia-smi")
    if not smi:
        out["error"] = "nvidia-smi not found"
        return out
    query = (
        "index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,"
        "utilization.memory,power.draw,fan.speed,clocks.sm"
    )
    try:
        proc = subprocess.run(
            [smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        out["error"] = str(e)
        return out
    if proc.returncode != 0:
        out["error"] = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()[:500]
        return out
    out["available"] = True
    out["backend"] = "nvidia-smi"
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        base = _parse_nvidia_smi_line(",".join(parts[:7]))
        if not base:
            continue
        if len(parts) > 7:
            pwr = parts[7]
            if pwr not in ("[N/A]", "N/A", ""):
                try:
                    base["power_w"] = float(pwr)
                except ValueError:
                    pass
        if len(parts) > 8:
            fan = parts[8]
            if fan not in ("[N/A]", "N/A", ""):
                try:
                    base["fan_percent"] = float(fan)
                except ValueError:
                    pass
        if len(parts) > 9:
            clk = parts[9]
            if clk not in ("[N/A]", "N/A", ""):
                try:
                    base["clock_sm_mhz"] = float(clk)
                except ValueError:
                    pass
        out["devices"].append(base)
    return out


def _build_summary(cpu: dict, ram: dict, gpus: dict) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if cpu.get("available"):
        summary["cpu_percent"] = cpu.get("percent")
        summary["cpu_temp_c"] = cpu.get("temp_c")
    if ram.get("available"):
        summary["ram_percent"] = ram.get("percent")
        summary["ram_used_gb"] = ram.get("used_gb")
        summary["ram_total_gb"] = ram.get("total_gb")
    summary["gpus"] = []
    for d in gpus.get("devices") or []:
        summary["gpus"].append(
            {
                "index": d["index"],
                "util_percent": d.get("util_percent"),
                "vram_used_gb": d.get("vram_used_gb"),
                "vram_total_gb": d.get("vram_total_gb"),
                "vram_percent": d.get("vram_percent"),
                "temp_c": d.get("temp_c"),
            }
        )
    return summary


def collect_system_stats(*, sample_cpu: bool = True) -> dict[str, Any]:
    """Return compact summary + detailed host metrics."""
    warnings: list[str] = []
    cpu = _collect_cpu(sample=sample_cpu)
    ram = _collect_ram()
    gpus = _collect_gpus()
    if cpu.get("error"):
        warnings.append(cpu["error"])
    if ram.get("error"):
        warnings.append(ram["error"])
    if gpus.get("error") and not gpus.get("devices"):
        warnings.append(gpus["error"])

    return {
        "ok": True,
        "ts": time.time(),
        "summary": _build_summary(cpu, ram, gpus),
        "detail": {
            "cpu": cpu,
            "ram": ram,
            "gpus": gpus,
            "warnings": warnings,
        },
    }
