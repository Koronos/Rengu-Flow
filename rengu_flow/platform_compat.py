"""Cross-platform helpers (Linux / WSL / Windows) for venv layout, processes, ports, IO.

This module is **stdlib-only at import time**. ``psutil`` is imported lazily inside the
functions that need it because it is a ``[ui]`` optional dependency, not a core training
dependency: the training core must keep importing this module without psutil installed.

Tests monkeypatch the module-level ``IS_WINDOWS`` flag (every function reads it at call
time) to exercise both platforms from a single host.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import NoReturn

# Value of subprocess.CREATE_NEW_PROCESS_GROUP on Windows; defined here so the constant is
# available (and testable) on POSIX where the attribute does not exist.
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

IS_WINDOWS = sys.platform == "win32"


def _detect_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


IS_WSL = _detect_wsl()


# ----------------------------------------------------------------------- CUDA allocator (WSL)

_CUDA_ALLOC_ENV = "PYTORCH_CUDA_ALLOC_CONF"
# WSL2 has no working "No Sysmem Fallback" switch, so near the VRAM ceiling the driver silently
# pages to host RAM (huge, erratic step times) instead of OOMing. These native-allocator knobs cut
# fragmentation so we hit that wall less. They are *defaults*: a value the user set explicitly wins.
_WSL_ALLOC_DEFAULTS = {
    "garbage_collection_threshold": "0.8",
    "max_split_size_mb": "256",
}


def configure_cuda_allocator(*, is_wsl: bool | None = None, env: dict | None = None, log: bool = True) -> str | None:
    """Apply WSL-safe CUDA caching-allocator settings. No-op off WSL.

    On WSL2/WDDM the expandable-segments allocator uses CUDA's virtual-memory APIs
    (``cuMemMap`` / ``cuMemSetAccess``). Those raise ``CUDA driver error: device not ready`` when
    cuDNN allocates convolution workspace, so any conv model (e.g. SDXL's UNet) crashes early in
    the backward pass; transformer-only models don't allocate that workspace, which is why the
    failure looks model-specific. This forces ``expandable_segments:False`` regardless of what the
    shell or ``[training.env]`` set, and fills in low-fragmentation defaults, preserving any knob
    the user set explicitly.

    MUST run before the first ``import torch`` in the process — the caching allocator parses
    ``PYTORCH_CUDA_ALLOC_CONF`` when torch is imported. ``rengu_flow/__init__`` calls it for exactly
    that reason (this module is stdlib-only at import time, so it is safe to call that early).
    Returns the resulting env value, or ``None`` when not on WSL.
    """
    env = os.environ if env is None else env
    is_wsl = IS_WSL if is_wsl is None else is_wsl
    if not is_wsl:
        return None

    pairs: dict[str, str] = {}
    for part in env.get(_CUDA_ALLOC_ENV, "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, val = part.partition(":")
        pairs[key.strip()] = val.strip()

    had_expandable_true = pairs.get("expandable_segments", "").lower() == "true"
    pairs["expandable_segments"] = "False"
    for key, val in _WSL_ALLOC_DEFAULTS.items():
        pairs.setdefault(key, val)

    env[_CUDA_ALLOC_ENV] = ",".join(f"{k}:{v}" for k, v in pairs.items())
    if log and had_expandable_true:
        print(
            "rengu_flow: WSL detected — forcing expandable_segments:False "
            "(cuMemMap is unsupported under WSL2/WDDM and crashes cuDNN conv workspace with "
            f"'CUDA driver error: device not ready'). {_CUDA_ALLOC_ENV}={env[_CUDA_ALLOC_ENV]}"
        )
    return env[_CUDA_ALLOC_ENV]


# --------------------------------------------------------------------------- venv layout


def venv_bin_dir(venv: Path) -> Path:
    """Return the executables directory inside *venv*: ``Scripts`` (Windows) or ``bin``."""
    return Path(venv) / ("Scripts" if IS_WINDOWS else "bin")


def venv_exe(venv: Path, name: str) -> Path:
    """Path to executable *name* inside *venv*, with ``.exe`` appended on Windows."""
    exe = f"{name}.exe" if IS_WINDOWS else name
    return venv_bin_dir(venv) / exe


# ---------------------------------------------------------------------- process management


def popen_kwargs_new_group() -> dict:
    """``subprocess.Popen`` kwargs to start the child in its own group/session.

    POSIX: ``start_new_session=True`` so the child leads a new session and signals sent to
    the parent do not propagate to it. Windows: ``CREATE_NEW_PROCESS_GROUP`` so the child is
    isolated from the parent's Ctrl+C and can later be signalled independently.
    """
    if IS_WINDOWS:
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def pid_alive(pid: int | None) -> bool:
    """Best-effort check whether process *pid* is currently running."""
    if pid is None:
        return False
    if IS_WINDOWS:
        try:
            import psutil
        except ImportError:
            return True  # cannot determine without psutil; assume alive
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_tree(pid: int | None, *, grace: float = 5.0) -> None:
    """Terminate *pid* and all descendants: graceful first, then forced.

    Uses ``psutil`` (children-first ``terminate()`` then ``kill()``) when available; falls
    back to a POSIX process-group / ``os.kill`` path when psutil is missing.
    """
    if pid is None:
        return
    try:
        import psutil
    except ImportError:
        _terminate_no_psutil(pid)
        return
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True)
    procs.append(parent)
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _gone, alive = psutil.wait_procs(procs, timeout=grace)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def _terminate_no_psutil(pid: int) -> None:
    import signal

    if IS_WINDOWS:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def reexec(path: Path, argv: list[str]) -> NoReturn:
    """Hand off to *path* + *argv*: ``execv`` on POSIX, run-then-``sys.exit`` on Windows.

    On Windows ``os.execv`` orphans the console and mangles exit-code propagation, so we run
    the target as a child and exit with its return code instead.
    """
    path = Path(path)
    if IS_WINDOWS:
        completed = subprocess.run([str(path), *argv])
        sys.exit(completed.returncode)
    os.execv(str(path), [str(path), *argv])


# ------------------------------------------------------------------------------ networking


def find_free_port(start: int = 29500, count: int = 101, host: str = "127.0.0.1") -> int:
    """First bindable TCP port in ``[start, start + count)``; falls back to *start*.

    Cross-platform replacement for parsing ``ss``/``netstat`` output: probe by binding.

    Note: ``SO_REUSEADDR`` is set only on POSIX. On Windows it has different semantics and
    would let us bind to a port that is already in active use, defeating the probe.
    """
    for port in range(start, start + count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if not IS_WINDOWS:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    return start


def free_port_owned_by(port: int, patterns: tuple[str, ...]) -> None:
    """Free *port* if held by one of *our* processes; raise otherwise.

    Replaces the Linux-only ``ss`` + ``/proc/<pid>/cmdline`` + ``kill`` flow with psutil.
    If a listener on *port* has a command line matching any of *patterns*, terminate its
    process tree. If held by an unrelated process, raise ``SystemExit``. No-op when the port
    is free or psutil is unavailable.
    """
    try:
        import psutil
    except ImportError:
        return
    try:
        conns = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError):
        return
    for conn in conns:
        if not conn.laddr or conn.laddr.port != port:
            continue
        if conn.status != psutil.CONN_LISTEN:
            continue
        pid = conn.pid
        if pid is None:
            continue
        try:
            cmd = " ".join(psutil.Process(pid).cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cmd = ""
        if cmd and any(p in cmd for p in patterns):
            print(f"==> Stopping process on port {port} (PID {pid})...")
            terminate_process_tree(pid)
            return
        raise SystemExit(
            f"rengu: port {port} in use by PID {pid}: {cmd.strip() or '<unknown process>'}"
        )


# ----------------------------------------------------------------------------------- misc


def open_browser(url: str) -> None:
    """Open *url* in the default browser cross-platform; print a fallback on failure."""
    try:
        if webbrowser.open(url):
            return
    except webbrowser.Error:
        pass
    print(f"Open in browser: {url}")


def http_health_ok(url: str, headers: dict | None = None, timeout: float = 2.0) -> bool:
    """Return True if a GET to *url* returns HTTP status < 400. Replaces a ``curl`` probe."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local URL)
            return resp.status < 400
    except urllib.error.HTTPError as exc:
        return exc.code < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False
