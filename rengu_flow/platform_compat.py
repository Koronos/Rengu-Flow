"""Per-OS behavior as a single strategy object (Linux / WSL / Windows).

Everything that differs by platform — venv layout, process groups, the training engine default,
caching policy, filesystem quirks — lives on a :class:`Platform` subclass instead of being a
``sys.platform == 'win32'`` check sprinkled across the codebase. One instance is resolved at import
into :data:`PLATFORM`; call sites dispatch on it (``PLATFORM.default_engine``,
``PLATFORM.metadata_keep_in_memory``, …). Adding or tweaking a platform is then a single class, not
a hunt for scattered conditionals.

This module is **stdlib-only at import time**. ``psutil`` is imported lazily inside the methods that
need it because it is a ``[ui]`` optional dependency, not a core training dependency.

The module-level functions (``venv_bin_dir``, ``popen_kwargs_new_group``, …) are thin delegations to
``PLATFORM`` kept for backward compatibility. Tests swap behavior by monkeypatching ``PLATFORM``
(e.g. ``monkeypatch.setattr(pc, "PLATFORM", pc.WindowsPlatform())``).
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


def _detect_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


# ============================================================================ Platform strategy


class Platform:
    """Default (POSIX) behavior. Subclasses override only what genuinely differs."""

    name = "posix"
    is_windows = False
    is_wsl = False

    # --- training / caching policy -------------------------------------------------------------
    #: Engine backend used when the user/config leaves it unset.
    default_engine = "deepspeed"
    #: DeepSpeed pipeline parallel (multi-GPU) is available. Windows has no NCCL.
    supports_multi_gpu = True
    #: torch.multiprocessing "file_system" sharing strategy is supported (POSIX shared memory).
    torch_file_system_sharing = True
    #: ``datasets.load_from_disk(keep_in_memory=...)`` for the small metadata caches. Windows keeps
    #: them in RAM (no mmap) because it cannot overwrite an mmap'd Arrow file on a later run.
    metadata_keep_in_memory = False

    def cache_worker_count(self, requested: int | None, *, default: int) -> int:
        """Worker count for the cache map/pool. ``default`` when unset, else the requested value."""
        return default if requested is None else max(1, int(requested))

    # --- filesystem ----------------------------------------------------------------------------
    #: Creating symlinks needs no special privilege.
    supports_symlinks = True

    def config_path(self, p: str | os.PathLike) -> str:
        """A filesystem path as a portable, TOML-safe string. POSIX paths are already safe;
        Windows uses forward slashes so a path is a valid TOML basic string and a config written on
        one OS loads on the other."""
        return str(p)

    # --- venv layout ---------------------------------------------------------------------------
    def venv_bin_dir(self, venv: Path) -> Path:
        return Path(venv) / "bin"

    def venv_exe(self, venv: Path, name: str) -> Path:
        return self.venv_bin_dir(venv) / name

    # --- process management --------------------------------------------------------------------
    def popen_new_group_kwargs(self) -> dict:
        """``subprocess.Popen`` kwargs to start the child in its own session, so signals sent to
        the parent do not propagate to it."""
        return {"start_new_session": True}

    def pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def kill_pid(self, pid: int) -> None:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def reexec(self, path: Path, argv: list[str]) -> NoReturn:
        os.execv(str(path), [str(path), *argv])

    # --- networking ----------------------------------------------------------------------------
    #: SO_REUSEADDR is safe on POSIX (lets us probe a port without TIME_WAIT noise). On Windows it
    #: lets bind() succeed on a port already in active use, defeating the free-port probe.
    free_port_reuseaddr = True


class WslPlatform(Platform):
    """Linux under WSL2. Same as POSIX except the WSL-specific CUDA allocator workaround applies."""

    name = "wsl"
    is_wsl = True


class WindowsPlatform(Platform):
    name = "windows"
    is_windows = True

    default_engine = "accelerate"
    supports_multi_gpu = False
    torch_file_system_sharing = False
    cache_uses_worker_process = False
    metadata_keep_in_memory = True
    supports_symlinks = False  # needs Developer Mode / admin
    free_port_reuseaddr = False

    def cache_worker_count(self, requested: int | None, *, default: int) -> int:
        return 1  # in-process only: no fork, and a spawned pool can't share the queue handoff

    def config_path(self, p: str | os.PathLike) -> str:
        return Path(p).as_posix()

    def venv_bin_dir(self, venv: Path) -> Path:
        return Path(venv) / "Scripts"

    def venv_exe(self, venv: Path, name: str) -> Path:
        return self.venv_bin_dir(venv) / f"{name}.exe"

    def popen_new_group_kwargs(self) -> dict:
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP}

    def pid_alive(self, pid: int) -> bool:
        try:
            import psutil
        except ImportError:
            return True  # cannot determine without psutil; assume alive
        return psutil.pid_exists(pid)

    def kill_pid(self, pid: int) -> None:
        import signal

        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    def reexec(self, path: Path, argv: list[str]) -> NoReturn:
        # os.execv orphans the console and mangles exit-code propagation on Windows; run the target
        # as a child and exit with its return code instead.
        completed = subprocess.run([str(path), *argv])
        sys.exit(completed.returncode)


def _detect_platform() -> Platform:
    if sys.platform == "win32":
        return WindowsPlatform()
    if _detect_wsl():
        return WslPlatform()
    return Platform()


#: The single, process-wide platform strategy. The only place ``sys.platform`` is branched on.
PLATFORM: Platform = _detect_platform()

# Backward-compatible module flags (read from the resolved strategy).
IS_WINDOWS = PLATFORM.is_windows
IS_WSL = PLATFORM.is_wsl


# ============================================================================ module-level API
# Thin delegations to PLATFORM so existing imports keep working.


def venv_bin_dir(venv: Path) -> Path:
    """Return the executables directory inside *venv*: ``Scripts`` (Windows) or ``bin``."""
    return PLATFORM.venv_bin_dir(venv)


def venv_exe(venv: Path, name: str) -> Path:
    """Path to executable *name* inside *venv*, with ``.exe`` appended on Windows."""
    return PLATFORM.venv_exe(venv, name)


def popen_kwargs_new_group() -> dict:
    """``subprocess.Popen`` kwargs to start the child in its own group/session."""
    return PLATFORM.popen_new_group_kwargs()


def pid_alive(pid: int | None) -> bool:
    """Best-effort check whether process *pid* is currently running."""
    if pid is None:
        return False
    return PLATFORM.pid_alive(pid)


def terminate_process_tree(pid: int | None, *, grace: float = 5.0) -> None:
    """Terminate *pid* and all descendants: graceful first, then forced.

    Uses ``psutil`` (children-first ``terminate()`` then ``kill()``) when available; falls back to
    the platform's plain ``kill_pid`` when psutil is missing.
    """
    if pid is None:
        return
    try:
        import psutil
    except ImportError:
        PLATFORM.kill_pid(pid)
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


def reexec(path: Path, argv: list[str]) -> NoReturn:
    """Hand off to *path* + *argv*: ``execv`` on POSIX, run-then-``sys.exit`` on Windows."""
    PLATFORM.reexec(Path(path), argv)


def find_free_port(start: int = 29500, count: int = 101, host: str = "127.0.0.1") -> int:
    """First bindable TCP port in ``[start, start + count)``; falls back to *start*.

    Cross-platform replacement for parsing ``ss``/``netstat`` output: probe by binding.
    """
    for port in range(start, start + count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if PLATFORM.free_port_reuseaddr:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    return start


def free_port_owned_by(port: int, patterns: tuple[str, ...]) -> None:
    """Free *port* if held by one of *our* processes; raise otherwise.

    If a listener on *port* has a command line matching any of *patterns*, terminate its process
    tree. If held by an unrelated process, raise ``SystemExit``. No-op when the port is free or
    psutil is unavailable.
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

    On WSL2/WDDM the expandable-segments allocator uses CUDA's virtual-memory APIs (``cuMemMap`` /
    ``cuMemSetAccess``). Those raise ``CUDA driver error: device not ready`` when cuDNN allocates
    convolution workspace, so any conv model (e.g. SDXL's UNet) crashes early in the backward pass.
    This forces ``expandable_segments:False`` regardless of what the shell or ``[training.env]`` set,
    and fills in low-fragmentation defaults, preserving any knob the user set explicitly.

    MUST run before the first ``import torch`` in the process — the caching allocator parses
    ``PYTORCH_CUDA_ALLOC_CONF`` when torch is imported. ``rengu_flow/__init__`` calls it for exactly
    that reason. Returns the resulting env value, or ``None`` when not on WSL.
    """
    env = os.environ if env is None else env
    is_wsl = PLATFORM.is_wsl if is_wsl is None else is_wsl
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
